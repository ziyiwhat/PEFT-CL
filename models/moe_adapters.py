import logging
import math
from copy import deepcopy
from typing import List

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets

from models.base import BaseLearner
from third_party.moe_clip import clip


class _ClipDataset(Dataset):
    def __init__(self, images, labels, preprocess, use_path=False):
        self.images = images
        self.labels = labels.astype(np.int64)
        self.preprocess = preprocess
        self.use_path = use_path

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = self.images[index]
        if self.use_path:
            image = Image.open(image).convert("RGB")
        else:
            if isinstance(image, Image.Image):
                pass
            elif isinstance(image, np.ndarray):
                if image.ndim == 2:
                    image = np.stack([image] * 3, axis=-1)
                image = Image.fromarray(image)
            else:
                raise ValueError("Unsupported image type for CLIP preprocessing.")
        tensor = self.preprocess(image)
        label = int(self.labels[index])
        return tensor, label


def _load_class_names(dataset_name: str, data_root: str) -> List[str]:
    dataset_name = dataset_name.lower()
    if "cifar100" in dataset_name:
        return datasets.CIFAR100(root=data_root, train=True, download=True).classes
    if "cifar10" in dataset_name:
        return datasets.CIFAR10(root=data_root, train=True, download=True).classes
    raise ValueError(f"No class-name mapping implemented for dataset `{dataset_name}`.")


class Learner(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        device = self._device
        clip_name = args.get("clip_model_name", "ViT-B/16")
        clip_pretrained = args.get("clip_pretrained", True)
        clip_model, train_preprocess, eval_preprocess = clip.load(
            clip_name, device=device, jit=False, pretrained=clip_pretrained
        )
        self.clip_model = clip_model.to(device)
        self.clip_model.eval()
        self._network = self.clip_model  # for logging utilities
        self.train_preprocess = train_preprocess
        self.eval_preprocess = eval_preprocess

        self.prompt_template = args.get("prompt_template", "a photo of a {}.")
        self.lr = args.get("adapter_lr", 1e-3)
        self.weight_decay = args.get("adapter_weight_decay", 0.0)
        self.label_smoothing = args.get("adapter_label_smoothing", 0.0)
        self.adapter_epochs = args.get("adapter_epochs", 1)
        self.batch_size = args.get("batch_size", 128)
        self.num_workers = args.get("num_workers", 8)
        self.data_path = args.get("data_path", "./data")

        self.class_order = None
        self.ordered_class_names = None
        self.text_tokens = None

        dataset_name = args.get("dataset", "cifar100")
        self.base_class_names = _load_class_names(dataset_name, self.data_path)

    def incremental_train(self, data_manager):
        if self.class_order is None:
            self.class_order = deepcopy(data_manager._class_order)
            self.ordered_class_names = [self.base_class_names[idx] for idx in self.class_order]

        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)
        logging.info("Learning on {}-{}".format(self._known_classes, self._total_classes))

        self._ensure_router_capacity(self._cur_task)

        train_loader = self._build_dataloader(
            data_manager,
            np.arange(self._known_classes, self._total_classes),
            source="train",
            preprocess=self.train_preprocess,
            mode="train",
            shuffle=True,
        )
        self.test_loader = self._build_dataloader(
            data_manager,
            np.arange(0, self._total_classes),
            source="test",
            preprocess=self.eval_preprocess,
            mode="test",
            shuffle=False,
        )

        current_class_names = self.ordered_class_names[self._known_classes : self._total_classes]
        self._train_task(train_loader, current_class_names)
        self._update_text_tokens()

    def _ensure_router_capacity(self, task_id):
        for module in self.clip_model.modules():
            router_list = getattr(module, "router_list", None)
            w_noise_list = getattr(module, "w_noise_list", None)
            if router_list is None or w_noise_list is None:
                continue
            while len(router_list) <= task_id:
                base_weight = router_list[0]
                dim, experts = base_weight.shape
                device = base_weight.device
                router_list.append(torch.nn.Parameter(torch.zeros(dim, experts, device=device)))
                w_noise_list.append(torch.nn.Parameter(torch.zeros(dim, experts, device=device)))

    def _build_dataloader(self, data_manager, indices, source, preprocess, mode, shuffle):
        data, targets, dataset = data_manager.get_dataset(
            indices, source=source, mode=mode, ret_data=True
        )
        clip_dataset = _ClipDataset(data, targets, preprocess, getattr(dataset, "use_path", False))
        return DataLoader(
            clip_dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def _train_task(self, train_loader, class_names):
        if len(train_loader) == 0:
            return

        for param in self.clip_model.parameters():
            param.requires_grad = False

        trainable_keywords = ("adaptmlp", "router", "noise")
        trainable_params = []
        for name, param in self.clip_model.named_parameters():
            if any(k in name for k in trainable_keywords):
                param.requires_grad = True
                trainable_params.append(param)

        optimizer = torch.optim.AdamW(trainable_params, lr=self.lr, weight_decay=self.weight_decay)
        total_iterations = max(len(train_loader) * max(1, self.adapter_epochs), 1)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_iterations)

        texts = clip.tokenize([self.prompt_template.format(name) for name in class_names]).to(self._device)

        self.clip_model.train()
        iteration = 0
        for _ in range(self.adapter_epochs):
            for inputs, targets in train_loader:
                inputs = inputs.to(self._device, non_blocking=True)
                targets = targets.to(self._device, non_blocking=True) - self._known_classes
                logits, _ = self.clip_model(inputs, texts, taskid=self._cur_task, is_train=True)
                loss = F.cross_entropy(logits, targets, label_smoothing=self.label_smoothing)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()
                iteration += 1

        self.clip_model.eval()

    def _update_text_tokens(self):
        seen_names = self.ordered_class_names[: self._total_classes]
        prompts = [self.prompt_template.format(name) for name in seen_names]
        self.text_tokens = clip.tokenize(prompts).to(self._device)

    def eval_task(self):
        if self.text_tokens is None or self.test_loader is None:
            return None, None

        y_pred, y_true = self._gather_predictions(self.test_loader, self.text_tokens)
        cnn_accy = self._evaluate(y_pred, y_true)
        return cnn_accy, None

    def _gather_predictions(self, loader, text_tokens):
        preds = []
        targets = []
        with torch.no_grad():
            for images, labels in loader:
                images = images.to(self._device, non_blocking=True)
                logits, _ = self.clip_model(images, text_tokens, taskid=self._cur_task, is_train=False)
                topk = torch.topk(logits, k=self.topk, dim=1)[1].cpu().numpy()
                preds.append(topk)
                targets.append(labels.numpy())
        return np.concatenate(preds), np.concatenate(targets)

