import logging
import math
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
from sklearn.cluster import KMeans
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from backbone.sinet_inflora import SiNet
from backbone.vit_inflora import Attention_LoRA
from models.base import BaseLearner
from utils.schedulers import CosineSchedule
from utils.toolkit import tensor2numpy


class Learner(BaseLearner):
    def __init__(self, args):
        super().__init__(args)

        # Align total sessions / rank defaults if missing
        args.setdefault("total_sessions", args.get("nb_tasks", 1))
        args.setdefault("rank", 10)
        args.setdefault("num_workers", 8)

        self._network = SiNet(args)
        for module in self._network.modules():
            if isinstance(module, Attention_LoRA):
                module.init_param()

        self.optim = args.get("optim", "adam")
        self.EPSILON = args.get("EPSILON", 1e-8)
        self.init_epoch = args.get("init_epoch", 20)
        self.init_lr = args.get("init_lr", 5e-4)
        self.init_lr_decay = args.get("init_lr_decay", 0.1)
        self.init_weight_decay = args.get("init_weight_decay", 0.0)
        self.epochs = args.get("epochs", 20)
        self.lrate = args.get("lrate", 5e-4)
        self.lrate_decay = args.get("lrate_decay", 0.1)
        self.batch_size = args.get("batch_size", 128)
        self.weight_decay = args.get("weight_decay", 0.0)
        self.num_workers = args.get("num_workers", 8)
        self.lamb = args.get("lamb", 0.95)
        self.lame = args.get("lame", 1.0)
        self.total_sessions = args.get("total_sessions", args.get("nb_tasks", 1))
        self.dataset = args.get("dataset", "cifar224")

        self.topk = 1
        self.class_num = self._network.class_num
        self.debug = False

        self.all_keys = []
        self.feature_list = []
        self.project_type = []

    def after_task(self):
        self._known_classes = self._total_classes
        logging.info("Exemplar size: {}".format(self.exemplar_size))

    def incremental_train(self, data_manager):
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)
        self._network.update_fc(self._total_classes)

        logging.info("Learning on {}-{}".format(self._known_classes, self._total_classes))

        train_dataset = data_manager.get_dataset(
            np.arange(self._known_classes, self._total_classes), source="train", mode="train"
        )
        self.train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers
        )
        test_dataset = data_manager.get_dataset(np.arange(0, self._total_classes), source="test", mode="test")
        self.test_loader = DataLoader(
            test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers
        )

        self._train(self.train_loader, self.test_loader)
        self.clustering(self.train_loader)

    def _train(self, train_loader, test_loader):
        self._network.to(self._device)

        for name, param in self._network.named_parameters():
            param.requires_grad_(False)
            try:
                if f"classifier_pool.{self._network.module.numtask - 1}" in name:
                    param.requires_grad_(True)
                if f"lora_B_k.{self._network.module.numtask - 1}" in name:
                    param.requires_grad_(True)
                if f"lora_B_v.{self._network.module.numtask - 1}" in name:
                    param.requires_grad_(True)
            except AttributeError:
                if f"classifier_pool.{self._network.numtask - 1}" in name:
                    param.requires_grad_(True)
                if f"lora_B_k.{self._network.numtask - 1}" in name:
                    param.requires_grad_(True)
                if f"lora_B_v.{self._network.numtask - 1}" in name:
                    param.requires_grad_(True)

        enabled = {name for name, param in self._network.named_parameters() if param.requires_grad}

        with torch.no_grad():
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs = inputs.to(self._device)
                self._network(inputs, get_cur_feat=True)
            if self._cur_task == 0:
                for module in self._network.modules():
                    if isinstance(module, Attention_LoRA):
                        cur_matrix = module.cur_matrix
                        u, _, _ = torch.linalg.svd(cur_matrix)
                        module.lora_A_k[self._cur_task].weight.data.copy_(u[:, : module.rank].T / math.sqrt(3))
                        module.lora_A_v[self._cur_task].weight.data.copy_(u[:, : module.rank].T / math.sqrt(3))
                        module.cur_matrix.zero_()
                        module.n_cur_matrix = 0
            else:
                kk = 0
                for module in self._network.modules():
                    if isinstance(module, Attention_LoRA):
                        cur_matrix = module.cur_matrix
                        if self.project_type[kk] == "remove":
                            cur_matrix = cur_matrix - torch.mm(self.feature_mat[kk], cur_matrix)
                        else:
                            cur_matrix = torch.mm(self.feature_mat[kk], cur_matrix)
                        u, _, _ = torch.linalg.svd(cur_matrix, full_matrices=False)
                        module.lora_A_k[self._cur_task].weight.data.copy_(u[:, : module.rank].T / math.sqrt(3))
                        module.lora_A_v[self._cur_task].weight.data.copy_(u[:, : module.rank].T / math.sqrt(3))
                        module.cur_matrix.zero_()
                        module.n_cur_matrix = 0
                        kk += 1

        logging.info(f"Parameters to be updated: {enabled}")

        if len(self._multiple_gpus) > 1:
            self._network = nn.DataParallel(self._network, self._multiple_gpus)

        if self._cur_task == 0:
            optimizer, scheduler, run_epoch = self._build_optimizer(self.init_lr, self.init_weight_decay, self.init_epoch)
        else:
            optimizer, scheduler, run_epoch = self._build_optimizer(self.lrate, self.weight_decay, self.epochs)

        self.run_epoch = run_epoch
        self.train_function(train_loader, test_loader, optimizer, scheduler)

        if len(self._multiple_gpus) > 1:
            self._network = self._network.module

        with torch.no_grad():
            for _, (_, inputs, _) in enumerate(train_loader):
                inputs = inputs.to(self._device)
                self._network(inputs, get_cur_feat=True)

            mat_list = []
            for module in self._network.modules():
                if isinstance(module, Attention_LoRA):
                    mat_list.append(deepcopy(module.cur_matrix))
                    module.cur_matrix.zero_()
                    module.n_cur_matrix = 0
            self.update_DualGPM(mat_list)

            self.feature_mat = []
            for idx, feat in enumerate(self.feature_list):
                if isinstance(feat, torch.Tensor):
                    feat_np = feat.detach().cpu().numpy()
                else:
                    feat_np = np.asarray(feat)

                uf = torch.from_numpy(np.dot(feat_np, feat_np.T)).float()
                logging.info("Layer {} - Projection Matrix shape: {}".format(idx + 1, uf.shape))
                self.feature_mat.append(uf)

    def _build_optimizer(self, lr, weight_decay, epochs):
        if self.optim == "sgd":
            optimizer = optim.SGD(
                self._network.parameters(),
                momentum=0.9,
                lr=lr,
                weight_decay=weight_decay,
            )
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=epochs)
        elif self.optim == "adam":
            optimizer = optim.Adam(
                self._network.parameters(),
                lr=lr,
                weight_decay=weight_decay,
                betas=(0.9, 0.999),
            )
            scheduler = CosineSchedule(optimizer=optimizer, K=epochs)
        else:
            raise ValueError(f"Unsupported optimizer: {self.optim}")
        return optimizer, scheduler, epochs

    def train_function(self, train_loader, test_loader, optimizer, scheduler):
        prog_bar = tqdm(range(self.run_epoch))
        for _, epoch in enumerate(prog_bar):
            self._network.eval()
            losses = 0.0
            correct, total = 0, 0

            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                mask = (targets >= self._known_classes).nonzero().view(-1)
                if mask.numel() == 0:
                    continue
                inputs = torch.index_select(inputs, 0, mask)
                targets = torch.index_select(targets, 0, mask) - self._known_classes

                logits = self._network(inputs)["logits"]
                loss = F.cross_entropy(logits, targets)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()

                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)
                if self.debug and i > 10:
                    break

            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / max(total, 1), decimals=2)
            info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}".format(
                self._cur_task, epoch + 1, self.run_epoch, losses / max(len(train_loader), 1), train_acc
            )
            prog_bar.set_description(info)

        logging.info(info)

    def clustering(self, dataloader):
        features = []
        for _, (_, inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(self._device), targets.to(self._device)
            mask = (targets >= self._known_classes).nonzero().view(-1)
            if mask.numel() == 0:
                continue
            inputs = torch.index_select(inputs, 0, mask)
            with torch.no_grad():
                feature = self._network.extract_vector(inputs)
            feature = feature / feature.norm(dim=-1, keepdim=True)
            features.append(feature)
        if not features:
            return
        features = torch.cat(features, 0).cpu().detach().numpy()
        clustering = KMeans(n_clusters=5, random_state=0).fit(features)
        self.all_keys.append(torch.tensor(clustering.cluster_centers_).to(self._device))

    def update_DualGPM(self, mat_list):
        threshold = (self.lame - self.lamb) * self._cur_task / max(self.total_sessions, 1) + self.lamb
        logging.info("Threshold: %s", threshold)
        if len(self.feature_list) == 0:
            for activation in mat_list:
                u, s, _ = np.linalg.svd(activation, full_matrices=False)
                sval_total = (s**2).sum()
                sval_ratio = (s**2) / sval_total if sval_total > 0 else s
                r = np.sum(np.cumsum(sval_ratio) < threshold)
                basis = u[:, : max(r, 1)]
                self.feature_list.append(basis)
                self.project_type.append("remove" if r < (activation.shape[0] / 2) else "retain")
        else:
            for i, activation in enumerate(mat_list):
                if self.project_type[i] == "remove":
                    _, s1, _ = np.linalg.svd(activation, full_matrices=False)
                    sval_total = (s1**2).sum()
                    act_hat = activation - np.dot(np.dot(self.feature_list[i], self.feature_list[i].transpose()), activation)
                    u, s, _ = np.linalg.svd(act_hat, full_matrices=False)
                    sval_ratio = (s**2) / sval_total if sval_total > 0 else s
                    accumulated = (sval_total - (s**2).sum()) / sval_total if sval_total > 0 else 0
                    r = 0
                    for val in sval_ratio:
                        if accumulated < threshold:
                            accumulated += val
                            r += 1
                        else:
                            break
                    if r == 0:
                        logging.info("Skip Updating DualGPM for layer: %s", i + 1)
                        continue
                    ui = np.hstack((self.feature_list[i], u[:, 0:r]))
                    if ui.shape[1] > ui.shape[0]:
                        self.feature_list[i] = ui[:, : ui.shape[0]]
                    else:
                        self.feature_list[i] = ui
                else:
                    _, s1, _ = np.linalg.svd(activation, full_matrices=False)
                    sval_total = (s1**2).sum()
                    act_hat = np.dot(np.dot(self.feature_list[i], self.feature_list[i].transpose()), activation)
                    u, s, _ = np.linalg.svd(act_hat, full_matrices=False)
                    sval_ratio = (s**2) / sval_total if sval_total > 0 else s
                    accumulated = (s**2).sum() / sval_total if sval_total > 0 else 0
                    r = 0
                    for val in sval_ratio:
                        if accumulated >= (1 - threshold):
                            accumulated -= val
                            r += 1
                        else:
                            break
                    if r == 0:
                        logging.info("Skip Updating DualGPM for layer: %s", i + 1)
                        continue

                    act_feature = self.feature_list[i] - np.dot(
                        np.dot(u[:, 0:r], u[:, 0:r].transpose()), self.feature_list[i]
                    )
                    ui, _, _ = np.linalg.svd(act_feature)
                    self.feature_list[i] = ui[:, : self.feature_list[i].shape[1] - r]

        logging.info("-" * 40)
        logging.info("Gradient Constraints Summary")
        logging.info("-" * 40)
        for idx, feat in enumerate(self.feature_list):
            logging.info("%d layer constraint size: %d", idx + 1, feat.shape[1])

