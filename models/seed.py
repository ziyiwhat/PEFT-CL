import sys
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader

from models.base import BaseLearner

SEED_ROOT = Path(__file__).resolve().parents[1] / "third_party/seed"
if str(SEED_ROOT) not in sys.path:
    sys.path.append(str(SEED_ROOT))

from third_party.seed.approach.seed import Appr as SeedApproach
from third_party.seed.networks.network import ExtractorEnsemble
from third_party.seed.networks.resnet32_linear_turbo import resnet32


class _SeedDatasetWrapper(Dataset):
    """Wrapper to strip index from DummyDataset's (idx, image, label) return."""
    def __init__(self, base_dataset):
        self.base_dataset = base_dataset
    
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        _, image, label = self.base_dataset[idx]
        return image, label
    
    def __getattr__(self, name):
        """Forward attribute access to the underlying dataset."""
        return getattr(self.base_dataset, name)


class Learner(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self.seed_model = None
        self.seed_appr = None
        self.taskcla = None
        self.test_loader = None
        self.batch_size = args.get("batch_size", 128)
        self.num_workers = args.get("num_workers", 4)
        self.seed_params = {
            "nepochs": args.get("seed_nepochs", 100),
            "ftepochs": args.get("seed_ftepochs", 50),
            "lr": args.get("seed_lr", 0.05),
            "lr_min": args.get("seed_lr_min", 1e-4),
            "lr_factor": args.get("seed_lr_factor", 3),
            "lr_patience": args.get("seed_lr_patience", 5),
            "clipgrad": args.get("seed_clipgrad", 10000),
            "momentum": args.get("seed_momentum", 0.9),
            "wd": args.get("seed_weight_decay", 0.0),
            "ftwd": args.get("seed_ft_weight_decay", 0.0),
            "max_experts": args.get("seed_max_experts", 5),
            "gmms": args.get("seed_gmms", 1),
            "alpha": args.get("seed_alpha", 0.99),
            "tau": args.get("seed_tau", 3.0),
            "shared": args.get("seed_shared", 0),
            "use_multivariate": args.get("seed_use_multivariate", True),
            "use_nmc": args.get("seed_use_nmc", False),
            "initialization_strategy": args.get("seed_init_strategy", "first"),
            "compensate_drifts": args.get("seed_compensate_drifts", False),
        }
        self.SeedApproachCls = SeedApproach
        self.EnsembleCls = ExtractorEnsemble
        self.ResNet32Fn = resnet32

    def _initialize_seed_components(self, data_manager):
        if self.seed_model is not None:
            return
        taskcla = [(task_idx, data_manager.get_task_size(task_idx)) for task_idx in range(data_manager.nb_tasks)]
        backbone = self.ResNet32Fn(num_features=64, num_classes=taskcla[0][1])
        self.seed_model = self.EnsembleCls(backbone, taskcla, network_type="resnet32", device=self._device)
        self.seed_model.to(self._device)
        self.seed_appr = self.SeedApproachCls(self.seed_model, device=self._device, **self.seed_params)
        self.taskcla = taskcla
        self._network = self.seed_model

    def incremental_train(self, data_manager):
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)
        self._initialize_seed_components(data_manager)

        train_dataset_raw = data_manager.get_dataset(
            list(range(self._known_classes, self._total_classes)),
            source="train",
            mode="train",
        )
        setattr(train_dataset_raw, "transform", train_dataset_raw.trsf)
        train_dataset = _SeedDatasetWrapper(train_dataset_raw)
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

        val_dataset_raw = data_manager.get_dataset(
            list(range(self._known_classes, self._total_classes)),
            source="train",
            mode="test",
        )
        setattr(val_dataset_raw, "transform", val_dataset_raw.trsf)
        val_dataset = _SeedDatasetWrapper(val_dataset_raw)
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

        test_dataset_raw = data_manager.get_dataset(
            list(range(0, self._total_classes)),
            source="test",
            mode="test",
        )
        setattr(test_dataset_raw, "transform", test_dataset_raw.trsf)
        test_dataset = _SeedDatasetWrapper(test_dataset_raw)
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

        self.seed_appr.train_loop(self._cur_task, train_loader, val_loader)

    def eval_task(self):
        if self.test_loader is None:
            return None, None
        loss, acc_taw, acc_tag = self.seed_appr.eval(self._cur_task, self.test_loader)
        acc = round(acc_tag * 100, 2)
        grouped = {"total": acc, f"task-{self._cur_task}": acc, "loss": round(loss, 4)}
        metrics = {"grouped": grouped, "top1": acc, "top5": acc}
        return metrics, None

    def after_task(self):
        self._known_classes = self._total_classes

