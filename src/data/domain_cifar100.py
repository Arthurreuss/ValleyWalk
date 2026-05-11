"""Domain CIFAR-100 continual-learning benchmark.

5 tasks, each applying a different image corruption to every CIFAR-100 image.
All tasks share the same 100 classes — this is a Domain-IL setting (no class
split; only the input distribution shifts).

Corruptions are applied on-the-fly in __getitem__ using the `imagecorruptions`
library, which operates on uint8 numpy arrays of shape (H, W, C).

Standard CIFAR-100 normalisation is applied after corruption:
    mean = (0.5071, 0.4867, 0.4408)
    std  = (0.2675, 0.2565, 0.2761)

Usage:
    from src.data.domain_cifar100 import DomainCIFAR100

    dataset = DomainCIFAR100(cfg, batch_size=64)
    train_loader, test_loader = dataset.get_task_loaders(task_id=0)
    # → batches of shape (batch, 3, 32, 32), labels in [0, 99]
"""

from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import torchvision
import torchvision.transforms as transforms

from imagecorruptions import corrupt


# ---------------------------------------------------------------------------
# CIFAR-100 normalisation constants (computed from the training set)
# ---------------------------------------------------------------------------
_CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
_CIFAR100_STD  = (0.2675, 0.2565, 0.2761)


# ---------------------------------------------------------------------------
# Internal per-task dataset
# ---------------------------------------------------------------------------

class _CorruptedCIFAR100Task(Dataset):
    """Wraps a base CIFAR-100 dataset and applies a fixed corruption + normalise.

    The base dataset must return raw PIL Images (``transform=None``), so this
    class controls the full pipeline:
        PIL → numpy uint8 → corrupt → PIL → ToTensor → Normalize

    Args:
        base_dataset:    Torchvision CIFAR100 dataset with transform=None.
        corruption_name: Name of the corruption to apply (e.g. "gaussian_noise").
        severity:        Corruption severity level in [1, 5].
        normalise:       torchvision transform applied after corruption
                         (typically ToTensor + Normalize).
    """

    def __init__(
        self,
        base_dataset: Dataset,
        corruption_name: str,
        severity: int,
        normalise: transforms.Compose,
    ) -> None:
        self.base_dataset = base_dataset
        self.corruption_name = corruption_name
        self.severity = severity
        self.normalise = normalise

    def __len__(self) -> int:
        return len(self.base_dataset)  # type: ignore[arg-type]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """Return (x, label) where x has shape (3, 32, 32), normalised.

        Args:
            idx: Sample index.

        Returns:
            x:     Float32 tensor, shape (3, 32, 32), normalised to CIFAR-100 stats.
            label: Integer class label in [0, 99].
        """
        img_pil, label = self.base_dataset[idx]   # PIL Image, HWC uint8

        if self.corruption_name == "none":
            # Clean task — no corruption, just normalise the raw PIL image.
            x = self.normalise(img_pil)  # (3, 32, 32)
            return x, label

        # PIL → numpy uint8 (H, W, C) — required by imagecorruptions
        img_np = np.array(img_pil, dtype=np.uint8)

        # Apply corruption in-place (returns uint8 numpy array)
        img_corrupted = corrupt(
            img_np,
            corruption_name=self.corruption_name,
            severity=self.severity,
        )

        # numpy → PIL → Tensor (ToTensor converts uint8 to float [0,1]) → Normalize
        img_back = Image.fromarray(img_corrupted)
        x = self.normalise(img_back)   # (3, 32, 32)
        return x, label


# ---------------------------------------------------------------------------
# Public class
# ---------------------------------------------------------------------------

class DomainCIFAR100:
    """Domain CIFAR-100 continual-learning dataset manager.

    Loads CIFAR-100 once (no transform) and creates per-task loaders on demand
    by wrapping the base dataset in ``_CorruptedCIFAR100Task``.

    Args:
        cfg:         Dataset config object (Hydra OmegaConf).  Must expose:
                       - num_tasks (int)
                       - corruption_types (list[str])
                       - severity (int)
                       - download_dir (str)
        batch_size:  Batch size for the returned DataLoaders.
        num_workers: Number of worker processes for DataLoaders.
                     Defaults to 0 (main process only) for reproducibility.
    """

    def __init__(
        self,
        cfg,
        batch_size: int = 64,
        num_workers: int = 0,
    ) -> None:
        self.num_tasks: int = cfg.num_tasks
        self.download_dir: str = cfg.download_dir
        self.batch_size: int = batch_size
        self.num_workers: int = num_workers
        self.severity: int = int(cfg.severity)
        self.corruption_types: List[str] = list(cfg.corruption_types)

        if len(self.corruption_types) != self.num_tasks:
            raise ValueError(
                f"corruption_types has {len(self.corruption_types)} entries but "
                f"num_tasks={self.num_tasks}."
            )

        # -----------------------------------------------------------------
        # Normalisation transform applied after corruption
        # -----------------------------------------------------------------
        self._normalise = transforms.Compose([
            transforms.ToTensor(),                          # uint8 PIL → float [0,1]
            transforms.Normalize(_CIFAR100_MEAN, _CIFAR100_STD),
        ])

        # -----------------------------------------------------------------
        # Load CIFAR-100 once with no transform (raw PIL images).
        # Corruption + normalisation are applied per sample in __getitem__.
        # -----------------------------------------------------------------
        self._train_base = torchvision.datasets.CIFAR100(
            root=self.download_dir,
            train=True,
            download=True,
            transform=None,
        )
        self._test_base = torchvision.datasets.CIFAR100(
            root=self.download_dir,
            train=False,
            download=True,
            transform=None,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_task_loaders(
        self, task_id: int
    ) -> Tuple[DataLoader, DataLoader]:
        """Return (train_loader, test_loader) for task ``task_id``.

        Both loaders yield batches of ``(x, labels)`` where:
        - ``x`` has shape ``(batch_size, 3, 32, 32)``, float32, normalised.
        - ``labels`` has shape ``(batch_size,)``, dtype int64, in [0, 99].

        Args:
            task_id: Zero-based task index in [0, num_tasks).

        Returns:
            train_loader: Shuffled loader over 50 000 CIFAR-100 training images.
            test_loader:  Un-shuffled loader over 10 000 CIFAR-100 test images.

        Raises:
            ValueError: If task_id is out of range.
        """
        if not (0 <= task_id < self.num_tasks):
            raise ValueError(
                f"task_id must be in [0, {self.num_tasks}), got {task_id}."
            )

        corruption = self.corruption_types[task_id]

        train_ds = _CorruptedCIFAR100Task(
            self._train_base, corruption, self.severity, self._normalise
        )
        test_ds = _CorruptedCIFAR100Task(
            self._test_base, corruption, self.severity, self._normalise
        )

        train_loader = DataLoader(
            train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=False,
        )
        test_loader = DataLoader(
            test_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=False,
        )

        return train_loader, test_loader

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def corruption_for_task(self, task_id: int) -> str:
        """Return the corruption name for a given task.

        Args:
            task_id: Zero-based task index.

        Returns:
            Corruption name string (e.g. ``"gaussian_noise"``).
        """
        return self.corruption_types[task_id]
