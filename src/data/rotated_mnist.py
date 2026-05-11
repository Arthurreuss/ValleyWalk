"""Rotated MNIST continual-learning benchmark.

N tasks, each rotating every MNIST image by a fixed angle.
Rotations are evenly spaced over [0°, 180°) with step = 180 / num_tasks:

    num_tasks=20 → 9° steps:  [-90°, -81°, …, 81°]
    num_tasks=5  → 36° steps: [-90°, -54°, -18°, 18°, 54°]
    num_tasks=3  → 60° steps: [-90°, -30°, 30°]

A custom schedule can be provided via ``rotations_deg`` in the dataset config.

Usage:
    from src.data.rotated_mnist import RotatedMNIST

    dataset = RotatedMNIST(cfg, batch_size=64)
    train_loader, test_loader = dataset.get_task_loaders(task_id=0)  # 0°
    train_loader, test_loader = dataset.get_task_loaders(task_id=1)  # 180/num_tasks °

Batch shapes: (batch_size, 784) for both x tensors, (batch_size,) for labels.
"""

from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import torchvision
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF


# ---------------------------------------------------------------------------
# Internal per-task dataset
# ---------------------------------------------------------------------------

class _RotatedMNISTTask(Dataset):
    """Wraps a base MNIST dataset and applies a fixed rotation + flatten.

    The base dataset must already apply ``transforms.ToTensor()``, so each
    item arrives as a float32 tensor of shape ``(1, 28, 28)`` in [0, 1].

    Args:
        base_dataset: Torchvision MNIST dataset (ToTensor already applied).
        angle_deg:    Clockwise rotation angle in degrees.
    """

    def __init__(self, base_dataset: Dataset, angle_deg: float) -> None:
        self.base_dataset = base_dataset
        self.angle_deg = float(angle_deg)

    def __len__(self) -> int:
        return len(self.base_dataset)  # type: ignore[arg-type]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """Return (x, label) where x has shape (784,).

        Args:
            idx: Sample index.

        Returns:
            x:     Flattened float32 tensor, shape (784,).
            label: Integer class label in [0, 9].
        """
        img, label = self.base_dataset[idx]  # img: (1, 28, 28)

        # Apply rotation only when angle is non-zero (avoids a no-op call).
        if self.angle_deg != 0.0:
            img = TF.rotate(img, self.angle_deg)  # still (1, 28, 28)

        # Flatten to 784 as required by the MLP backbone.
        x = img.view(-1)  # (784,)
        return x, label


# ---------------------------------------------------------------------------
# Public class
# ---------------------------------------------------------------------------

class RotatedMNIST:
    """Rotated MNIST continual-learning dataset manager.

    Loads MNIST once and creates per-task loaders on demand by applying
    ``torchvision.transforms.functional.rotate``.

    Args:
        cfg:         Dataset config object (Hydra OmegaConf).  Must expose:
                       - num_tasks (int)
                       - rotations_deg (list[float] | None)
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

        # -----------------------------------------------------------------
        # Determine rotation schedule
        # -----------------------------------------------------------------
        if cfg.rotations_deg is None:
            # Evenly space num_tasks rotations over [-90°, +90°).
            # Step = 180 / num_tasks, so the spacing scales with task count:
            #   num_tasks=20 → 9° steps  [-90, -81, …, 81]
            #   num_tasks=5  → 36° steps [-90, -54, -18, 18, 54]
            #   num_tasks=3  → 60° steps [-90, -30, 30]
            step = 180.0 / self.num_tasks
            self.rotations_deg: List[float] = [
                float(-90.0 + step * t) for t in range(self.num_tasks)
            ]
        else:
            self.rotations_deg = [float(a) for a in cfg.rotations_deg]

        if len(self.rotations_deg) != self.num_tasks:
            raise ValueError(
                f"rotations_deg has {len(self.rotations_deg)} entries but "
                f"num_tasks={self.num_tasks}."
            )

        # -----------------------------------------------------------------
        # Load MNIST once; shared across all tasks (only the rotation varies)
        # -----------------------------------------------------------------
        to_tensor = transforms.ToTensor()
        self._train_base = torchvision.datasets.MNIST(
            root=self.download_dir,
            train=True,
            download=True,
            transform=to_tensor,
        )
        self._test_base = torchvision.datasets.MNIST(
            root=self.download_dir,
            train=False,
            download=True,
            transform=to_tensor,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_task_loaders(
        self, task_id: int
    ) -> Tuple[DataLoader, DataLoader]:
        """Return (train_loader, test_loader) for task ``task_id``.

        Both loaders yield batches of ``(x, labels)`` where:
        - ``x`` has shape ``(batch_size, 784)``, float32 in [0, 1].
        - ``labels`` has shape ``(batch_size,)``, dtype int64.

        Args:
            task_id: Zero-based task index in [0, num_tasks).

        Returns:
            train_loader: Shuffled loader over the 60 000 MNIST training images.
            test_loader:  Un-shuffled loader over the 10 000 MNIST test images.

        Raises:
            ValueError: If task_id is out of range.
        """
        if not (0 <= task_id < self.num_tasks):
            raise ValueError(
                f"task_id must be in [0, {self.num_tasks}), got {task_id}."
            )

        angle = self.rotations_deg[task_id]

        train_ds = _RotatedMNISTTask(self._train_base, angle)
        test_ds  = _RotatedMNISTTask(self._test_base,  angle)

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

    def angle_for_task(self, task_id: int) -> float:
        """Return the rotation angle (degrees) for a given task.

        Args:
            task_id: Zero-based task index.

        Returns:
            Rotation angle in degrees.
        """
        return self.rotations_deg[task_id]
