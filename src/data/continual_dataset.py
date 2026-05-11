"""Continual dataset wrapper: task iterator + replay buffer management.

`ContinualDataset` wraps `RotatedMNIST`, `DomainCIFAR10`, or `DomainCIFAR100` and
provides a unified interface for the training loop:

1. `task_iterator()` — a generator that yields one task at a time.

2. `sample_replay(batch_size)` — draws a uniform random batch from the buffer,
   which grows to hold samples from all completed tasks.

Buffer population is the responsibility of each method's `end_task()` call in
the training loop (ER, GEM, CACL all do this).  The dataset does NOT populate
the buffer — doing so would double-fill it and inflate the reservoir sampler's
`_n_seen` counter, biasing the buffer toward early-task data.

Usage:
    cont = ContinualDataset(cfg.dataset, cfg.memory, batch_size=64)
    for task_id, train_loader, test_loaders in cont.task_iterator():
        for x, y in train_loader:
            loss = model(x, y)
            if len(cont) > 0:
                x_r, y_r, _ = cont.sample_replay(64)
        method.end_task(task_id, train_loader)  # fills buffer
"""

from typing import Dict, Generator, Tuple

import torch
from torch.utils.data import DataLoader

from src.data.memory_buffer import ReservoirBuffer
from src.data.rotated_mnist import RotatedMNIST
from src.data.domain_cifar10 import DomainCIFAR10
from src.data.domain_cifar100 import DomainCIFAR100


# Registry of supported dataset names → class
_DATASET_REGISTRY = {
    "rot_mnist":    RotatedMNIST,
    "dom_cifar10":  DomainCIFAR10,
    "dom_cifar100": DomainCIFAR100,
}


class ContinualDataset:
    """Wraps a benchmark dataset with a growing replay buffer.

    Args:
        dataset_cfg: Dataset config (Hydra OmegaConf).  Must expose ``name``
                     (str) and the fields required by the underlying dataset
                     class (see RotatedMNIST, DomainCIFAR100).
        memory_cfg:  Memory config.  Must expose ``total_budget`` (int).
        batch_size:  Batch size forwarded to the underlying DataLoaders.
        num_workers: Worker count forwarded to the underlying DataLoaders.
    """

    def __init__(
        self,
        dataset_cfg,
        memory_cfg,
        batch_size: int = 64,
        num_workers: int = 0,
    ) -> None:
        name = dataset_cfg.name
        if name not in _DATASET_REGISTRY:
            raise ValueError(
                f"Unknown dataset '{name}'. "
                f"Supported: {list(_DATASET_REGISTRY.keys())}"
            )

        self._dataset_cfg = dataset_cfg
        self.num_tasks: int = dataset_cfg.num_tasks
        self._batch_size = batch_size
        self._num_workers = num_workers

        # Underlying benchmark dataset (RotatedMNIST, DomainCIFAR10, or DomainCIFAR100)
        self._dataset = _DATASET_REGISTRY[name](
            dataset_cfg, batch_size=batch_size, num_workers=num_workers
        )

        # Validate sampling strategy — only reservoir is implemented
        sampling = str(memory_cfg.sampling)
        if sampling != "reservoir":
            raise NotImplementedError(
                f"memory.sampling='{sampling}' is not implemented. "
                f"Only 'reservoir' is supported."
            )

        # Replay buffer shared across all tasks
        self._buffer = ReservoirBuffer(total_budget=memory_cfg.total_budget)

    # ------------------------------------------------------------------
    # Core generator
    # ------------------------------------------------------------------

    def task_iterator(
        self,
    ) -> Generator[Tuple[int, DataLoader, Dict[int, DataLoader]], None, None]:
        """Iterate over tasks, yielding loaders and filling the buffer.

        Yields:
            task_id:      Zero-based task index.
            train_loader: DataLoader over the current task's training set.
            test_loaders: ``dict`` mapping each seen task id (0 … task_id)
                          to its test DataLoader.  Use these to evaluate
                          forward transfer and forgetting after every step.

        Side effects:
            None.  Buffer population is handled by each method's end_task().
        """
        for task_id in range(self.num_tasks):
            # Fresh loaders for this task
            train_loader, _ = self._dataset.get_task_loaders(task_id)

            # Test loaders for every task seen so far (includes current task).
            # Built eagerly so the caller can start evaluating immediately.
            test_loaders: Dict[int, DataLoader] = {
                t: self._dataset.get_task_loaders(t)[1]
                for t in range(task_id + 1)
            }

            yield task_id, train_loader, test_loaders

    # ------------------------------------------------------------------
    # Replay
    # ------------------------------------------------------------------

    def sample_replay(
        self, batch_size: int
    ) -> Tuple[torch.Tensor, torch.Tensor, list]:
        """Sample a batch uniformly from the replay buffer.

        Args:
            batch_size: Number of samples to draw.

        Returns:
            x_batch:  Tensor of shape (batch_size, *input_shape).
            y_batch:  Tensor of shape (batch_size,).
            task_ids: List of int task IDs, length batch_size.

        Raises:
            ValueError: If called before any task has been completed (empty
                        buffer).
        """
        return self._buffer.sample(batch_size)

    # ------------------------------------------------------------------
    # Convenience / introspection
    # ------------------------------------------------------------------

    @property
    def buffer(self) -> ReservoirBuffer:
        """Public accessor for the shared replay buffer."""
        return self._buffer

    def __len__(self) -> int:
        """Return the number of samples currently stored in the buffer."""
        return len(self._buffer)

