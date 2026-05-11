"""Replay memory buffer using reservoir sampling (Algorithm R, Vitter 1985).

`ReservoirBuffer` stores a fixed-size random subset of all items ever seen,
guaranteeing that at any point each item has the same probability (budget/N)
of occupying a slot — regardless of the order in which items arrive.

Usage:
    buf = ReservoirBuffer(total_budget=500)
    buf.add(x, y, task_id=0)        # called per sample during/after a task
    x_r, y_r, tids = buf.sample(64) # called during training for replay
"""

import random
from typing import List, Tuple

import torch


class ReservoirBuffer:
    """Fixed-capacity replay buffer with reservoir-sampling insertion.

    All tensors are stored on CPU to keep GPU memory free for forward passes.

    Args:
        total_budget: Maximum number of samples the buffer can hold.
    """

    def __init__(self, total_budget: int) -> None:
        self.total_budget: int = total_budget

        # Internal storage — one Python list per field for simplicity.
        # Each element corresponds to one stored sample.
        self._xs: List[torch.Tensor] = []
        self._ys: List[torch.Tensor] = []
        self._task_ids: List[int] = []

        # Total items seen so far (across all add() calls), used for the
        # reservoir acceptance probability budget / _n_seen.
        self._n_seen: int = 0

    # ------------------------------------------------------------------
    # Core reservoir-sampling insertion
    # ------------------------------------------------------------------

    def add(self, x: torch.Tensor, y: torch.Tensor, task_id: int) -> None:
        """Insert one sample into the buffer using reservoir sampling.

        Algorithm R: the k-th arriving item is placed in a uniformly random
        slot with probability budget/k.  This gives every item seen so far
        an equal probability of being in the buffer at any time.

        Args:
            x:       Input tensor for one sample (any shape).
            y:       Label tensor for one sample (scalar or vector).
            task_id: Integer task identifier for the sample.
        """
        self._n_seen += 1

        if len(self._xs) < self.total_budget:
            # Buffer not yet full — always accept.
            self._xs.append(x.cpu())
            self._ys.append(y.cpu())
            self._task_ids.append(int(task_id))
        else:
            # Buffer full — replace slot j with probability budget / _n_seen.
            # random.randrange(n) is uniform over [0, n-1], so P(j < budget)
            # = budget / _n_seen exactly.
            j = random.randrange(self._n_seen)
            if j < self.total_budget:
                self._xs[j] = x.cpu()
                self._ys[j] = y.cpu()
                self._task_ids[j] = int(task_id)

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def sample(
        self, batch_size: int
    ) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
        """Sample `batch_size` items uniformly at random (with replacement).

        Args:
            batch_size: Number of items to draw.

        Returns:
            x_batch:  Tensor of shape (batch_size, *x.shape).
            y_batch:  Tensor of shape (batch_size, *y.shape).
            task_ids: List of integer task identifiers, length batch_size.

        Raises:
            ValueError: If the buffer is empty.
        """
        n = len(self._xs)
        if n == 0:
            raise ValueError("Cannot sample from an empty ReservoirBuffer.")

        indices = [random.randrange(n) for _ in range(batch_size)]
        x_batch = torch.stack([self._xs[i] for i in indices])
        y_batch = torch.stack([self._ys[i] for i in indices])
        task_ids = [self._task_ids[i] for i in indices]
        return x_batch, y_batch, task_ids

    def sample_all(
        self,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
        """Return every currently-stored sample exactly once.

        Used by ER when ``method.replay_full_buffer=true`` to compute the
        *exact* empirical past-task gradient — i.e. the gradient of the
        cross-entropy averaged over every sample in the buffer, with no
        sampling noise.  When the buffer holds the full past-task training
        set (the G2/G4 condition with ``memory.total_budget`` equal to the
        per-task training size), the resulting gradient is the deterministic
        full-data gradient at the current parameters.

        Returns:
            x_batch:  Tensor of shape (len(buffer), *x.shape).
            y_batch:  Tensor of shape (len(buffer), *y.shape).
            task_ids: List of integer task identifiers, length len(buffer).

        Raises:
            ValueError: If the buffer is empty.
        """
        n = len(self._xs)
        if n == 0:
            raise ValueError("Cannot sample from an empty ReservoirBuffer.")
        x_batch = torch.stack(self._xs)
        y_batch = torch.stack(self._ys)
        task_ids = list(self._task_ids)
        return x_batch, y_batch, task_ids

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        """Return the number of samples currently stored."""
        return len(self._xs)

    # ------------------------------------------------------------------
    # Serialisation (for checkpointing mid-task)
    # ------------------------------------------------------------------

    def state_dict(self) -> dict:
        """Return a serialisable snapshot of the buffer.

        Safe to pass to `torch.save()`.  The snapshot captures internal
        counter state so that reservoir sampling remains unbiased after
        a checkpoint-restore cycle.
        """
        return {
            "total_budget": self.total_budget,
            "xs": list(self._xs),          # list of CPU tensors
            "ys": list(self._ys),
            "task_ids": list(self._task_ids),
            "n_seen": self._n_seen,
        }

    def load_state_dict(self, state: dict) -> None:
        """Restore buffer state from a dict produced by `state_dict()`.

        Args:
            state: Dict as returned by `state_dict()`.
        """
        self.total_budget = state["total_budget"]
        self._xs = list(state["xs"])
        self._ys = list(state["ys"])
        self._task_ids = list(state["task_ids"])
        self._n_seen = state["n_seen"]
