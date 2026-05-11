"""Abstract base class for all continual-learning methods.

Every method (ER, GEM, NCL, CACL) subclasses ``BaseMethod`` and implements
the three abstract hooks.  The training loop in ``scripts/train.py`` only
calls this interface, so it is fully method-agnostic.

Interface contract:
    observe(x, y, task_id) → dict
        Perform one gradient step on the current mini-batch.
        Must return at least {"loss": float}.  Methods may include
        method-specific keys (e.g. "rho", "eta" for CACL).

    end_task(task_id, train_loader)
        Called once at the end of each task.  Use for post-task bookkeeping:
        updating the replay buffer, computing Fisher matrices, caching
        reference gradients, etc.

    evaluate(test_loader) → float
        Run inference on a data loader and return top-1 accuracy.
        Implemented here — subclasses do not need to override.

    get_step_diagnostics() → dict
        Return the diagnostic scalars logged after the *last* observe() call.
        Defaults to an empty dict; override in methods that produce diagnostics
        (e.g. CACL returns cone_fallback, trust_radius, …).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class BaseMethod(ABC):
    """Abstract base for all continual-learning training methods.

    Parameters
    ----------
    model : nn.Module
        The neural network being trained.  Subclasses should store a
        reference and use it in ``observe``.
    cfg : omegaconf.DictConfig
        The *method* sub-config (e.g. ``cfg.method`` from the master config).
        Subclasses are free to read any keys they need.
    """

    def __init__(self, model: nn.Module, cfg: Any) -> None:
        self.model = model
        self.cfg = cfg

    # ------------------------------------------------------------------
    # Abstract interface — subclasses *must* implement these
    # ------------------------------------------------------------------

    @abstractmethod
    def observe(
        self,
        x_batch: torch.Tensor,
        y_batch: torch.Tensor,
        task_id: int,
    ) -> Dict[str, float]:
        """Perform one gradient step on (x_batch, y_batch).

        Parameters
        ----------
        x_batch : torch.Tensor
            Input features, shape (B, ...).
        y_batch : torch.Tensor
            Target labels, shape (B,).
        task_id : int
            Zero-based index of the task currently being trained.

        Returns
        -------
        dict
            Must contain at least ``{"loss": float}``.  Method-specific
            extras (e.g. ``"rho"``, ``"eta"``) are also allowed and will
            be forwarded to the dual-write logger.
        """
        raise NotImplementedError

    @abstractmethod
    def end_task(self, task_id: int, train_loader: DataLoader) -> None:
        """Post-task bookkeeping hook.

        Called exactly once after all mini-batches in a task have been
        processed.  Typical uses:

        * Populate / update the replay memory buffer (ER, GEM, CACL).
        * Compute and cache Fisher matrices or reference gradients (NCL, GEM).
        * Reset per-task counters.

        Parameters
        ----------
        task_id : int
            Zero-based index of the task that just finished.
        train_loader : DataLoader
            The full training loader for the just-finished task.  Available
            for methods that need a second pass (e.g. Fisher estimation).
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Concrete methods — subclasses *may* override but need not
    # ------------------------------------------------------------------

    @torch.no_grad()
    def evaluate(self, test_loader: DataLoader) -> float:
        """Compute top-1 accuracy on a test loader.

        The model is temporarily put into eval mode and restored afterwards.

        Parameters
        ----------
        test_loader : DataLoader
            Yields (x, y) batches.

        Returns
        -------
        float
            Fraction of correctly classified samples in [0, 1].
        """
        self.model.eval()
        correct = 0
        total = 0
        for x, y in test_loader:
            # Move to the same device as the model parameters
            device = next(self.model.parameters()).device
            x = x.to(device)
            y = y.to(device)
            logits = self.model(x)
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
        self.model.train()
        return correct / total if total > 0 else 0.0

    def get_step_diagnostics(self) -> Dict[str, Any]:
        """Return method-specific diagnostic scalars from the last step.

        The training loop calls this after every ``observe()`` and forwards
        the result to the dual-write logger.  The base implementation returns
        an empty dict — methods with rich diagnostics (e.g. CACL) should
        override this and return the relevant scalars.

        Returns
        -------
        dict
            Diagnostic key-value pairs.  All values should be JSON-serialisable
            scalars (float, int, bool).
        """
        return {}
