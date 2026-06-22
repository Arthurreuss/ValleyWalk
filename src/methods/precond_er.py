"""Preconditioned ER — natural-gradient experience replay (baseline).

A second-order replay baseline: vanilla ER whose update direction is the
**damped natural gradient** rather than the raw gradient.  On each step the
joint ER gradient ``g = ∇(L_current + L_replay)`` is replaced by

    x = δ · (F + δI)⁻¹ g

where ``F`` is the Fisher information matrix and ``δ`` is the damping.  This
rescales the gradient by inverse curvature — small steps along high-curvature
directions, large steps along flat ones — which is the "rescale the loss
landscape" move (as opposed to deflating/removing directions).

The system ``(F + δI) x = g`` is solved with **Conjugate Gradient** (CG),
which needs only Fisher-vector products (``src/optim/fisher.py``) and never
forms ``F``.  CG yields the *full-rank* damped natural gradient: every
direction is scaled by its true ``1/(λ + δ)``, not just a top-d subspace.

Scaling note: the raw ``(F + δI)⁻¹ g`` scales flat directions (λ ≈ 0) by
``1/δ``.  Multiplying by ``δ`` (the ``δ·`` above) normalises flat directions
back to unit gain, so the effective step size matches plain ER and ``δ → ∞``
recovers gradient descent exactly.

The Fisher (= generalized Gauss-Newton for softmax-CE) is positive
semi-definite by construction, so ``F + δI`` is positive-definite and CG is
well-posed — no negative eigenvalues to handle.

CG is optionally **warm-started** with the previous step's solution: the
natural gradient changes slowly between steps, so this cuts the iteration
count substantially.

For task 0 (empty buffer) the method degrades to plain SGD on the current
batch: there is no replay loss to precondition against, and the Fisher of a
freshly-initialised single-task loss carries no preservation signal.

Usage::

    buffer = ReservoirBuffer(total_budget=500)
    method = PrecondER(model, cfg, buffer)
    for task_id, train_loader, test_loaders in dataset.task_iterator():
        for x, y in train_loader:
            result = method.observe(x, y, task_id)
        method.end_task(task_id, train_loader)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.methods.base_method import BaseMethod
from src.data.memory_buffer import ReservoirBuffer
from src.optim.fisher import make_fisher_vp, conjugate_gradient


class PrecondER(BaseMethod):
    """Fisher-preconditioned experience replay.

    Parameters
    ----------
    model : nn.Module
        The neural network being trained.
    cfg : omegaconf.DictConfig
        **Full** Hydra config.  PrecondER reads ``cfg.training`` for the
        optimiser hyper-parameters and ``cfg.method`` for its own settings
        (``replay_batch_size``, ``fisher``, ``cg``).
    buffer : ReservoirBuffer
        Shared replay buffer.  Populated by ``end_task()`` and sampled inside
        ``observe()`` to form the replay mini-batch.
    """

    def __init__(self, model: nn.Module, cfg: Any, buffer: ReservoirBuffer) -> None:
        super().__init__(model, cfg.method)

        self._full_cfg = cfg
        self.buffer = buffer

        self._params: List[nn.Parameter] = [
            p for p in model.parameters() if p.requires_grad
        ]
        self._n_params: int = sum(p.numel() for p in self._params)
        self._device: torch.device = next(model.parameters()).device

        mcfg = cfg.method

        # null → match the current-task batch size (default ER behaviour).
        rbs = getattr(mcfg, "replay_batch_size", None)
        self._replay_batch_size = int(rbs) if rbs is not None else None

        # ── Fisher / damping ──────────────────────────────────────────────
        self._fisher_target: str = str(mcfg.fisher.target)          # "joint"|"replay"
        self._damping: float = float(mcfg.fisher.damping)           # δ
        if self._fisher_target not in ("joint", "replay"):
            raise ValueError(
                f"fisher.target must be 'joint' or 'replay', "
                f"got '{self._fisher_target}'"
            )
        if self._damping <= 0.0:
            raise ValueError(f"fisher.damping must be > 0, got {self._damping}")

        # ── Conjugate-gradient solver ─────────────────────────────────────
        self._cg_iters: int = int(mcfg.cg.iters)        # max CG iterations
        self._cg_tol: float = float(mcfg.cg.tol)        # relative residual tol
        self._cg_warm_start: bool = bool(mcfg.cg.warm_start)
        if self._cg_iters < 1:
            raise ValueError(f"cg.iters must be >= 1, got {self._cg_iters}")

        # Momentum read from training config so the §4.6 momentum cross
        # (training.momentum=0.9) applies here too — momentum is accumulated on
        # top of the damped natural-gradient direction, mirroring NCL.
        self.optimizer = torch.optim.SGD(
            model.parameters(),
            lr=float(cfg.training.lr),
            momentum=float(getattr(cfg.training, "momentum", 0.0)),
            weight_decay=float(cfg.training.weight_decay),
        )
        self.loss_fn = nn.CrossEntropyLoss()

        # ── Mutable state ─────────────────────────────────────────────────
        self._step_count: int = 0
        # Previous step's natural-gradient solution, reused to warm-start CG.
        self._prev_x: Optional[torch.Tensor] = None

        # Diagnostics from the most recent observe() call.
        self._last_diagnostics: Dict[str, Any] = self._zero_diagnostics()

    # ------------------------------------------------------------------
    # Gradient helpers
    # ------------------------------------------------------------------

    def _get_flat_grad(self) -> torch.Tensor:
        """Collect parameter .grad into a single flat tensor on the device."""
        parts: List[torch.Tensor] = []
        for p in self._params:
            if p.grad is not None:
                parts.append(p.grad.detach().reshape(-1))
            else:
                parts.append(torch.zeros(p.numel(), device=self._device))
        return torch.cat(parts)

    def _overwrite_grad(self, d_flat: torch.Tensor) -> None:
        """Write a flat gradient vector back into the parameter .grad fields."""
        offset = 0
        for p in self._params:
            numel = p.numel()
            slice_ = d_flat[offset : offset + numel].view_as(p)
            if p.grad is None:
                p.grad = slice_.detach().clone()
            else:
                p.grad.data.copy_(slice_)
            offset += numel

    def _zero_diagnostics(self) -> Dict[str, Any]:
        """Diagnostics dict for task-0 steps (plain SGD, no natural gradient)."""
        return {
            "cg_iters": 0,
            "cg_residual": 0.0,
            "precond_ratio": 1.0,
        }

    # ------------------------------------------------------------------
    # BaseMethod abstract interface
    # ------------------------------------------------------------------

    def observe(
        self,
        x_batch: torch.Tensor,
        y_batch: torch.Tensor,
        task_id: int,
    ) -> Dict[str, float]:
        """One preconditioned-ER gradient step.

        Falls back to plain SGD on task 0 / empty buffer.  On later tasks it
        descends the joint ER loss preconditioned by the low-rank Fisher.

        Returns
        -------
        dict
            ``{"loss": float}`` — the joint loss being descended (current loss
            only on task 0).
        """
        self.model.train()
        x_batch = x_batch.to(self._device)
        y_batch = y_batch.to(self._device)

        if task_id == 0 or len(self.buffer) == 0:
            return self._sgd_step(x_batch, y_batch)
        return self._precond_step(x_batch, y_batch)

    def _sgd_step(
        self, x_batch: torch.Tensor, y_batch: torch.Tensor
    ) -> Dict[str, float]:
        """Plain SGD fallback for task 0 / empty buffer."""
        self.optimizer.zero_grad()
        loss = self.loss_fn(self.model(x_batch), y_batch)
        loss.backward()
        self.optimizer.step()

        self._step_count += 1
        self._last_diagnostics = self._zero_diagnostics()
        return {"loss": loss.item()}

    def _precond_step(
        self, x_batch: torch.Tensor, y_batch: torch.Tensor
    ) -> Dict[str, float]:
        """Natural-gradient step on the joint ER loss (task > 0)."""
        # ── Replay batch ──────────────────────────────────────────────────
        replay_bs = self._replay_batch_size or x_batch.size(0)
        x_replay, y_replay, _ = self.buffer.sample(replay_bs)
        x_replay = x_replay.to(self._device)
        y_replay = y_replay.to(self._device)

        # ── Joint ER gradient: g = ∇(L_current + L_replay) ────────────────
        self.optimizer.zero_grad()
        current_loss = self.loss_fn(self.model(x_batch), y_batch)
        replay_loss = self.loss_fn(self.model(x_replay), y_replay)
        joint_loss = current_loss + replay_loss
        joint_loss.backward()
        g = self._get_flat_grad()
        g_norm = torch.linalg.norm(g).item()

        # ── Damped natural gradient via CG: solve (F + δI) x = g ───────────
        # Fisher target: "replay" uses past-task curvature only; "joint" uses
        # the curvature of the concatenated batch the step moves through.
        if self._fisher_target == "replay":
            fx, fy = x_replay, y_replay
        else:
            fx = torch.cat([x_batch, x_replay], dim=0)
            fy = torch.cat([y_batch, y_replay], dim=0)
        fisher_vp = make_fisher_vp(self.model, fx, fy, self._params)

        x0 = self._prev_x if self._cg_warm_start else None
        x, n_iter = conjugate_gradient(
            fisher_vp,
            g,
            damping=self._damping,
            iters=self._cg_iters,
            tol=self._cg_tol,
            x0=x0,
        )
        if self._cg_warm_start:
            self._prev_x = x.detach()

        # Final residual for diagnostics: ‖(F + δI)x − g‖ / ‖g‖.
        residual = torch.linalg.norm(
            fisher_vp(x) + self._damping * x - g
        ).item() / (g_norm + 1e-12)

        # δ-normalisation: flat directions (λ ≈ 0) keep unit gain, so the step
        # size matches plain ER and δ → ∞ recovers gradient descent exactly.
        d = self._damping * x

        self._overwrite_grad(d)
        self.optimizer.step()

        self._last_diagnostics = {
            "cg_iters": n_iter,
            "cg_residual": residual,
            # < 1 ⇒ the natural gradient shrank the step (high-curvature energy
            # removed); ≈ 1 ⇒ direction was already well-conditioned.
            "precond_ratio": torch.linalg.norm(d).item() / (g_norm + 1e-12),
        }

        self._step_count += 1
        return {"loss": joint_loss.item()}

    def end_task(self, task_id: int, train_loader: DataLoader) -> None:
        """Populate the replay buffer with data from the completed task."""
        with torch.no_grad():
            for x_batch, y_batch in train_loader:
                x_batch = x_batch.cpu()
                y_batch = y_batch.cpu()
                for i in range(x_batch.size(0)):
                    self.buffer.add(x_batch[i], y_batch[i], task_id)

        # Drop the warm-start: the loss landscape (and so the natural gradient)
        # shifts at a task boundary, making the previous solution a poor guess.
        self._prev_x = None

    def get_step_diagnostics(self) -> Dict[str, Any]:
        """Return natural-gradient diagnostics from the last ``observe()`` call.

        Keys: ``cg_iters`` (CG iterations performed this step),
        ``cg_residual`` (relative residual ‖(F+δI)x − g‖/‖g‖ after the solve),
        ``precond_ratio`` (‖δ·x‖ / ‖g‖; < 1 when curvature shrank the step).
        """
        return dict(self._last_diagnostics)
