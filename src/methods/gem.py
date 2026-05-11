"""Gradient Episodic Memory (GEM) and Averaged GEM (A-GEM).

Reference gradients are computed **dynamically at each observe() step** from
buffer mini-batches, matching the original paper formulations exactly.

Two variants, controlled by ``cfg.method.gem.reference_gradient``:

  - ``per_task`` (GEM — Lopez-Paz & Ranzato 2017):
        One reference gradient per past task, each computed from a
        ``mem_batch_size`` mini-batch of that task's buffer samples.  The
        current-task gradient is projected to the nearest point satisfying
        all T inner-product constraints simultaneously (dual QP via
        quadprog, size T×T — much smaller than D×D).

  - ``joint`` (A-GEM — Chaudhry et al. 2019):
        A single reference gradient from a ``mem_batch_size`` mini-batch
        drawn uniformly from **all** buffer samples.  Single-constraint
        case admits a closed-form projection — O(D) vs O(T²·D) for
        per_task, making it scale to many tasks.

end_task() only populates the replay buffer.  Reference gradients are
**never cached between steps** — they are computed fresh inside each
observe() call.

References:
    Lopez-Paz & Ranzato (2017) "Gradient Episodic Memory for Continual Learning"
    https://arxiv.org/abs/1706.08840
    Chaudhry et al. (2019) "Efficient Lifelong Learning with A-GEM"
    https://arxiv.org/abs/1812.00420

Usage::

    buffer = ReservoirBuffer(total_budget=500)
    method = GEM(model, cfg, buffer)
    for task_id, train_loader, test_loaders in dataset.task_iterator():
        for x, y in train_loader:
            result = method.observe(x, y, task_id)   # → {"loss": float}
        method.end_task(task_id, train_loader)        # fills buffer
"""

from __future__ import annotations

import random
from typing import Dict, List, Tuple

import numpy as np
import quadprog
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.methods.base_method import BaseMethod
from src.data.memory_buffer import ReservoirBuffer


class GEM(BaseMethod):
    """Gradient Episodic Memory (GEM / A-GEM).

    Parameters
    ----------
    model : nn.Module
        The neural network being trained.
    cfg : omegaconf.DictConfig
        **Full** Hydra config.  GEM reads ``cfg.training`` for optimiser
        hyper-parameters and ``cfg.method.gem`` for GEM-specific settings.
    buffer : ReservoirBuffer
        Shared replay buffer.  Populated by ``end_task()`` and sampled in
        ``observe()`` for reference gradient computation.
    """

    def __init__(self, model: nn.Module, cfg, buffer: ReservoirBuffer) -> None:
        super().__init__(model, cfg.method)

        self._full_cfg = cfg
        self.buffer = buffer

        self._mode: str = str(cfg.method.gem.reference_gradient)  # "per_task" | "joint"
        self._margin: float = float(cfg.method.gem.margin)
        self._mem_batch_size: int = int(cfg.method.gem.mem_batch_size)

        if self._mode not in ("per_task", "joint"):
            raise ValueError(
                f"gem.reference_gradient must be 'per_task' or 'joint', "
                f"got '{self._mode}'"
            )

        self.optimizer = torch.optim.SGD(
            model.parameters(),
            lr=float(cfg.training.lr),
            weight_decay=float(cfg.training.weight_decay),
        )
        self.loss_fn = nn.CrossEntropyLoss()
        self._device: torch.device = next(model.parameters()).device

        self._n_params: int = sum(
            p.numel() for p in model.parameters() if p.requires_grad
        )

        # Tasks whose data is already in the buffer (updated by end_task).
        # Reference gradients are only computed when this is non-empty.
        self._past_tasks: List[int] = []

    # ------------------------------------------------------------------
    # Gradient helper utilities
    # ------------------------------------------------------------------

    def _get_flat_grad(self) -> torch.Tensor:
        """Collect all parameter .grad values into a single flat CPU tensor.

        Parameters whose .grad is None contribute zeros.
        """
        parts: List[torch.Tensor] = []
        for p in self.model.parameters():
            if not p.requires_grad:
                continue
            if p.grad is not None:
                parts.append(p.grad.data.view(-1).cpu())
            else:
                parts.append(torch.zeros(p.numel()))
        return torch.cat(parts)

    def _overwrite_grad(self, d: torch.Tensor) -> None:
        """Write a flat gradient vector back into model parameter .grad fields.

        Parameters
        ----------
        d : torch.Tensor
            Flat gradient vector of length ``self._n_params``, on any device.
        """
        d = d.cpu()
        offset = 0
        for p in self.model.parameters():
            if not p.requires_grad:
                continue
            numel = p.numel()
            grad_slice = d[offset : offset + numel].view_as(p).to(p.device)
            if p.grad is None:
                p.grad = grad_slice.clone()
            else:
                p.grad.data.copy_(grad_slice)
            offset += numel

    # ------------------------------------------------------------------
    # Dynamic reference gradient computation
    # ------------------------------------------------------------------

    def _compute_ref_grad(self, xs: torch.Tensor, ys: torch.Tensor) -> torch.Tensor:
        """Compute a flat reference gradient from a mini-batch.

        Runs a forward+backward in eval mode (on-device) and returns the
        gradient as a flat CPU tensor.  The caller is responsible for saving
        the current-task gradient before calling this method.

        Parameters
        ----------
        xs : torch.Tensor
            Input mini-batch already on ``self._device``.
        ys : torch.Tensor
            Label mini-batch already on ``self._device``.

        Returns
        -------
        torch.Tensor
            Flat gradient on CPU, shape ``(D,)``.
        """
        self.model.eval()
        self.optimizer.zero_grad()
        loss = self.loss_fn(self.model(xs), ys)
        loss.backward()
        g = self._get_flat_grad()
        self.model.train()
        return g

    def _sample_task_batch(self, task_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample ``mem_batch_size`` items for a specific task from the buffer.

        Samples with replacement so the size is always exactly
        ``mem_batch_size``, even if fewer matching entries exist.

        Parameters
        ----------
        task_id : int
            Task to filter by.

        Returns
        -------
        xs, ys : torch.Tensor
            Mini-batch tensors on ``self._device``.
        """
        indices = [i for i, t in enumerate(self.buffer._task_ids) if t == task_id]
        sampled = [random.choice(indices) for _ in range(self._mem_batch_size)]
        xs = torch.stack([self.buffer._xs[i] for i in sampled]).to(self._device)
        ys = torch.stack([self.buffer._ys[i] for i in sampled]).to(self._device)
        return xs, ys

    # ------------------------------------------------------------------
    # Gradient projection
    # ------------------------------------------------------------------

    def _project_gradient(
        self,
        g: torch.Tensor,
        ref_grads: List[torch.Tensor],
    ) -> torch.Tensor:
        """Project gradient ``g`` to satisfy all GEM inner-product constraints.

        The constraint for reference gradient g_ref_t is:
            ⟨d, g_ref_t⟩ ≥ -margin

        **joint mode** (single constraint) — closed-form O(D):
            v* = max(0, -(⟨g, g_ref⟩ + margin) / ‖g_ref‖²)
            d* = g + v* · g_ref

        **per_task mode** (T constraints) — dual QP of size T×T:
            min_{v≥0}  ½ vᵀ(GGᵀ)v + vᵀ(Gg + margin·1)
            d* = g + Gᵀ v*

        Parameters
        ----------
        g : torch.Tensor
            Flat current-task gradient, shape ``(D,)``.
        ref_grads : list of torch.Tensor
            Reference gradients to constrain against (flat CPU tensors).

        Returns
        -------
        torch.Tensor
            Projected gradient, same shape and device as ``g``.
        """
        if not ref_grads:
            return g

        if self._mode == "joint":
            # ── Single-constraint analytical projection ─────────────────
            g_ref = ref_grads[0].to(g.device)
            dot = torch.dot(g, g_ref).item()

            if dot >= -self._margin:
                return g  # constraint already satisfied

            gref_sq = torch.dot(g_ref, g_ref).item()
            if gref_sq < 1e-12:
                return g  # degenerate reference — skip

            v = -(dot + self._margin) / gref_sq
            return g + v * g_ref

        else:
            # ── Per-task dual QP via quadprog ───────────────────────────
            G = torch.stack([gr.to(g.device) for gr in ref_grads])  # (T, D)
            dots = (G @ g).cpu()

            if (dots >= -self._margin).all():
                return g  # all constraints satisfied

            GGT = (G @ G.T).double().cpu().numpy()
            Gg_np = dots.double().numpy()
            T_dim = GGT.shape[0]

            GGT_reg = GGT + 1e-8 * np.eye(T_dim)
            a = -(Gg_np + self._margin)
            C = np.eye(T_dim)
            b = np.zeros(T_dim)

            v_star = quadprog.solve_qp(GGT_reg, a, C, b)[0]
            v_star = np.maximum(v_star, 0.0)  # numerical safety

            v_star_t = torch.tensor(v_star, dtype=g.dtype, device=g.device)
            return g + G.T @ v_star_t

    # ------------------------------------------------------------------
    # BaseMethod abstract interface
    # ------------------------------------------------------------------

    def observe(
        self,
        x_batch: torch.Tensor,
        y_batch: torch.Tensor,
        task_id: int,
    ) -> Dict[str, float]:
        """One gradient step with optional GEM projection.

        Reference gradients are computed fresh from buffer mini-batches at
        every step.  On the first task (no past tasks yet) falls back to
        plain SGD — identical to standard training.

        Parameters
        ----------
        x_batch : torch.Tensor
            Current-task inputs, shape ``(B, ...)``.
        y_batch : torch.Tensor
            Current-task labels, shape ``(B,)``.
        task_id : int
            Zero-based index of the task being trained.

        Returns
        -------
        dict
            ``{"loss": float}`` — scalar CE loss on the current batch.
        """
        self.model.train()
        x_batch = x_batch.to(self._device)
        y_batch = y_batch.to(self._device)

        # ── Current-task gradient ──────────────────────────────────────
        self.optimizer.zero_grad()
        logits = self.model(x_batch)
        current_loss = self.loss_fn(logits, y_batch)
        current_loss.backward()

        # ── GEM projection (skipped on first task — no past tasks yet) ─
        if self._past_tasks:
            g = self._get_flat_grad()  # save current gradient before ref computations

            if self._mode == "joint":
                # A-GEM: one ref grad from a full-buffer mini-batch
                x_ref, y_ref, _ = self.buffer.sample(self._mem_batch_size)
                x_ref = x_ref.to(self._device)
                y_ref = y_ref.to(self._device)
                ref_grads = [self._compute_ref_grad(x_ref, y_ref)]
            else:
                # GEM: one ref grad per past task, each from a task mini-batch
                ref_grads = [
                    self._compute_ref_grad(*self._sample_task_batch(t))
                    for t in self._past_tasks
                ]

            # _compute_ref_grad zeroes .grad internally; restore projected grad
            g_proj = self._project_gradient(g, ref_grads)
            self._overwrite_grad(g_proj)

        self.optimizer.step()
        return {"loss": current_loss.item()}

    def end_task(self, task_id: int, train_loader: DataLoader) -> None:
        """Populate the replay buffer with data from the just-finished task.

        Reference gradients are **not** cached here — they are computed
        dynamically inside ``observe()`` at every training step.

        Parameters
        ----------
        task_id : int
            Zero-based index of the task that just finished.
        train_loader : DataLoader
            Training data loader for the completed task.
        """
        with torch.no_grad():
            for x_batch, y_batch in train_loader:
                x_batch = x_batch.cpu()
                y_batch = y_batch.cpu()
                for i in range(x_batch.size(0)):
                    self.buffer.add(x_batch[i], y_batch[i], task_id)

        if task_id not in self._past_tasks:
            self._past_tasks.append(task_id)
