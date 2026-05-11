"""Experience Replay (ER) continual-learning method.

ER is the simplest replay-based baseline: at each training step it forms a
*joint* mini-batch by combining the current-task batch with a replay batch
sampled from the memory buffer, then takes a single SGD step on the
combined cross-entropy loss.

    joint_loss = CE(f(x_current), y_current) + CE(f(x_replay), y_replay)

On the very first task the buffer is empty, so ER degrades to plain SGD.

References:
    Ratcliff (1990) "Connectionist models of recognition memory"
    Lopez-Paz & Ranzato (2017) "Gradient Episodic Memory for Continual Learning"

Usage::

    buffer = ReservoirBuffer(total_budget=500)
    method = ER(model, cfg, buffer)
    for task_id, train_loader, test_loaders in dataset.task_iterator():
        for x, y in train_loader:
            result = method.observe(x, y, task_id)   # → {"loss": float}
        method.end_task(task_id, train_loader)       # fills buffer
"""

from __future__ import annotations

import math
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.methods.base_method import BaseMethod
from src.data.memory_buffer import ReservoirBuffer


class ER(BaseMethod):
    """Experience Replay.

    Parameters
    ----------
    model : nn.Module
        The neural network being trained.
    cfg : omegaconf.DictConfig
        **Full** Hydra config.  ER reads ``cfg.training`` for optimiser
        hyper-parameters (``lr``, ``weight_decay``) and passes
        ``cfg.method`` up to ``BaseMethod``.
    buffer : ReservoirBuffer
        Shared replay buffer — populated by ``end_task()`` and sampled inside
        ``observe()``.  Passed in from the outside so the same buffer object
        can be shared with the training loop (for checkpointing etc.).
    """

    def __init__(self, model: nn.Module, cfg, buffer: ReservoirBuffer) -> None:
        # BaseMethod stores model and the *method* sub-config
        super().__init__(model, cfg.method)

        self._full_cfg = cfg        # kept for any downstream use
        self.buffer = buffer

        # Standard SGD — using cfg.training ensures ER is controlled by the
        # same knobs as every other method's base optimiser.
        self.optimizer = torch.optim.SGD(
            model.parameters(),
            lr=float(cfg.training.lr),
            momentum=float(cfg.training.momentum),
            weight_decay=float(cfg.training.weight_decay),
        )
        self.loss_fn = nn.CrossEntropyLoss()

        # Cache device from the model's first parameter so we don't call
        # next(params) on every observe() call.
        self._device: torch.device = next(model.parameters()).device

        self._mode: str = str(cfg.method.mode)  # "standard" | "balanced"
        gb = cfg.method.grad_balance
        self._normalize_components: bool = bool(gb.normalize_components)
        self._task_weighted: bool = bool(gb.task_weighted)
        self._params = [p for p in model.parameters() if p.requires_grad]
        self._last_grad_ratio: float = 1.0

        self._reset_momentum_steps: int = int(cfg.method.reset_momentum_steps)
        self._current_task_id: int = -1
        self._task_step: int = 0
        self._last_true_grad_cosine: float = 0.0
        self._last_true_grad_mag_ratio: float = 1.0

        # ── g_true diagnostic state ──────────────────────────────────────
        # ``g_true`` here means ∇_θ L_past(θ) computed on a *fresh, large*
        # sample of the previous task's training data — i.e. an unbiased
        # estimate of the true past-task gradient at the current parameters.
        # It is what ``g_replay`` (the buffer estimate) is *trying* to be.
        # The cosine and magnitude-ratio diagnostics below are with respect
        # to this g_true, so they directly measure the directional and
        # magnitude bias of the buffer estimate (Chapter 3.2.2 / 3.2.1).
        # Set via ``set_diagnostic_data()`` from the training loop at the
        # start of each task with task_id > 0.  When unset (default), the
        # diagnostics fall back to neutral values (0.0 / 1.0).
        self._diagnostic_data: Optional[tuple] = None
        self._g_true_flat: Optional[torch.Tensor] = None
        self._g_true_norm: Optional[torch.Tensor] = None

        # Decouple replay batch size from current batch size. None → match the
        # current-task batch (legacy behaviour). A large int approximates the
        # full past-task gradient (used in the full-data-replay condition).
        rbs = cfg.method.get("replay_batch_size", None)
        self._replay_batch_size: Optional[int] = int(rbs) if rbs is not None else None

        # When true, use the entire buffer (each stored sample exactly once)
        # as the replay batch.  Overrides ``replay_batch_size``.  Combined
        # with ``memory.total_budget`` equal to the per-task training-set
        # size, this gives the exact empirical past-task gradient at the
        # current parameters — the G2 / G4 condition in §4.4.
        self._replay_full_buffer: bool = bool(
            cfg.method.get("replay_full_buffer", False)
        )

        # ── λ-curriculum (homotopy on new-task loss weight) ───────────────
        # Mirrors CACL's lambda_curriculum block.  When disabled (default)
        # every formula below reduces to the original ER step exactly.
        lc = cfg.method.get("lambda_curriculum", None)
        if lc is None:
            self._lc_enabled: bool = False
            self._lc_ramp_steps: int = 0
            self._lc_schedule: str = "linear"
            self._lc_ema_alpha: float = 0.05
            self._lc_lambda_min: float = 0.0
        else:
            self._lc_enabled = bool(lc.enabled)
            self._lc_ramp_steps = int(lc.ramp_steps)
            self._lc_schedule = str(lc.schedule)
            self._lc_ema_alpha = float(lc.get("ema_alpha", 0.05))
            self._lc_lambda_min = float(lc.get("lambda_min", 0.0))

        # EMA of ‖g_replay‖/‖g_new‖ — only used by schedule="adaptive".
        # Reset to 0 at each task boundary so λ starts at 0 on the first step
        # of every new task (matching the slow-start behaviour of the static
        # schedules).  See _curriculum_lambda for the update rule.
        self._lc_r_ema: float = 0.0

        # ── Symmetric LR warm-up (M2 falsification) ───────────────────────
        # Scales the optimiser's lr by progress for the first warmup_steps of
        # each task with task_id > 0.  Unlike the λ-curriculum, BOTH the new
        # and replay gradients are slowed equally — used to test whether the
        # depth reduction comes from asymmetric scaling or just small steps.
        lw = cfg.method.get("lr_warmup", None)
        if lw is None:
            self._lr_warmup_enabled: bool = False
            self._lr_warmup_steps: int = 0
        else:
            self._lr_warmup_enabled = bool(lw.enabled)
            self._lr_warmup_steps = int(lw.warmup_steps)

    def _curriculum_lambda(
        self,
        task_id: int,
        g_new_norm: Optional[float] = None,
        g_replay_norm: Optional[float] = None,
    ) -> float:
        """Curriculum weight λ on the new-task loss for the current step.

        Returns ``1.0`` (no curriculum) when the curriculum is disabled, on
        task 0 (no past tasks to preserve), or after the ramp has completed.
        Otherwise ramps from 0 → 1 over the first ``ramp_steps`` of the task
        using the configured schedule.  ``self._task_step`` is the within-task
        step counter (already maintained by ``observe()``).

        Schedules:
          - linear (default): λ = progress
          - cosine:           λ = 0.5·(1 − cos(π·progress))
          - step:             λ = 0 for first N steps, then 1
                              (= "replay-only delay" M3 falsification)
          - adaptive:         λ = clip(EMA(‖g_replay‖/‖g_new‖), 0, 1).
                              Self-paced: λ tracks the running gradient ratio,
                              which starts near 0 (replay grad ≈ 0 at θ_T_0*)
                              and climbs to 1 as the gradients rebalance.
                              Ignores ``ramp_steps``; requires the gradient
                              norms to be passed in.

        After the schedule resolves a raw λ, the result is floored at
        ``self._lc_lambda_min`` (default 0.0).  A small positive floor keeps
        the new-task gradient from being fully suppressed at step 1 — most
        relevant for ``adaptive`` at θ_T_0*, where ‖g_replay‖ ≈ 0 drives
        EMA(r) ≈ 0 and the initial interpolation can be slower than needed.

        Parameters
        ----------
        g_new_norm, g_replay_norm : float, optional
            Current-step gradient norms.  Required only when
            ``schedule == "adaptive"``; ignored for time-indexed schedules.
        """
        if not self._lc_enabled or task_id == 0:
            return 1.0

        if self._lc_schedule == "adaptive":
            # Self-paced; ignores ramp_steps.  Uses last EMA value if norms
            # aren't supplied (defensive — should not happen in normal flow).
            if g_new_norm is not None and g_replay_norm is not None:
                r = g_replay_norm / (g_new_norm + 1e-8)
                self._lc_r_ema = (
                    (1.0 - self._lc_ema_alpha) * self._lc_r_ema
                    + self._lc_ema_alpha * r
                )
            lam_raw = min(max(self._lc_r_ema, 0.0), 1.0)
            return float(max(lam_raw, self._lc_lambda_min))

        # Time-indexed schedules: linear / cosine / step.
        if self._lc_ramp_steps <= 0 or self._task_step > self._lc_ramp_steps:
            return 1.0
        progress = (self._task_step - 1) / self._lc_ramp_steps  # _task_step is 1-indexed at use site
        if progress >= 1.0:
            return 1.0
        if self._lc_schedule == "cosine":
            lam_raw = 0.5 * (1.0 - math.cos(math.pi * progress))
        elif self._lc_schedule == "step":
            lam_raw = 0.0
        else:
            lam_raw = progress  # linear (default)
        return max(lam_raw, self._lc_lambda_min)

    def _lr_warmup_factor(self, task_id: int) -> float:
        """Symmetric lr-warmup factor for the current step.

        Returns ``1.0`` (no warm-up) when disabled, on task 0, or after the
        warm-up window has completed.  Otherwise ramps linearly from 0 → 1
        over ``warmup_steps`` at the start of each new task.  Mirrors the
        gating of ``_curriculum_lambda`` so the two ablations are matched.
        """
        if (
            not self._lr_warmup_enabled
            or task_id == 0
            or self._lr_warmup_steps <= 0
            or self._task_step > self._lr_warmup_steps
        ):
            return 1.0
        factor = (self._task_step - 1) / self._lr_warmup_steps
        if factor >= 1.0:
            return 1.0
        return factor

    def _step_optimizer(self, lr_factor: float) -> None:
        """Single optimiser step with optional lr scaling, restored after."""
        if lr_factor == 1.0:
            self.optimizer.step()
            return
        saved_lrs = [pg["lr"] for pg in self.optimizer.param_groups]
        for pg in self.optimizer.param_groups:
            pg["lr"] = pg["lr"] * lr_factor
        try:
            self.optimizer.step()
        finally:
            for pg, lr in zip(self.optimizer.param_groups, saved_lrs):
                pg["lr"] = lr

    # ------------------------------------------------------------------
    # g_true diagnostic computation
    # ------------------------------------------------------------------

    def set_diagnostic_data(
        self,
        x: Optional[torch.Tensor],
        y: Optional[torch.Tensor],
    ) -> None:
        """Provide a fixed batch of past-task data for g_true diagnostics.

        ``g_true`` ≔ ∇_θ L_past(θ) computed on ``(x, y)`` directly (no
        buffer).  When this is set with a sufficiently large, distribution-
        ally faithful sample of past-task training data, ``g_true`` is a
        low-variance, low-bias estimate of the true past-task gradient at
        the current parameters — i.e. what ``g_replay`` (the buffer-based
        mini-batch gradient) is trying to approximate.

        The cosine and magnitude-ratio diagnostics returned by
        ``get_last_true_grad_cosine()`` and ``get_last_true_grad_mag_ratio()``
        are then computed against this g_true, giving direct per-step
        readouts of:

          - ``cos(g_replay, g_true)``  — directional bias of the buffer
            estimate (1.0 = no bias; < 1 = buffer points the wrong way).
          - ``‖g_replay‖ / ‖g_true‖``  — magnitude bias of the buffer
            estimate (1.0 = unbiased scale; < 1 = buffer underestimates
            past-gradient magnitude).

        Pass ``(None, None)`` to clear the diagnostic data (e.g. at the end
        of a task).

        Parameters
        ----------
        x, y : torch.Tensor or None
            Inputs and labels from the previous task(s).  Pass the entire
            past-task training set to obtain the deterministic empirical
            past-task gradient at the current parameters (zero sampling
            noise — this is what makes the diagnostic the *true* past-task
            gradient, not an estimate of it).  Tensors are moved to the
            model's device internally.
        """
        if x is None or y is None:
            self._diagnostic_data = None
        else:
            self._diagnostic_data = (x.to(self._device), y.to(self._device))

    def _compute_g_true(self) -> None:
        """Forward+backward over the diagnostic batch to populate g_true.

        Side-effects only: stores ``self._g_true_flat`` (the flat parameter-
        gradient vector) and ``self._g_true_norm`` (its L2 norm).  Leaves
        ``.grad`` zeroed so the caller can proceed with the normal training
        pass without contamination.

        No-op when ``self._diagnostic_data`` is None — the diagnostics are
        then disabled for this step (cosine/mag-ratio fall back to neutral
        defaults).
        """
        if self._diagnostic_data is None:
            self._g_true_flat = None
            self._g_true_norm = None
            return

        x_diag, y_diag = self._diagnostic_data
        self.optimizer.zero_grad()
        diag_loss = self.loss_fn(self.model(x_diag), y_diag)
        diag_loss.backward()
        self._g_true_flat = torch.cat(
            [p.grad.detach().flatten() for p in self._params]
        )
        self._g_true_norm = torch.linalg.norm(self._g_true_flat)
        # Clear .grad so the subsequent training pass starts clean.
        self.optimizer.zero_grad()

    def _record_buffer_diagnostics(
        self,
        g_replay_flat: torch.Tensor,
        g_replay_norm: torch.Tensor,
    ) -> None:
        """Compute ``cos(g_replay, g_true)`` and ``‖g_replay‖/‖g_true‖``.

        Updates ``self._last_true_grad_cosine`` and
        ``self._last_true_grad_mag_ratio`` in-place.  Falls back to neutral
        values (0.0 / 1.0) when g_true was not computed for this step.
        """
        if self._g_true_flat is None or self._g_true_norm is None:
            self._last_true_grad_cosine = 0.0
            self._last_true_grad_mag_ratio = 1.0
            return
        cos_dir = torch.dot(g_replay_flat, self._g_true_flat) / (
            g_replay_norm * self._g_true_norm + 1e-8
        )
        self._last_true_grad_cosine = cos_dir.item()
        self._last_true_grad_mag_ratio = (
            g_replay_norm / (self._g_true_norm + 1e-8)
        ).item()

    # ------------------------------------------------------------------
    # BaseMethod abstract interface
    # ------------------------------------------------------------------

    def observe(
        self,
        x_batch: torch.Tensor,
        y_batch: torch.Tensor,
        task_id: int,
    ) -> Dict[str, float]:
        """One gradient step combining current-task and replay data.

        On task 0 (empty buffer) both modes degrade to plain SGD.

        In ``standard`` mode: one forward/backward on ``current_loss + replay_loss``.
        In ``balanced`` mode: two separate backward passes with configurable
        gradient combining and normalisation (see ``grad_balance`` config).

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
            ``{"loss": float}``
        """
        if task_id != self._current_task_id:
            self._current_task_id = task_id
            self._task_step = 0
            # Reset adaptive-schedule EMA so λ starts at 0 on each new task.
            self._lc_r_ema = 0.0
        if self._reset_momentum_steps > 0 and self._task_step < self._reset_momentum_steps:
            for p in self.optimizer.state.values():
                p.pop("momentum_buffer", None)
        self._task_step += 1

        self.model.train()
        x_batch = x_batch.to(self._device)
        y_batch = y_batch.to(self._device)

        # ── g_true (full past-data gradient at current θ) ──────────────
        # Computed up-front so the cos / magnitude-ratio diagnostics in the
        # step functions can compare g_replay against an unbiased estimate
        # of the true past-task gradient.  No-op when no diagnostic data has
        # been set (task 0, or training loop not wired for diagnostics).
        # Side-effect: leaves .grad zeroed.
        self._compute_g_true()

        self.optimizer.zero_grad()

        # ── Current-task loss ──────────────────────────────────────────
        logits = self.model(x_batch)
        current_loss = self.loss_fn(logits, y_batch)

        # ── Task 0: buffer empty — plain SGD, same for both modes ─────
        # No warm-up on task 0 (model is at random init — slowing it would
        # just delay learning the first task with no benefit).
        if len(self.buffer) == 0:
            current_loss.backward()
            self.optimizer.step()
            return {"loss": current_loss.item()}

        if self._replay_full_buffer:
            x_replay, y_replay, _ = self.buffer.sample_all()
        else:
            replay_size = self._replay_batch_size or x_batch.size(0)
            x_replay, y_replay, _ = self.buffer.sample(replay_size)
        x_replay = x_replay.to(self._device)
        y_replay = y_replay.to(self._device)

        replay_logits = self.model(x_replay)
        replay_loss = self.loss_fn(replay_logits, y_replay)

        lr_factor: float = self._lr_warmup_factor(task_id)

        # λ is resolved inside the step functions: time-indexed schedules read
        # ``self._task_step``; the adaptive schedule needs the current-step
        # gradient norms, which are only available after the backward passes.
        if self._mode == "standard":
            return self._standard_step(current_loss, replay_loss, task_id, lr_factor)
        return self._balanced_step(current_loss, replay_loss, task_id, lr_factor)

    def _standard_step(
        self,
        current_loss: torch.Tensor,
        replay_loss: torch.Tensor,
        task_id: int,
        lr_factor: float = 1.0,
    ) -> Dict[str, float]:
        """Original ER: gradient is ``λ·g_new + g_replay``.

        Two backward passes are used so the gradient norms are available
        before λ is resolved (required by the adaptive schedule).  For
        time-indexed schedules (linear/cosine/step) λ is the same as if it
        had been resolved up-front and the result is mathematically
        identical to a single backward on ``λ · current_loss + replay_loss``
        (modulo float-summation order).
        """
        # Pass 1: g_new
        current_loss.backward(retain_graph=True)
        g_new_flat = torch.cat([p.grad.detach().flatten() for p in self._params])
        g_new_norm = torch.linalg.norm(g_new_flat)

        # Pass 2: g_replay (on a clean .grad so it isn't contaminated by g_new)
        self.optimizer.zero_grad()
        replay_loss.backward()
        g_replay_flat = torch.cat([p.grad.detach().flatten() for p in self._params])
        g_replay_norm = torch.linalg.norm(g_replay_flat)

        # Magnitude-bias readout (Chapter 3.2.1):  ‖g_new‖ / ‖g_replay‖.
        self._last_grad_ratio = (g_new_norm / (g_replay_norm + 1e-8)).item()
        # Buffer-fidelity diagnostics: g_replay vs g_true (full past data).
        self._record_buffer_diagnostics(g_replay_flat, g_replay_norm)

        # Resolve λ now that both norms are known (adaptive uses them; time-
        # indexed schedules ignore them).
        lam = self._curriculum_lambda(
            task_id,
            g_new_norm=g_new_norm.item(),
            g_replay_norm=g_replay_norm.item(),
        )

        # Compose joint gradient λ·g_new + g_replay and write into .grad.
        g_joint = lam * g_new_flat + g_replay_flat
        offset = 0
        for p in self._params:
            numel = p.numel()
            p.grad = g_joint[offset: offset + numel].view_as(p).clone()
            offset += numel

        self._step_optimizer(lr_factor)
        # Logged loss tracks L_λ — what the optimiser actually descended.
        return {"loss": (lam * current_loss + replay_loss).item(), "replay_loss": replay_loss.item()}

    def _balanced_step(
        self,
        current_loss: torch.Tensor,
        replay_loss: torch.Tensor,
        task_id: int,
        lr_factor: float = 1.0,
    ) -> Dict[str, float]:
        """Two-pass ER with configurable gradient combining and normalisation.

        λ is resolved after both gradient norms are computed so the adaptive
        schedule can use the current-step ratio.  When lam == 1.0 every
        formula reduces to the original balanced step; when lam == 0 the
        joint direction is the pure replay direction (after final unit-
        normalisation).
        """
        # ── Compute separate gradients ─────────────────────────────────
        current_loss.backward()
        g_new_flat = torch.cat([p.grad.detach().flatten() for p in self._params])

        self.optimizer.zero_grad()
        replay_loss.backward()
        g_replay_flat = torch.cat([p.grad.detach().flatten() for p in self._params])

        g_new_norm    = torch.linalg.norm(g_new_flat)
        g_replay_norm = torch.linalg.norm(g_replay_flat)
        # Magnitude-bias readout (Chapter 3.2.1):  ‖g_new‖ / ‖g_replay‖.
        self._last_grad_ratio = (g_new_norm / (g_replay_norm + 1e-8)).item()
        # Buffer-fidelity diagnostics: g_replay vs g_true (full past data).
        self._record_buffer_diagnostics(g_replay_flat, g_replay_norm)

        # Resolve λ now (adaptive schedule consumes the gradient norms;
        # time-indexed schedules ignore them).
        lam = self._curriculum_lambda(
            task_id,
            g_new_norm=g_new_norm.item(),
            g_replay_norm=g_replay_norm.item(),
        )

        # ── Weights ────────────────────────────────────────────────────
        if self._task_weighted:
            w_new    = 1.0 / (task_id + 1)
            w_replay = float(task_id) / (task_id + 1)
        else:
            w_new = w_replay = 0.5

        # ── Combine, then normalise joint once ─────────────────────────
        # Curriculum: scale the new-task contribution by λ.  At λ=0 this
        # reduces to a pure replay direction; at λ=1 it matches the original
        # balanced combine bit-for-bit.
        if self._normalize_components:
            g_joint = (lam * w_new * (g_new_flat    / (g_new_norm    + 1e-12))
                     + w_replay    * (g_replay_flat / (g_replay_norm + 1e-12)))
        else:
            g_joint = lam * w_new * g_new_flat + w_replay * g_replay_flat

        g_joint = g_joint / (torch.linalg.norm(g_joint) + 1e-12)

        # ── Write normalised direction into .grad for optimizer.step() ─
        offset = 0
        for p in self._params:
            numel = p.numel()
            p.grad = g_joint[offset: offset + numel].view_as(p).clone()
            offset += numel

        self._step_optimizer(lr_factor)
        # Logged loss tracks L_λ — what the optimiser actually descended.
        return {"loss": (lam * current_loss + replay_loss).item(), "replay_loss": replay_loss.item()}

    def get_last_grad_ratio(self) -> float:
        """Return ``‖g_new‖ / ‖g_replay‖`` from the last ``observe()`` call.

        This is the *update-side* magnitude-bias readout (Chapter 3.2.1):
        how much the unnormalised ER sum is dominated by the current-task
        gradient relative to the replay-buffer gradient.  Diverges at θ_0*
        (where ‖g_replay‖ → 0 by stationarity).

        Returns ``1.0`` on task 0 (no replay; ratio undefined).
        """
        return self._last_grad_ratio

    def get_last_true_grad_cosine(self) -> float:
        """Return ``cos(g_replay, g_true)`` from the last ``observe()`` call.

        ``g_true`` is the gradient computed on the diagnostic batch passed
        via ``set_diagnostic_data()`` — the full past-task training set,
        which gives the exact empirical past-task gradient at the current
        parameters (zero sampling noise).  ``g_replay`` is the buffer-based
        mini-batch gradient that the method actually used this step.

        This cosine is the *directional bias* readout (Chapter 3.2.2): the
        angle between what the buffer told the method to do and what the
        true past-task gradient would have asked for.  ``1.0`` = no
        directional bias; ``< 1`` = buffer is rotated off the true past
        gradient (sampling noise + composition bias).

        Named ``true_grad_*`` (rather than just ``grad_*``) to disambiguate
        from ``get_last_grad_ratio()``, which compares replay to the
        *current-task* gradient (the update-side asymmetry of Chapter 3.2.1)
        and has nothing to do with g_true.

        Returns ``0.0`` on task 0 or when no diagnostic data has been set.
        """
        return self._last_true_grad_cosine

    def get_last_true_grad_mag_ratio(self) -> float:
        """Return ``‖g_replay‖ / ‖g_true‖`` from the last ``observe()`` call.

        Magnitude bias of the *buffer estimate* relative to the true past
        gradient (cf. ``set_diagnostic_data()`` for the definition of
        g_true).  Distinct from ``get_last_grad_ratio()``, which compares
        replay to current-task gradient magnitudes; this metric compares
        replay to the gradient it is meant to estimate.

        ``1.0`` = unbiased magnitude; ``< 1`` = buffer underestimates the
        true past-gradient magnitude.

        Returns ``1.0`` on task 0 or when no diagnostic data has been set.
        """
        return self._last_true_grad_mag_ratio

    def end_task(self, task_id: int, train_loader: DataLoader) -> None:
        """Populate the replay buffer with data from the just-finished task.

        Each sample is inserted individually into the ``ReservoirBuffer`` so
        that the reservoir-sampling acceptance probability is computed
        correctly (it depends on the global ``_n_seen`` counter in the
        buffer, which increments per sample, not per batch).

        Parameters
        ----------
        task_id : int
            Zero-based index of the task that just finished.
        train_loader : DataLoader
            Training loader for the completed task.  Iterated once in
            inference mode — no gradients are computed here.
        """
        with torch.no_grad():
            for x_batch, y_batch in train_loader:
                # CPU tensors expected by ReservoirBuffer; move off GPU if
                # the loader put them on device (unusual but possible).
                x_batch = x_batch.cpu()
                y_batch = y_batch.cpu()
                for i in range(x_batch.size(0)):
                    self.buffer.add(x_batch[i], y_batch[i], task_id)

