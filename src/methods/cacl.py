"""CACL (Curvature-Aware Continual Learning) — full Algorithm 1.  T5.5.

On each training step for task > 0 the method:

  1. Forms the joint loss over the current mini-batch and a replay batch.
  2. Computes the joint gradient g_joint.
  3. Every ``amortize_K`` steps, runs Lanczos on the replay (or joint)
     Hessian to extract the top-d eigenvectors V_danger (the "dangerous
     subspace" of high replay curvature).
  4. Deflates g_hat_joint by removing its projection onto V_danger.
  5. Cone-projects the deflated gradient back within alpha_deg of g_hat_joint.
  6. Scales d_star by the trust-region radius η and applies the update.
  7. Measures actual vs. predicted loss reduction; adapts η.

For task 0 (empty buffer), CACL degrades to plain SGD to avoid computing
HVPs of a zero-loss function (which produces garbage eigenvectors).

Sign convention note:
    ``d_star`` from ``cone_project`` is a unit vector in the gradient (ascent)
    direction — ``params -= eta * d_star`` performs gradient descent.  The
    directional derivative passed to ``TrustRegion.compute_ratio`` is the
    *negated* dot product  ``-(g_joint · d_star) < 0``  to match the formula
    ``predicted_reduction = -directional_deriv * eta > 0`` expected by the
    trust-region code (verified by ``tests/test_trust_region.py``).

References:
    CACL paper (draft) — Algorithm 1, Equations 14, 16, 17.

Usage::

    buffer = ReservoirBuffer(total_budget=500)
    method = CACL(model, cfg, buffer)
    for task_id, train_loader, test_loaders in dataset.task_iterator():
        for x, y in train_loader:
            result = method.observe(x, y, task_id)
            diags  = method.get_step_diagnostics()
        method.end_task(task_id, train_loader)
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.methods.base_method import BaseMethod
from src.data.memory_buffer import ReservoirBuffer
from src.optim.lanczos import hvp, lanczos
from src.optim.cone_projection import deflate, cone_project
from src.optim.trust_region import TrustRegion


class CACL(BaseMethod):
    """Curvature-Aware Continual Learning (CACL) — full Algorithm 1.

    Parameters
    ----------
    model : nn.Module
        The neural network being trained.
    cfg : omegaconf.DictConfig
        **Full** Hydra config (not just the method sub-config).  CACL reads
        ``cfg.training.lr`` for the fallback learning rate when the trust
        region is disabled, and ``cfg.method`` for all CACL hyper-parameters
        (cone, lanczos, hessian, trust_region).
    buffer : ReservoirBuffer
        Shared replay buffer.  Populated by ``end_task()`` and sampled inside
        ``observe()`` to form the replay mini-batch.
    """

    def __init__(self, model: nn.Module, cfg: Any, buffer: ReservoirBuffer) -> None:
        # BaseMethod stores model and the *method* sub-config (cfg.method)
        super().__init__(model, cfg.method)

        self._full_cfg = cfg
        self.buffer = buffer

        # Cache trainable parameter list and total parameter count once.
        self._params: List[nn.Parameter] = [
            p for p in model.parameters() if p.requires_grad
        ]
        self._n_params: int = sum(p.numel() for p in self._params)
        self._device: torch.device = next(model.parameters()).device

        # ── Unpack method-level hyper-parameters ──────────────────────────
        mcfg = cfg.method

        gb = cfg.method.grad_balance
        self._normalize_components: bool = bool(gb.normalize_components)
        self._task_weighted: bool = bool(gb.task_weighted)


        # Cone projection
        self._alpha_deg: float = float(mcfg.cone.alpha_deg)

        # Lanczos / Hessian decomposition
        self._k: int = int(mcfg.lanczos.k)                    # iterations
        self._d: int = int(mcfg.lanczos.d)                    # top-d eigenvectors
        self._amortize_K: int = int(mcfg.lanczos.amortize_K)  # recompute interval
        # "replay" | "joint" — which loss Hessian to Lanczos-decompose
        self._hessian_target: str = str(mcfg.hessian.target)

        # Trust region
        self._trust_enabled: bool = bool(mcfg.trust_region.enabled)
        # "joint" | "replay" — which loss the trust ratio evaluates
        self._tr_loss_target: str = str(mcfg.trust_region.loss_target)

        if self._trust_enabled:
            # Store contract_thresh for the accepted flag in diagnostics.
            self._contract_thresh: float = float(mcfg.trust_region.contract_threshold)
            self._initial_trust_radius: float = float(mcfg.trust_region.initial_radius)
            self.trust_region = TrustRegion(
                initial_radius=self._initial_trust_radius,
                # Note: YAML keys are *_threshold; TrustRegion args are *_thresh
                expand_thresh=float(mcfg.trust_region.expand_threshold),
                contract_thresh=self._contract_thresh,
                min_r=float(mcfg.trust_region.min_radius),
                max_r=float(mcfg.trust_region.max_radius),
            )
        else:
            # Ablation A2: fixed learning rate when trust region is disabled.
            self._fixed_lr: float = float(cfg.training.lr)
            self._contract_thresh = 0.0  # unused; kept for uniform attribute access

        self.loss_fn = nn.CrossEntropyLoss()

        # ── λ-curriculum (homotopy on new-task loss weight) ───────────────
        # Defensive parse: older configs / unit-test SimpleNamespaces may not
        # carry the lambda_curriculum block — treat that as "disabled".
        lc = getattr(mcfg, "lambda_curriculum", None)
        if lc is None:
            self._lc_enabled: bool = False
            self._lc_ramp_steps: int = 0
            self._lc_schedule: str = "linear"
        else:
            self._lc_enabled = bool(lc.enabled)
            self._lc_ramp_steps = int(lc.ramp_steps)
            self._lc_schedule = str(lc.schedule)

        # ── Mutable algorithm state ────────────────────────────────────────
        # Global step counter across all tasks; drives Lanczos amortization.
        self._step_count: int = 0
        # Current task index — set at the start of each observe() call so
        # _cacl_step can compute task-proportional gradient weights.
        self._current_task_id: int = 0
        # Steps within the current task (resets on task transition); used to
        # evaluate the λ-curriculum schedule.
        self._task_step: int = 0
        # Tracks the previous task_id passed to observe() so we can detect
        # transitions (and reset _task_step).  -1 sentinel triggers a reset
        # on the first observe() call.
        self._task_id_at_last_observe: int = -1

        # Cached dangerous subspace from the most recent Lanczos run.
        # Initialised to shape (D, 0) so that deflate() is a no-op until
        # the first Lanczos step runs on task 1.
        self._V_danger: torch.Tensor = torch.zeros(
            self._n_params, 0, device=self._device
        )
        self._eigenvalues: torch.Tensor = torch.zeros(0, device=self._device)

        # Gradient ratio ‖g_new‖ / ‖g_replay‖ from the last CACL step.
        # Values >> 1 indicate new-task gradient dominates (task boundary).
        self._last_grad_ratio: float = 1.0

        # Diagnostics dict populated after every observe() call.
        # NOTE: get_step_diagnostics() returns exactly 7 keys (spec T5.5).
        # Extended state for T6.3 diagnostics writers is exposed via separate
        # methods: get_last_eigenvalues(), get_last_lanczos_updated(),
        # get_last_trust_accepted().
        self._last_diagnostics: Dict[str, Any] = {}

        # ── T6.3 extended state (not in get_step_diagnostics) ─────────────
        # Last eigenvalue list — updated on every Lanczos-recompute step.
        self._last_eigenvalues_list: List[float] = []
        # Whether Lanczos ran on the most recent observe() call.
        self._last_lanczos_updated: bool = False
        # Whether the last CACL step was "accepted" by the trust region
        # (rho >= contract_thresh).  True when trust region is disabled.
        self._last_trust_accepted: bool = False

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _unflat(self, v_flat: torch.Tensor) -> List[torch.Tensor]:
        """Reshape a flat parameter vector into a list matching self._params shapes."""
        parts: List[torch.Tensor] = []
        offset = 0
        for p in self._params:
            numel = p.numel()
            parts.append(v_flat[offset : offset + numel].view(p.shape))
            offset += numel
        return parts

    def _make_hvp_fn(
        self,
        x_current: torch.Tensor,
        y_current: torch.Tensor,
        x_replay: torch.Tensor,
        y_replay: torch.Tensor,
    ):
        """Build a flat-in / flat-out HVP oracle for Lanczos.

        Each call to the returned ``hvp_fn`` recomputes the target loss from
        scratch — necessary because ``torch.autograd.grad`` with
        ``create_graph=True`` consumes the graph after the second backward,
        so no graph can be reused across Lanczos iterations.

        Parameters
        ----------
        x_current, y_current : torch.Tensor
            Current-task mini-batch (used only when ``hessian.target == "joint"``).
        x_replay, y_replay : torch.Tensor
            Replay mini-batch (always used).

        Returns
        -------
        Callable[[Tensor], Tensor]
            Oracle ``v_flat → Hv_flat`` with signature expected by
            :func:`~src.optim.lanczos.lanczos`.
        """
        model = self.model
        params = self._params
        loss_fn = self.loss_fn
        target = self._hessian_target

        # loss_fn signature: (params_ignored) → scalar Tensor.
        # The model uses its own stored parameters (= params); the argument is
        # accepted but ignored (standard Pearlmutter pattern).
        def _replay_loss_fn(p_ignored: Any) -> torch.Tensor:
            return loss_fn(model(x_replay), y_replay)

        def _joint_loss_fn(p_ignored: Any) -> torch.Tensor:
            return (
                loss_fn(model(x_current), y_current)
                + loss_fn(model(x_replay), y_replay)
            )

        base_loss_fn = _replay_loss_fn if target == "replay" else _joint_loss_fn

        def hvp_fn(v_flat: torch.Tensor) -> torch.Tensor:
            """Apply H to a flat vector (flat in → flat out)."""
            v_list = self._unflat(v_flat)
            return hvp(base_loss_fn, params, v_list)

        return hvp_fn

    def _get_step_size(self) -> float:
        """Return the current step size η."""
        return (
            self.trust_region.get_step_size()
            if self._trust_enabled
            else self._fixed_lr
        )

    def _apply_update(self, d_star: torch.Tensor, eta: float) -> None:
        """Gradient-descent update: params -= η · d_star."""
        with torch.no_grad():
            offset = 0
            for p in self._params:
                numel = p.numel()
                p.data -= eta * d_star[offset : offset + numel].view_as(p)
                offset += numel

    def _eval_tr_loss(
        self,
        x_current: torch.Tensor,
        y_current: torch.Tensor,
        x_replay: torch.Tensor,
        y_replay: torch.Tensor,
        lam: float = 1.0,
    ) -> float:
        """Evaluate the trust-ratio target loss with no gradients.

        Which loss is used depends on ``trust_region.loss_target`` in the
        config: ``"joint"`` (λ · current + replay CE) or ``"replay"`` (replay
        CE only).  ``lam`` is the curriculum weight and must match what the
        gradient step descended; the trust ratio is invalid otherwise.
        """
        with torch.no_grad():
            if self._tr_loss_target == "replay":
                return self.loss_fn(self.model(x_replay), y_replay).item()
            # default: "joint" — must mirror the L_λ that was just descended.
            return (
                lam * self.loss_fn(self.model(x_current), y_current)
                + self.loss_fn(self.model(x_replay), y_replay)
            ).item()

    def _curriculum_lambda(self, task_id: int) -> float:
        """Curriculum weight λ on the new-task loss for the current step.

        Returns ``1.0`` (no curriculum) when the curriculum is disabled, on
        task 0 (no past tasks to preserve), or after the ramp has completed.
        Otherwise ramps from 0 → 1 over the first ``ramp_steps`` of the task
        using the configured schedule (``linear`` or ``cosine``).
        """
        if (
            not self._lc_enabled
            or task_id == 0
            or self._lc_ramp_steps <= 0
            or self._task_step >= self._lc_ramp_steps
        ):
            return 1.0
        progress = self._task_step / self._lc_ramp_steps
        if self._lc_schedule == "cosine":
            return 0.5 * (1.0 - math.cos(math.pi * progress))
        return progress  # linear (default)

    def _zero_diagnostics(self) -> Dict[str, Any]:
        """All-zero diagnostics dict for task-0 or degenerate (zero-gradient) steps.

        Also resets the T6.3 extended state: no Lanczos update, step accepted.
        """
        tr_radius: float = (
            self.trust_region.radius if self._trust_enabled else self._fixed_lr
        )
        # Reset T6.3 extended state — no Lanczos ran, step trivially accepted.
        self._last_lanczos_updated = False
        self._last_trust_accepted = True  # SGD / zero-grad step is always "taken"
        self._last_grad_ratio = 1.0       # no mismatch on task 0
        return {
            "cone_fallback": False,
            "trust_radius": tr_radius,
            "trust_rho": 0.0,
            "top_eigenvalue": 0.0,
            "eigenvalue_ratio": 0.0,
            "g_deflated_norm": 0.0,
            "beta": 0.0,
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
        """One CACL gradient step.

        Falls back to plain SGD on task 0 (no replay data) to avoid computing
        HVPs of a zero-loss function, which would produce garbage eigenvectors.
        On subsequent tasks the full Algorithm 1 runs.

        Parameters
        ----------
        x_batch : torch.Tensor
            Current-task inputs, shape ``(B, ...)``.
        y_batch : torch.Tensor
            Current-task labels, shape ``(B,)``.
        task_id : int
            Zero-based index of the task currently being trained.

        Returns
        -------
        dict
            On task 0: ``{"loss": float, "eta": float}``.
            On task > 0: ``{"loss": float, "rho": float, "eta": float}``.
        """
        self.model.train()
        x_batch = x_batch.to(self._device)
        y_batch = y_batch.to(self._device)

        # Detect task transition: reset within-task step counter so the
        # λ-curriculum schedule restarts at λ=0 for each new task.
        if task_id != self._task_id_at_last_observe:
            self._task_step = 0
            self._task_id_at_last_observe = task_id

        self._current_task_id = task_id

        if task_id == 0 or len(self.buffer) == 0:
            result = self._sgd_step(x_batch, y_batch)
        else:
            result = self._cacl_step(x_batch, y_batch)

        self._task_step += 1
        return result

    def _sgd_step(
        self, x_batch: torch.Tensor, y_batch: torch.Tensor
    ) -> Dict[str, float]:
        """Plain SGD fallback for task 0 / empty buffer."""
        for p in self._params:
            p.grad = None

        logits = self.model(x_batch)
        loss = self.loss_fn(logits, y_batch)

        grads = torch.autograd.grad(loss, self._params, create_graph=False)
        g_flat = torch.cat([g.flatten() for g in grads])

        eta = self._get_step_size()
        with torch.no_grad():
            offset = 0
            for p in self._params:
                numel = p.numel()
                p.data -= eta * g_flat[offset : offset + numel].view_as(p)
                offset += numel

        self._step_count += 1
        self._last_diagnostics = self._zero_diagnostics()
        return {"loss": loss.item(), "eta": eta}

    def _cacl_step(
        self, x_batch: torch.Tensor, y_batch: torch.Tensor
    ) -> Dict[str, float]:
        """Full CACL Algorithm 1 step (task > 0, non-empty buffer)."""

        # ── Steps 1-2: Sample replay; compute joint loss ───────────────────
        x_replay, y_replay, _ = self.buffer.sample(x_batch.size(0))
        x_replay = x_replay.to(self._device)
        y_replay = y_replay.to(self._device)

        for p in self._params:
            p.grad = None

        logits_cur = self.model(x_batch)
        current_loss = self.loss_fn(logits_cur, y_batch)

        logits_rep = self.model(x_replay)
        replay_loss = self.loss_fn(logits_rep, y_replay)

        # ── Curriculum weight on new-task contribution ────────────────────
        # Effective loss being descended this step: L_λ = L_replay + λ · L_current.
        # When the curriculum is disabled (lam == 1.0) every quantity below
        # reduces to the original ER/CACL formulas exactly.
        lam: float = self._curriculum_lambda(self._current_task_id)

        old_joint_loss: float = current_loss.item() + replay_loss.item()
        # Effective L_λ at the *current* parameters; mirrors what _eval_tr_loss
        # will recompute after the step so that the trust ratio is consistent.
        old_tr_joint_loss: float = lam * current_loss.item() + replay_loss.item()

        # Trust-ratio "before" loss (joint or replay only, per config).  When
        # loss_target=="joint" this must mirror L_λ — the loss actually being
        # descended — not the unweighted joint, otherwise ρ is biased.
        old_tr_loss: float = (
            replay_loss.item()
            if self._tr_loss_target == "replay"
            else old_tr_joint_loss
        )

        # ── Step 3: Separate gradients; combine with configurable balancing ──
        # create_graph=False: the Lanczos HVP oracle rebuilds its own graph on
        # each call, so we do not need to keep this graph alive.
        g_new_grads = torch.autograd.grad(
            current_loss, self._params, create_graph=False
        )
        g_new_flat = torch.cat([g.flatten() for g in g_new_grads])

        g_replay_grads = torch.autograd.grad(
            replay_loss, self._params, create_graph=False
        )
        g_replay_flat = torch.cat([g.flatten() for g in g_replay_grads])

        g_new_norm    = torch.linalg.norm(g_new_flat)
        g_replay_norm = torch.linalg.norm(g_replay_flat)

        # Track ratio for diagnostics — values >> 1 reveal task-boundary
        # domination, which is the primary driver of the stability gap.
        self._last_grad_ratio = (g_new_norm / (g_replay_norm + 1e-8)).item()

        # ── Weights ────────────────────────────────────────────────────
        if self._task_weighted:
            w_new    = 1.0 / (self._current_task_id + 1)
            w_replay = float(self._current_task_id) / (self._current_task_id + 1)
        else:
            w_new = w_replay = 0.5

        # ── Combine, then normalise joint once ─────────────────────────
        # normalize_components=True:  normalise each gradient individually,
        #     then combine — direction only, magnitudes discarded.
        # normalize_components=False: combine raw (weighted) gradients,
        #     then normalise the joint — magnitudes influence the direction.
        # Curriculum: scale the new-task contribution by λ.  When λ=0 the
        # joint direction reduces to the pure replay direction; when λ=1
        # this is identical to the original CACL combine.
        if self._normalize_components:
            g_joint_flat = (lam * w_new * (g_new_flat    / (g_new_norm    + 1e-12))
                          + w_replay    * (g_replay_flat / (g_replay_norm + 1e-12)))
        else:
            g_joint_flat = lam * w_new * g_new_flat + w_replay * g_replay_flat

        # Raw weighted sum for the trust-region directional derivative —
        # must track ∇L_λ = ∇L_replay + λ · ∇L_current, the gradient of the
        # loss actually being descended.
        g_joint_raw = lam * g_new_flat + g_replay_flat

        g_norm: float = torch.linalg.norm(g_joint_flat).item()
        if g_norm < 1e-12:
            # Numerically zero gradient — skip update, hold radius constant.
            self._step_count += 1
            self._last_diagnostics = self._zero_diagnostics()
            return {
                "loss": old_joint_loss,
                "rho": 0.0,
                "eta": self._get_step_size(),
            }

        g_hat_joint = g_joint_flat / g_norm  # unit-norm direction

        # ── Step 4: Amortised Lanczos (every amortize_K steps) ────────────
        lanczos_ran: bool = (self._step_count % self._amortize_K == 0)
        if lanczos_ran:
            hvp_fn = self._make_hvp_fn(x_batch, y_batch, x_replay, y_replay)
            eigenvalues, V_danger = lanczos(
                hvp_fn,
                dim=self._n_params,
                k=self._k,
                d=self._d,
                device=self._device,
                dtype=torch.float32,
            )
            # Cache for subsequent (non-Lanczos) steps
            self._V_danger = V_danger        # shape (D, d_actual)
            self._eigenvalues = eigenvalues  # shape (d_actual,)
            # T6.3: persist eigenvalue list for CACLDiagnosticsWriter
            self._last_eigenvalues_list = eigenvalues.tolist()

        # ── Step 5: Deflate joint gradient by dangerous subspace ──────────
        g_deflated = deflate(g_hat_joint, self._V_danger)
        g_deflated_norm: float = torch.linalg.norm(g_deflated).item()

        # ── Step 6: Cone project ───────────────────────────────────────────
        # d_star is a unit vector in the gradient (ascent) direction;
        # params -= eta * d_star performs gradient descent.
        d_star, fallback = cone_project(g_hat_joint, g_deflated, self._alpha_deg)

        # ── Step 7: Step size ──────────────────────────────────────────────
        eta = self._get_step_size()

        # ── Step 8: Apply parameter update ────────────────────────────────
        self._apply_update(d_star, eta)

        # ── Step 9: New loss for trust ratio (forward pass, no grad) ──────
        # Pass the same lam as the step used so the trust ratio compares like
        # for like (L_λ before vs. L_λ after).
        new_tr_loss = self._eval_tr_loss(x_batch, y_batch, x_replay, y_replay, lam)

        # ── Steps 10-11: Trust-region ratio + radius update ───────────────
        # Directional derivative for compute_ratio: -(g_joint · d_star).
        #
        # d_star is in the ascent direction, so  g_joint · d_star > 0.
        # TrustRegion.compute_ratio expects a NEGATIVE directional_deriv for a
        # descent step (predicted_reduction = -deriv * eta > 0).  Negating
        # converts our positive dot product to the expected negative value.
        directional_deriv: float = -(torch.dot(g_joint_raw, d_star).item())

        rho_val: float = 0.0
        if self._trust_enabled:
            rho: Optional[float] = self.trust_region.compute_ratio(
                loss_new=new_tr_loss,
                loss_old=old_tr_loss,
                directional_deriv=directional_deriv,
                step_size=eta,
            )
            self.trust_region.update_radius(rho)
            rho_val = rho if rho is not None else 0.0

        # ── Diagnostics ───────────────────────────────────────────────────
        n_evals = len(self._eigenvalues)
        top_eigenvalue: float = self._eigenvalues[0].item() if n_evals > 0 else 0.0

        # Condition-number-like ratio: top / bottom Ritz eigenvalue.
        if n_evals > 1:
            bottom = self._eigenvalues[-1].abs().item()
            eigenvalue_ratio: float = top_eigenvalue / (bottom + 1e-12)
        else:
            eigenvalue_ratio = 1.0

        # beta: interpolation coefficient used in cone_project binary search.
        # 0.0 → deflated direction already inside cone (no blending used).
        # 1.0 → fallback triggered (binary search or degenerate case).
        beta: float = 1.0 if fallback else 0.0

        tr_radius: float = (
            self.trust_region.radius if self._trust_enabled else self._fixed_lr
        )

        # NOTE: exactly 7 keys — do NOT add keys here; use the T6.3 extended
        # state methods (get_last_eigenvalues / get_last_lanczos_updated /
        # get_last_trust_accepted) instead of growing this dict.
        self._last_diagnostics = {
            "cone_fallback": fallback,
            "trust_radius": tr_radius,
            "trust_rho": rho_val,
            "top_eigenvalue": top_eigenvalue,
            "eigenvalue_ratio": eigenvalue_ratio,
            "g_deflated_norm": g_deflated_norm,
            "beta": beta,
        }

        # ── T6.3 extended state ────────────────────────────────────────────
        self._last_lanczos_updated = lanczos_ran
        # A step is "accepted" when its trust-ratio was not below the contraction
        # threshold (i.e., the radius was not shrunk).  When trust is disabled,
        # every step is accepted by definition (fixed lr, no rejection).
        if self._trust_enabled:
            self._last_trust_accepted = (rho is not None) and (
                rho >= self._contract_thresh
            )
        else:
            self._last_trust_accepted = True

        self._step_count += 1

        return {
            "loss": old_joint_loss,
            "rho": rho_val,
            "eta": eta,
        }

    def end_task(self, task_id: int, train_loader: DataLoader) -> None:
        """Populate the replay buffer with data from the completed task.

        Mirrors ER / GEM: each sample is inserted individually so that the
        reservoir-sampling acceptance probability is computed correctly (it
        depends on the buffer's ``_n_seen`` counter, which increments per
        sample).

        Parameters
        ----------
        task_id : int
            Zero-based index of the task that just finished.
        train_loader : DataLoader
            Training loader for the completed task.
        """
        with torch.no_grad():
            for x_batch, y_batch in train_loader:
                x_batch = x_batch.cpu()
                y_batch = y_batch.cpu()
                for i in range(x_batch.size(0)):
                    self.buffer.add(x_batch[i], y_batch[i], task_id)

        # Reset the trust radius to its initial value so each task starts from
        # the same cautious baseline rather than inheriting a radius that may
        # have expanded or collapsed during the task that just ended.
        if self._trust_enabled:
            self.trust_region.radius = self._initial_trust_radius

    def get_step_diagnostics(self) -> Dict[str, Any]:
        """Return CACL-specific diagnostics from the last ``observe()`` call.

        Returns
        -------
        dict with keys:
            cone_fallback (bool): Whether the cone-projection fallback fired.
            trust_radius (float): Current trust-region radius η.
            trust_rho (float): Last trust ratio ρ (0.0 on task 0 or skip).
            top_eigenvalue (float): Largest Ritz value from the last Lanczos.
            eigenvalue_ratio (float): top / bottom Ritz eigenvalue ratio.
            g_deflated_norm (float): L2 norm of the deflated gradient.
            beta (float): Cone interpolation coefficient (0 = no fallback,
                1 = fallback triggered).

        Note: This dict has exactly 7 keys (per spec T5.5).  Extended state for
        T6.3 diagnostic writers is exposed via the ``get_last_*`` methods below.
        """
        return dict(self._last_diagnostics)

    # ------------------------------------------------------------------
    # T6.3 extended accessors (not part of the T5.5 spec contract)
    # ------------------------------------------------------------------

    def get_last_eigenvalues(self) -> List[float]:
        """Return the eigenvalues computed on the most recent Lanczos step.

        Returns the top-d Ritz values as a Python list of floats.  Returns an
        empty list before the first Lanczos step has run.  The caller should
        guard with :meth:`get_last_lanczos_updated` to avoid logging stale
        values on non-Lanczos steps.

        Returns
        -------
        list of float
            Top-d eigenvalues, descending order.  Empty before first Lanczos run.
        """
        return list(self._last_eigenvalues_list)

    def get_last_lanczos_updated(self) -> bool:
        """Return whether Lanczos was recomputed on the last ``observe()`` call.

        Returns
        -------
        bool
            ``True`` iff the last step was a Lanczos-recompute step
            (i.e., ``step_count % amortize_K == 0`` at the time of the call).
            Always ``False`` on task-0 (SGD fallback) steps.
        """
        return self._last_lanczos_updated

    def get_last_trust_accepted(self) -> bool:
        """Return whether the last step was accepted by the trust region.

        A step is "accepted" when its trust ratio ρ ≥ contract_threshold,
        meaning the actual loss reduction was at least a fraction of the
        predicted reduction and the radius was not penalised (shrunk).
        Always ``True`` when the trust region is disabled (fixed lr).

        Returns
        -------
        bool
        """
        return self._last_trust_accepted

    def get_last_grad_ratio(self) -> float:
        """Return ‖g_new‖ / ‖g_replay‖ from the last CACL step.

        Values >> 1 indicate the new-task gradient dominates (typical at task
        boundaries and the primary driver of the stability gap).  Values ≈ 1
        indicate both tasks contribute equally to the direction.  Always 1.0
        on task-0 steps (no replay, so the ratio is undefined).

        Returns
        -------
        float
        """
        return self._last_grad_ratio
