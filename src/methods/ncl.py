"""NCL — Natural Continual Learning (Kao et al., NeurIPS 2021).

This file implements the algorithm of

    Kao, Jensen, van de Ven, Bernacchia & Hennequin (2021)
    "Natural Continual Learning: Success is a Journey, Not (Just) a Destination."
    https://arxiv.org/abs/2106.08085

Specifically: paper Eq. (8) — the K-FAC-preconditioned natural-gradient
update with the Bayesian rubber-band restoring term — and Algorithm 1 in
Appendix E (the pseudocode the implementation tracks line-for-line).

**Per-task objective.** With L_k(θ) the CE (negative log-likelihood) loss on
task k, and (μ_{k-1}, Λ_{k-1}) the prior mean and precision summarising the
posterior after tasks 1..k-1, the objective is the negative log Laplace
posterior (paper Eq. 4),

    L_post(θ) = L_k(θ) + (1/2) (θ - μ_{k-1})ᵀ Λ_{k-1} (θ - μ_{k-1}).

**The NCL update (paper Eq. 8).** Rather than descend ∇L_post directly, NCL
preconditions by the *prior covariance* Λ_{k-1}^{-1} (so that motion in
high-prior-curvature directions is damped):

    θ ← θ - η [ Λ_{k-1}^{-1} ∇L_k(θ) + (θ - μ_{k-1}) ]

The radius r of the trust-region derivation (paper Eq. 7) is absorbed into η;
Algorithm 1 in Appendix E does not clip the step. We report a `tr_scale`
diagnostic of the form min(1, r / ‖step‖_Λ) for instrumentation but do not
rescale the update.

**K-FAC factorisation.** For a Linear layer with weight W ∈ R^{d_out × d_in}
(and bias augmented as the (d_in+1)-th column when present), the layer's
Fisher factorises as F_W ≈ A ⊗ G with

    A = E[ā āᵀ]   (ā = [x; 1] for biased layers)        — (d_in[+1], d_in[+1])
    G = E[g_s g_sᵀ], g_s = ∂L/∂(Wx + b)                 — (d_out, d_out)

so the per-layer natural-gradient direction and prior contribution combine
into the single line

    [ΔW_nat | Δb_nat] = G⁻¹ [∇W | ∇b] A⁻¹ + ([W − W*] | [b − b*])

which is what is written back into `.grad` for the optimiser.

**Online prior update (paper Eq. 5).** After each task, A_k = A_{k-1} + Â_k
and G_k = G_{k-1} + Ĝ_k (additive Kronecker accumulation — a simplification
of the optimal `nearest_kf_sum` of Appendix G that is standard in K-FAC
implementations) and μ_k ← θ_k*.

**Empirical Fisher.** Â_k and Ĝ_k use the gradient of the loss against the
*true labels* (empirical Fisher), not labels sampled from the model
(the proper Fisher used in the paper's reference code). For classification
the two are very close; the empirical version avoids a sampling step and
is the common choice across K-FAC implementations.

**Non-Linear layers** (BatchNorm, LayerNorm, embeddings) are not K-FAC'd —
their gradients are passed through unchanged, since the prior precision
restricted to them is treated as zero.

Usage::

    method = NCL(model, cfg)
    for task_id, train_loader, test_loaders in dataset.task_iterator():
        for x, y in train_loader:
            result = method.observe(x, y, task_id)   # → {"loss": float, ...}
        method.end_task(task_id, train_loader)        # K-FAC factors + μ snapshot
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader

from src.methods.base_method import BaseMethod


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _damped_inv(M: Tensor, damping: float) -> Tensor:
    """Compute (M + ε·I)⁻¹ with a hard fallback on LinAlgError."""
    n = M.size(0)
    I = torch.eye(n, device=M.device, dtype=M.dtype)
    try:
        return torch.linalg.inv(M + damping * I)
    except torch.linalg.LinAlgError:
        return torch.linalg.inv(M + (damping * 100.0) * I)


def _flat_params(model: nn.Module) -> Tensor:
    """Detached, CPU-side flat snapshot of every trainable parameter."""
    return torch.cat([p.detach().reshape(-1).cpu()
                      for p in model.parameters() if p.requires_grad])


# ---------------------------------------------------------------------------
# NCL
# ---------------------------------------------------------------------------

class NCL(BaseMethod):
    """K-FAC NCL — paper Eq. (8) + Algorithm 1 in Appendix E."""

    def __init__(self, model: nn.Module, cfg) -> None:
        super().__init__(model, cfg.method)
        self._full_cfg = cfg

        ncl_cfg = cfg.method.ncl
        self._fisher_samples: int = int(ncl_cfg.fisher_samples)
        self._damping: float = float(ncl_cfg.damping)
        # Trust-region radius r from paper Eq. (7). Used only to scale the
        # `tr_scale` diagnostic; the update itself is not rescaled (Eq. 8
        # absorbs r into η).
        self._trust_radius: float = float(getattr(ncl_cfg, "trust_radius", 1.0))
        # α in p_w = α·I (paper Algorithm 1 line 6). Bounds Λ⁻¹ — without
        # an initial prior, low-curvature directions of F_0 send Λ⁻¹∇L to
        # arbitrarily large magnitudes on task 1 and the natural-gradient
        # update diverges. With Λ_0 = α·I, the eigenvalues of Λ_k⁻¹ are
        # bounded by 1/α uniformly.
        self._prior_init: float = float(getattr(ncl_cfg, "prior_init", 1.0))

        self._device: torch.device = next(model.parameters()).device

        # Only nn.Linear layers receive K-FAC treatment; everything else is
        # passed through with its raw gradient (zero prior precision).
        self._linear_layers: Dict[str, nn.Linear] = {
            name: module
            for name, module in model.named_modules()
            if isinstance(module, nn.Linear)
        }

        # Per-layer Kronecker factors of the accumulated prior precision
        # Λ_{k-1} ≈ ⊕_l (A_l ⊗ G_l). Initialised to √α·I per factor so
        # that A_l ⊗ G_l = α·I = p_w for each layer at construction time;
        # subsequent tasks accumulate additively (A_l ← A_l + Â_k, etc.).
        # Stored on CPU so the full prior survives even for large layers;
        # copied to `self._device` per-step inside `_apply_natural_gradient`.
        sqrt_alpha = self._prior_init ** 0.5
        self._kfac_A: Dict[str, Tensor] = {}
        self._kfac_G: Dict[str, Tensor] = {}
        for name, module in self._linear_layers.items():
            d_in_aug = module.in_features + (1 if module.bias is not None else 0)
            d_out = module.out_features
            self._kfac_A[name] = sqrt_alpha * torch.eye(d_in_aug)
            self._kfac_G[name] = sqrt_alpha * torch.eye(d_out)

        # Per-layer snapshots of μ_{k-1} — initialised to θ_init so that on
        # task 0 the rubber-band term (θ - μ_0) is well-defined and equals
        # zero at the first step (paper convention: Λ_0 = p_w, μ_0 = θ_init,
        # i.e. the prior is centred at the initialisation point).
        self._prior_W: Dict[str, Tensor] = {}
        self._prior_b: Dict[str, Tensor] = {}
        for name, module in self._linear_layers.items():
            self._prior_W[name] = module.weight.data.detach().cpu().clone()
            if module.bias is not None:
                self._prior_b[name] = module.bias.data.detach().cpu().clone()

        # Flat μ_{k-1} over *all* trainable parameters (not just Linear
        # ones). Snapshot of θ_init at construction; refreshed in end_task.
        self._prior_mean: Optional[Tensor] = _flat_params(model)

        # Whether at least one task has produced a real K-FAC Fisher. On
        # task 0 (no posterior accumulated yet), the natural-gradient
        # rewrite is suppressed and `observe()` reduces to plain SGD — the
        # p_w = α·I initial prior alone is not enough signal to regularise
        # against (with α=1 the rubber-band toward θ_init dominates the
        # gradient and prevents task-0 learning). The flag is set inside
        # `end_task`, so the K-FAC inversion and the (θ - μ) term both
        # come online only after a genuine task posterior exists.
        self._has_prior: bool = False

        # Diagnostics from the most recent observe() call.
        self._last_kl_proxy: float = 0.0
        self._last_tr_scale: float = 1.0

        # Momentum read from training config so the §4.6 momentum cross
        # (training.momentum=0.9) actually applies to NCL — Kao et al. (2021)
        # Algorithm 1 line 15 uses momentum on top of the natural-gradient
        # direction (ρ = 0.9 throughout their feedforward experiments).
        self.optimizer = torch.optim.SGD(
            model.parameters(),
            lr=float(cfg.training.lr),
            momentum=float(cfg.training.momentum),
            weight_decay=float(cfg.training.weight_decay),
        )
        self.loss_fn = nn.CrossEntropyLoss()

    # ------------------------------------------------------------------
    # Natural-gradient step (Eq. 8) — preconditioning + rubber-band
    # ------------------------------------------------------------------

    def _apply_natural_gradient(self) -> None:
        """In-place: rewrite each Linear layer's `.grad` to

            G⁻¹ [∇W | ∇b] A⁻¹  +  ([W - W*] | [b - b*])

        which, when multiplied by -η inside the optimiser step, gives the
        Eq. (8) update direction. On task 0 (`_kfac_A` empty) this is a no-op
        and the optimiser sees the raw gradient.

        Side effect: updates `self._last_kl_proxy` and `self._last_tr_scale`.
        """
        # Task 0: no real posterior yet → fall back to plain SGD.
        if not self._has_prior or not self._kfac_A:
            self._last_kl_proxy = 0.0
            self._last_tr_scale = 1.0
            return

        kl_proxy = 0.0          # (1/2) Σ_l tr(δ_l^T G_l δ_l A_l)
        step_lambda_sq = 0.0    # ‖step‖_Λ²  with step = Λ⁻¹∇L + (θ - μ)

        rewrites: List[Tuple[nn.Linear, Tensor, Optional[Tensor]]] = []

        for name, module in self._linear_layers.items():
            if name not in self._kfac_A or module.weight.grad is None:
                continue

            A = self._kfac_A[name].to(self._device)
            G = self._kfac_G[name].to(self._device)
            A_inv = _damped_inv(A, self._damping)
            G_inv = _damped_inv(G, self._damping)

            grad_W = module.weight.grad.data  # (d_out, d_in)

            # (θ - μ_{k-1}) for this layer. The prior is anchored at the
            # previous task's optimum; if this is the first task that ever
            # produced a prior the dict has the entry, otherwise δ = 0.
            if name in self._prior_W:
                delta_W = module.weight.data - self._prior_W[name].to(self._device)
            else:
                delta_W = torch.zeros_like(grad_W)

            has_bias = module.bias is not None and module.bias.grad is not None
            if has_bias:
                if name in self._prior_b:
                    delta_b = module.bias.data - self._prior_b[name].to(self._device)
                else:
                    delta_b = torch.zeros_like(module.bias.data)

                grad_aug = torch.cat(
                    [grad_W, module.bias.grad.data.unsqueeze(1)], dim=1
                )
                delta_aug = torch.cat([delta_W, delta_b.unsqueeze(1)], dim=1)
            else:
                grad_aug = grad_W
                delta_aug = delta_W

            # Eq. 8 update direction (the term inside the brackets):
            #   step_aug = Λ⁻¹ ∇L + (θ - μ) = G⁻¹ ∇L_aug A⁻¹ + δ_aug
            nat_aug = G_inv @ grad_aug @ A_inv
            step_aug = nat_aug + delta_aug

            # kl_proxy contribution: (1/2) δ^T Λ δ = (1/2) tr(δ^T G δ A).
            #   For a Kronecker-factored Λ = A ⊗ G (vec-column convention):
            #     vec(δ)^T (A ⊗ G) vec(δ) = tr(δ^T G δ A).
            kl_proxy += 0.5 * (delta_aug * (G @ delta_aug @ A)).sum().item()

            # ‖step‖_Λ² = step^T Λ step = tr(step^T G step A) — same identity.
            step_lambda_sq += (step_aug * (G @ step_aug @ A)).sum().item()

            rewrites.append((module, step_aug, None))

        # ---- write the effective gradients back ----
        for module, step_aug, _ in rewrites:
            if module.bias is not None and module.bias.grad is not None:
                module.weight.grad.data = step_aug[:, :-1].contiguous()
                module.bias.grad.data = step_aug[:, -1].contiguous()
            else:
                module.weight.grad.data = step_aug

        # ---- diagnostics ----
        self._last_kl_proxy = float(kl_proxy)
        if step_lambda_sq > 0.0 and self._trust_radius > 0.0:
            step_lambda = step_lambda_sq ** 0.5
            self._last_tr_scale = float(min(1.0, self._trust_radius / step_lambda))
        else:
            self._last_tr_scale = 1.0

    # ------------------------------------------------------------------
    # K-FAC factor estimation
    # ------------------------------------------------------------------

    def _compute_kfac_factors(
        self, train_loader: DataLoader
    ) -> Tuple[Dict[str, Optional[Tensor]], Dict[str, Optional[Tensor]]]:
        """Estimate Â and Ĝ for every Linear layer from `train_loader`.

        Â = (1/N) Σ_n ā_n ā_nᵀ  with ā = [x; 1] for biased layers
        Ĝ = (1/N) Σ_n g_{s,n} g_{s,n}ᵀ,  g_s = ∂L / ∂(Wx + b)

        Mean-reduced CE means the backward-hook gradient ``g_out[0][i]``
        equals ``g_{s,i} / B``; we multiply by B before forming the outer
        product, so the result is (1/N) Σ g_s g_sᵀ regardless of how the
        mini-batches were sized.
        """
        a_sum: Dict[str, Optional[Tensor]] = {n: None for n in self._linear_layers}
        g_sum: Dict[str, Optional[Tensor]] = {n: None for n in self._linear_layers}
        a_cnt: Dict[str, int] = {n: 0 for n in self._linear_layers}

        saved_inputs: Dict[str, Tensor] = {}
        hooks: List = []

        for name, module in self._linear_layers.items():

            def _fwd(mod: nn.Linear, inp, out, _n: str = name) -> None:
                a = inp[0].detach()
                if mod.bias is not None:
                    ones = torch.ones(a.size(0), 1, device=a.device, dtype=a.dtype)
                    a = torch.cat([a, ones], dim=1)
                saved_inputs[_n] = a

            def _bwd(mod: nn.Linear, g_in, g_out, _n: str = name) -> None:
                gs = g_out[0].detach()
                a = saved_inputs.pop(_n, None)
                if a is None:
                    return
                B = gs.size(0)
                gs_sample = gs * B   # undo CE mean-reduction → per-sample g_s
                A_batch = a.T @ a
                G_batch = gs_sample.T @ gs_sample
                if a_sum[_n] is None:
                    a_sum[_n] = A_batch
                    g_sum[_n] = G_batch
                else:
                    a_sum[_n] = a_sum[_n] + A_batch
                    g_sum[_n] = g_sum[_n] + G_batch
                a_cnt[_n] += B

            hooks.append(module.register_forward_hook(_fwd))
            hooks.append(module.register_full_backward_hook(_bwd))

        # `register_full_backward_hook` wraps the module's outputs in a
        # custom autograd function. Any inplace activation (relu_, silu_,
        # …) that follows would then try to modify a view of that wrapped
        # output and PyTorch forbids it. Temporarily disabling inplace on
        # all activation modules sidesteps the conflict without altering
        # the forward result.
        inplace_modules = [
            m for m in self.model.modules() if hasattr(m, "inplace") and m.inplace
        ]
        for m in inplace_modules:
            m.inplace = False

        self.model.eval()
        n_seen = 0
        try:
            for x_batch, y_batch in train_loader:
                if n_seen >= self._fisher_samples:
                    break
                x_batch = x_batch.to(self._device)
                y_batch = y_batch.to(self._device)
                self.optimizer.zero_grad()
                self.loss_fn(self.model(x_batch), y_batch).backward()
                n_seen += x_batch.size(0)
        finally:
            for h in hooks:
                h.remove()
            for m in inplace_modules:
                m.inplace = True
            self.model.train()

        A_out: Dict[str, Optional[Tensor]] = {}
        G_out: Dict[str, Optional[Tensor]] = {}
        for name in self._linear_layers:
            if a_sum[name] is None:
                A_out[name] = G_out[name] = None
            else:
                A_out[name] = (a_sum[name] / max(a_cnt[name], 1)).cpu()
                G_out[name] = (g_sum[name] / max(a_cnt[name], 1)).cpu()

        return A_out, G_out

    # ------------------------------------------------------------------
    # BaseMethod interface
    # ------------------------------------------------------------------

    def observe(
        self,
        x_batch: Tensor,
        y_batch: Tensor,
        task_id: int,
    ) -> Dict[str, float]:
        """One natural-gradient step (paper Eq. 8) on the current batch."""
        self.model.train()
        x_batch = x_batch.to(self._device)
        y_batch = y_batch.to(self._device)

        self.optimizer.zero_grad()
        loss = self.loss_fn(self.model(x_batch), y_batch)
        loss.backward()

        self._apply_natural_gradient()

        self.optimizer.step()
        return {"loss": loss.item()}

    def end_task(self, task_id: int, train_loader: DataLoader) -> None:
        """Accumulate K-FAC factors and snapshot μ_k.

        Implements Eq. 5 in the K-FAC-factored form:

            Λ_k ← Λ_{k-1} + F_k            (additive accumulation)
            A_k ← A_{k-1} + Â_k            (input correlation)
            G_k ← G_{k-1} + Ĝ_k            (output-gradient correlation)
            μ_k ← θ_k*                     (current weights are the new mean)
        """
        A_new, G_new = self._compute_kfac_factors(train_loader)

        for name in self._linear_layers:
            if A_new[name] is None:
                continue
            if name not in self._kfac_A:
                self._kfac_A[name] = A_new[name]
                self._kfac_G[name] = G_new[name]
            else:
                self._kfac_A[name] = self._kfac_A[name] + A_new[name]
                self._kfac_G[name] = self._kfac_G[name] + G_new[name]

        # Per-layer μ snapshot — used by `_apply_natural_gradient`.
        for name, module in self._linear_layers.items():
            self._prior_W[name] = module.weight.data.detach().cpu().clone()
            if module.bias is not None:
                self._prior_b[name] = module.bias.data.detach().cpu().clone()

        # Flat μ snapshot covering *all* trainable parameters. Detaching
        # and cloning ensures future parameter updates do not mutate it.
        self._prior_mean = _flat_params(self.model)

        # Posterior now exists — activate Eq. (8) update on subsequent tasks.
        self._has_prior = True

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def get_step_diagnostics(self) -> Dict[str, float]:
        """Scalars logged after the last observe() call.

        - ``kl_proxy``: (1/2)(θ − μ)ᵀ Λ (θ − μ). Zero before any prior
          has been accumulated; otherwise non-negative by PSD-ness of Λ.
        - ``tr_scale``: min(1, r / ‖step‖_Λ) where step is the Eq. 8
          direction and r = `trust_radius`. Per Algorithm 1 the radius is
          implicit in the learning rate and *no* clip is applied — this
          value is reported as a diagnostic only.
        """
        return {
            "kl_proxy": float(self._last_kl_proxy),
            "tr_scale": float(self._last_tr_scale),
        }
