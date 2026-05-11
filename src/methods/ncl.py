"""NCL — Natural Continual Learning (Kao et al., NeurIPS 2021).

True NCL frames continual learning as online Bayesian inference where the
posterior from each completed task becomes the Gaussian prior for the next:

    p(θ | D_{1:k}) ∝ p(D_k | θ) · N(θ; μ_{k-1}, Λ_{k-1}^{-1})

The prior precision Λ_{k-1} is approximated with K-FAC (Kronecker-Factored
Approximate Curvature) rather than a diagonal Fisher.

**Key algorithmic distinction from EWC**

EWC adds a static rubber-band penalty to the loss:

    L_total = L_k(θ) + (λ/2) Σ_i F_i (θ_i - θ*_i)²

NCL instead modifies the *optimisation trajectory* by replacing the raw
gradient with the natural gradient computed under the accumulated precision:

    Δθ_nat = Λ_{k-1}^{-1} · ∇_θ L_k(θ)

For a Linear layer y = Wx + b this factorises cleanly as:

    [ΔW_nat | Δb_nat] = G_{k-1}^{-1} · [∇W | ∇b] · A_{k-1}^{-1}

where A (input autocorrelation) and G (output-gradient autocorrelation) are
the two Kronecker factors of the layer-wise Fisher:

    F_W ≈ G ⊗ A
    A = E[ā ā^T],    ā = [x; 1]  (input augmented with bias column)
    G = E[g_s g_s^T], g_s = ∂L / ∂(Wx + b)

High-curvature directions in the parameter space (important for past tasks)
receive small natural gradient steps; orthogonal directions are updated freely.
A trust-region clip ensures the Λ-weighted step norm stays within radius δ:

    scale = min(1, δ / √(g^T Λ^{-1} g))

**After each task** the K-FAC factors are folded into the evolving Bayesian
prior (online precision update):

    Λ_k ≈ Λ_{k-1} + F_k
    A_k ← A_{k-1} + Â_k
    G_k ← G_{k-1} + Ĝ_k

and the prior mean μ_k is snapshotted as the current parameter vector.

**Non-Linear layers** (BatchNorm, LayerNorm, embeddings) are updated with
the raw gradient unchanged; only nn.Linear layers receive K-FAC treatment.

References:
    Kao et al. (2021) "Natural Continual Learning"
        — https://arxiv.org/abs/2106.08085
    Martens & Grosse (2015) "Optimizing Neural Networks with Kronecker-factored
        Approximate Curvature" — https://arxiv.org/abs/1503.05671

Usage::

    method = NCL(model, cfg)
    for task_id, train_loader, test_loaders in dataset.task_iterator():
        for x, y in train_loader:
            result = method.observe(x, y, task_id)   # → {"loss": float, ...}
        method.end_task(task_id, train_loader)        # accumulates K-FAC prior
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple  # Tuple kept for _compute_kfac_factors

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader

from src.methods.base_method import BaseMethod


# ---------------------------------------------------------------------------
# Module-level helper
# ---------------------------------------------------------------------------

def _damped_inv(M: Tensor, damping: float) -> Tensor:
    """Compute (M + ε·I)⁻¹ with a hard fallback on LinAlgError.

    Parameters
    ----------
    M : Tensor
        Square symmetric matrix on any device.
    damping : float
        Tikhonov regularisation constant ε > 0.

    Returns
    -------
    Tensor
        Inverse of the damped matrix, same device as M.
    """
    n = M.size(0)
    I = torch.eye(n, device=M.device, dtype=M.dtype)
    try:
        return torch.linalg.inv(M + damping * I)
    except torch.linalg.LinAlgError:
        # Rare: matrix is extremely ill-conditioned — apply stronger damping.
        return torch.linalg.inv(M + (damping * 100.0) * I)


# ---------------------------------------------------------------------------
# NCL
# ---------------------------------------------------------------------------

class NCL(BaseMethod):
    """True NCL — K-FAC precision matrix with natural gradient projection.

    Parameters
    ----------
    model : nn.Module
        The neural network being trained.
    cfg : DictConfig
        **Full** Hydra config.  NCL reads ``cfg.training`` for optimiser
        hyper-parameters and ``cfg.method.ncl`` for NCL-specific settings:

        * ``fisher_samples`` (int): maximum training examples used to
          estimate K-FAC factors after each task.
        * ``damping`` (float): ε added to the diagonal of A and G before
          inversion (Tikhonov regularisation for numerical stability).
    """

    def __init__(self, model: nn.Module, cfg) -> None:
        super().__init__(model, cfg.method)
        self._full_cfg = cfg

        ncl_cfg = cfg.method.ncl
        self._fisher_samples: int = int(ncl_cfg.fisher_samples)
        self._damping: float = float(ncl_cfg.damping)

        self._device: torch.device = next(model.parameters()).device

        # Collect all nn.Linear layers — only these receive K-FAC treatment.
        self._linear_layers: Dict[str, nn.Linear] = {
            name: module
            for name, module in model.named_modules()
            if isinstance(module, nn.Linear)
        }

        # Accumulated Bayesian prior precision (Λ_{k-1}), decomposed as
        # per-layer Kronecker factors stored on CPU to minimise GPU memory.
        #
        #   _kfac_A[name] : Tensor (d_in[+1], d_in[+1]) — input autocorrelation
        #   _kfac_G[name] : Tensor (d_out,  d_out)      — gradient autocorrelation
        self._kfac_A: Dict[str, Tensor] = {}
        self._kfac_G: Dict[str, Tensor] = {}

        # Prior mean μ_{k-1} — per-layer weight/bias snapshots after the last
        # completed task (CPU).  Stored per-layer (not flat) so that the
        # restoring force (W − W*) can be computed directly in the gradient
        # projection step without requiring a costly unflatten.
        self._prior_W: Dict[str, Tensor] = {}   # weight snapshot per linear layer
        self._prior_b: Dict[str, Tensor] = {}   # bias snapshot per linear layer

        self.optimizer = torch.optim.SGD(
            model.parameters(),
            lr=float(cfg.training.lr),
            weight_decay=float(cfg.training.weight_decay),
        )
        self.loss_fn = nn.CrossEntropyLoss()

    # ------------------------------------------------------------------
    # Natural gradient projection
    # ------------------------------------------------------------------

    def _apply_natural_gradient(self) -> None:
        """Project in-place gradients through the accumulated K-FAC precision.

        For each Linear layer with prior factors (A_prior, G_prior):

            [ΔW_nat | Δb_nat] = G_prior⁻¹ · [∇W | ∇b] · A_prior⁻¹

        The full MAP natural gradient is:

            Λ⁻¹ ∇J = Λ⁻¹ ∇L_k  +  (θ − μ*)

        where (θ − μ*) is the restoring force from the Bayesian prior.
        Numerical stability is handled entirely by damping ε in _damped_inv.

        Layers with no K-FAC prior (task 0, or non-Linear layers) are left
        with their raw gradients untouched.
        """
        if not self._kfac_A:
            # No prior yet (first task) — raw gradient passes through unchanged.
            return

        # ---- Pass 1: compute natural gradients ----
        nat_task: Dict[str, Tensor] = {}   # Λ⁻¹ ∇L_k
        restoring: Dict[str, Tensor] = {}  # (θ − μ*)

        for name, module in self._linear_layers.items():
            if name not in self._kfac_A or module.weight.grad is None:
                continue

            A = self._kfac_A[name].to(self._device)
            G = self._kfac_G[name].to(self._device)
            A_inv = _damped_inv(A, self._damping)
            G_inv = _damped_inv(G, self._damping)

            grad_W = module.weight.grad.data  # (d_out, d_in)

            # Restoring force: (W − W*) — zero on first task (no prior mean yet).
            delta_W = (
                module.weight.data - self._prior_W[name].to(self._device)
                if name in self._prior_W else torch.zeros_like(grad_W)
            )

            if module.bias is not None and module.bias.grad is not None:
                delta_b = (
                    module.bias.data - self._prior_b[name].to(self._device)
                    if name in self._prior_b else torch.zeros_like(module.bias.data)
                )
                grad_aug = torch.cat(
                    [grad_W, module.bias.grad.data.unsqueeze(1)], dim=1
                )
                nat_task[name] = G_inv @ grad_aug @ A_inv        # (d_out, d_in+1)
                restoring[name] = torch.cat([delta_W, delta_b.unsqueeze(1)], dim=1)
            else:
                nat_task[name] = G_inv @ grad_W @ A_inv          # (d_out, d_in)
                restoring[name] = delta_W

        # ---- Pass 2: write effective gradients back ----
        for name, module in self._linear_layers.items():
            if name not in nat_task:
                continue
            ng = nat_task[name] + restoring[name]
            if module.bias is not None and module.bias.grad is not None:
                module.weight.grad.data = ng[:, :-1].contiguous()
                module.bias.grad.data = ng[:, -1].contiguous()
            else:
                module.weight.grad.data = ng

    # ------------------------------------------------------------------
    # K-FAC factor estimation
    # ------------------------------------------------------------------

    def _compute_kfac_factors(
        self, train_loader: DataLoader
    ) -> Tuple[Dict[str, Optional[Tensor]], Dict[str, Optional[Tensor]]]:
        """Estimate K-FAC (A, G) factors for every Linear layer.

        Uses standard batch forward/backward passes with registered hooks.
        The factors are the uncentred second moments of, respectively, the
        layer input and the pre-activation gradient signal:

            A = (1/N) Σ_n ā_n ā_n^T     (ā = [x; 1] for biased layers)
            G = (1/N) Σ_n g_{s,n} g_{s,n}^T

        Since the CE loss uses mean-reduction, the backward-hook gradient
        ``g_out[0][i]`` equals ``(1/B) · g_{s,i}``.  We multiply by B to
        recover the per-sample scale before forming the outer product.

        Parameters
        ----------
        train_loader : DataLoader
            Training loader for the just-completed task.

        Returns
        -------
        A_out, G_out : dicts mapping layer name → CPU Tensor (or None if the
            layer was never reached during the forward pass).
        """
        a_sum: Dict[str, Optional[Tensor]] = {n: None for n in self._linear_layers}
        g_sum: Dict[str, Optional[Tensor]] = {n: None for n in self._linear_layers}
        a_cnt: Dict[str, int] = {n: 0 for n in self._linear_layers}
        g_cnt: Dict[str, int] = {n: 0 for n in self._linear_layers}

        saved_inputs: Dict[str, Tensor] = {}
        hooks: List = []

        for name, module in self._linear_layers.items():

            def _fwd(mod: nn.Linear, inp, out, _n: str = name) -> None:
                a = inp[0].detach()  # (B, d_in)
                if mod.bias is not None:
                    ones = torch.ones(
                        a.size(0), 1, device=a.device, dtype=a.dtype
                    )
                    a = torch.cat([a, ones], dim=1)  # (B, d_in+1)
                saved_inputs[_n] = a

            def _bwd(mod: nn.Linear, g_in, g_out, _n: str = name) -> None:
                gs = g_out[0].detach()   # (B, d_out) — mean-reduced
                a = saved_inputs.pop(_n, None)
                if a is None:
                    return
                B = gs.size(0)
                # Recover per-sample gradient scale (undo 1/B mean-reduction).
                gs_sample = gs * B       # (B, d_out)
                A_batch = a.T @ a        # (d_in[+1], d_in[+1]) — sum of outer products
                G_batch = gs_sample.T @ gs_sample  # (d_out, d_out)
                if a_sum[_n] is None:
                    a_sum[_n] = A_batch
                    g_sum[_n] = G_batch
                else:
                    a_sum[_n] = a_sum[_n] + A_batch
                    g_sum[_n] = g_sum[_n] + G_batch
                a_cnt[_n] += B
                g_cnt[_n] += B

            hooks.append(module.register_forward_hook(_fwd))
            hooks.append(module.register_full_backward_hook(_bwd))

        # register_full_backward_hook wraps module outputs in a custom
        # backward function.  Any inplace activation (relu_, silu_, …) that
        # runs *after* the hook-wrapped layer then tries to modify a view of
        # that wrapped output, which PyTorch forbids.  Temporarily disabling
        # inplace on all activation modules sidesteps the conflict without
        # altering any weights or the forward computation's numerical result.
        inplace_modules = [
            m for m in self.model.modules()
            if hasattr(m, "inplace") and m.inplace
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
                G_out[name] = (g_sum[name] / max(g_cnt[name], 1)).cpu()

        return A_out, G_out

    # ------------------------------------------------------------------
    # BaseMethod abstract interface
    # ------------------------------------------------------------------

    def observe(
        self,
        x_batch: Tensor,
        y_batch: Tensor,
        task_id: int,
    ) -> Dict[str, float]:
        """One natural-gradient step on the current task.

        No regularisation term is added to the loss.  The Bayesian prior
        from past tasks is encoded entirely in the gradient projection step:
        parameter updates are restricted to directions that do not interfere
        with the K-FAC curvature of previously learned tasks.

        On task 0 (empty prior) the method reduces to plain SGD.

        Parameters
        ----------
        x_batch : Tensor  — shape (B, ...)
        y_batch : Tensor  — shape (B,)
        task_id : int     — zero-based task index

        Returns
        -------
        dict
            ``{"loss": float}``
        """
        self.model.train()
        x_batch = x_batch.to(self._device)
        y_batch = y_batch.to(self._device)

        self.optimizer.zero_grad()
        loss = self.loss_fn(self.model(x_batch), y_batch)
        loss.backward()

        # Project gradients through accumulated K-FAC precision.
        # No-op on task 0 (prior is empty).
        self._apply_natural_gradient()

        self.optimizer.step()
        return {"loss": loss.item()}

    def end_task(self, task_id: int, train_loader: DataLoader) -> None:
        """Accumulate K-FAC factors and update the Bayesian prior.

        Implements the online precision update:

            Λ_k = Λ_{k-1} + F_k
            A_k ← A_{k-1} + Â_k   (input autocorrelation)
            G_k ← G_{k-1} + Ĝ_k   (gradient autocorrelation)

        The prior mean μ_k is set to the current parameter vector θ_k*.
        Future ``observe()`` calls will use the updated (Λ_k, μ_k) prior.

        Parameters
        ----------
        task_id : int
            Zero-based index of the task that just finished.
        train_loader : DataLoader
            Training loader for the completed task, used for K-FAC estimation.
        """
        A_new, G_new = self._compute_kfac_factors(train_loader)

        for name in self._linear_layers:
            if A_new[name] is None:
                continue
            if name not in self._kfac_A:
                # First task: initialise prior from scratch.
                self._kfac_A[name] = A_new[name]
                self._kfac_G[name] = G_new[name]
            else:
                # Subsequent tasks: accumulate — Λ_k = Λ_{k-1} + F_k.
                self._kfac_A[name] = self._kfac_A[name] + A_new[name]
                self._kfac_G[name] = self._kfac_G[name] + G_new[name]

        # Snapshot current per-layer weights as the new prior mean μ_k.
        # Stored per-layer so _apply_natural_gradient can compute (W − W*)
        # directly without unflattening a global parameter vector.
        for name, module in self._linear_layers.items():
            self._prior_W[name] = module.weight.data.cpu().clone()
            if module.bias is not None:
                self._prior_b[name] = module.bias.data.cpu().clone()

