"""Fisher / generalized Gauss-Newton (GGN) vector products.

This module provides the curvature oracle used by :class:`PrecondER`
(``src/methods/precond_er.py``).  PrecondER preconditions the experience-replay
gradient by the **Fisher information matrix** rather than the raw loss Hessian
(which is indefinite for a softmax-CE network and would need careful sign
handling).

For a model with logits ``z = f(θ; x)`` and a softmax cross-entropy loss, the
Fisher equals the generalized Gauss-Newton matrix

    F = (1/B) Σ_b  J_bᵀ H_b J_b ,
        J_b = ∂z_b/∂θ                              (C × P Jacobian)
        H_b = diag(p_b) − p_b p_bᵀ                 (C × C output Hessian)
        p_b = softmax(z_b)

which is **symmetric positive semi-definite by construction** — there are no
negative eigenvalues to clip or absolute-value, which is exactly why the
preconditioner uses it rather than the raw Hessian.

The matrix is never formed explicitly.  ``make_fisher_vp`` returns a flat-in /
flat-out oracle ``v → F v`` (consumed by the :func:`conjugate_gradient` solver
below), computed via the Schraudolph (2002) GGN trick:

    1.  z   = f(θ; x)                              (one forward pass, reused)
    2.  Jv  = J v                                  (double-grad through a dummy)
    3.  HJv = (1/B)(p ⊙ Jv − p (p·Jv))            (per-sample output Hessian)
    4.  Fv  = Jᵀ (HJv)                             (vector-Jacobian product)

The forward pass (step 1) is computed once and its graph retained, so every
CG iteration reuses it — only steps 2-4 repeat per matrix-vector product.

References
----------
Schraudolph, N. N. (2002). Fast Curvature Matrix-Vector Products for
  Second-Order Gradient Descent. Neural Computation, 14(7), 1723-1738.
Martens, J. (2014). New Insights and Perspectives on the Natural Gradient
  Method. https://arxiv.org/abs/1412.1193  (Fisher = GGN for softmax-CE.)
"""

from __future__ import annotations

from typing import Callable, List, Optional, Tuple

import torch
import torch.nn as nn


def make_fisher_vp(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    params: List[torch.Tensor],
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Build a flat-in / flat-out Fisher (GGN) matrix-vector oracle.

    The returned oracle multiplies an arbitrary vector by the Fisher of the
    **mean** softmax cross-entropy over the supplied mini-batch.  The label
    tensor ``y`` is accepted for API symmetry with the loss but is *not* used:
    the GGN/Fisher depends only on the model's predictive distribution
    ``softmax(z)``, not on the targets.

    Parameters
    ----------
    model : nn.Module
        Network mapping ``x`` to logits of shape ``(B, C)``.  Whatever mode
        (train / eval) the model is in at call time is used for the forward
        pass; the caller is responsible for setting it.
    x : torch.Tensor
        Input mini-batch already on the model's device.
    y : torch.Tensor
        Labels (unused — see above).  Present so callers can pass the same
        ``(x, y)`` they use for the loss.
    params : list of torch.Tensor
        Trainable leaf parameters defining the space in which F acts.  The
        oracle's input/output flat vectors are laid out in this order.

    Returns
    -------
    Callable[[Tensor], Tensor]
        ``fisher_vp(v_flat) → Fv_flat`` where both vectors have length
        ``Σ p.numel() for p in params``.  Suitable as the ``matvec`` argument
        to :func:`conjugate_gradient`.
    """
    del y  # Fisher depends on the model distribution only, not the targets.

    shapes = [p.shape for p in params]
    numels = [p.numel() for p in params]

    # ── Step 1: single forward pass, graph retained across all oracle calls ──
    logits = model(x)                                  # (B, C)
    batch_size = logits.shape[0]
    # softmax probabilities define H_b; detached because the GGN treats the
    # output-Hessian as a constant reweighting (it is not differentiated).
    probs = torch.softmax(logits, dim=1).detach()      # (B, C)

    def _unflat(v_flat: torch.Tensor) -> List[torch.Tensor]:
        parts: List[torch.Tensor] = []
        offset = 0
        for sh, n in zip(shapes, numels):
            parts.append(v_flat[offset : offset + n].view(sh))
            offset += n
        return parts

    def fisher_vp(v_flat: torch.Tensor) -> torch.Tensor:
        v_list = _unflat(v_flat)

        # ── Step 2: Jv via the double-backward trick through a dummy w ──────
        # grad(z, params, grad_outputs=w) = Jᵀ w; differentiating its dot with
        # v back through w yields Jv = d/dw (wᵀ J v).
        w = torch.zeros_like(logits, requires_grad=True)
        jt_w = torch.autograd.grad(
            logits, params, grad_outputs=w, create_graph=True
        )
        jv = torch.autograd.grad(
            sum((g * v).sum() for g, v in zip(jt_w, v_list)),
            w,
            retain_graph=True,
        )[0]                                            # (B, C)

        # ── Step 3: apply the per-sample output Hessian, mean reduction ─────
        # H_b u = p_b ⊙ u − p_b (p_b · u);  the 1/B matches CE mean reduction.
        pj = (probs * jv).sum(dim=1, keepdim=True)      # (B, 1)
        h_jv = (probs * jv - probs * pj) / batch_size   # (B, C)

        # ── Step 4: Fv = Jᵀ (H Jv) ─────────────────────────────────────────
        fv = torch.autograd.grad(
            logits, params, grad_outputs=h_jv, retain_graph=True
        )
        return torch.cat([f.reshape(-1) for f in fv])

    return fisher_vp


def conjugate_gradient(
    matvec: Callable[[torch.Tensor], torch.Tensor],
    b: torch.Tensor,
    damping: float,
    iters: int,
    tol: float = 1e-4,
    x0: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, int]:
    """Solve ``(A + δI) x = b`` for ``x`` with Conjugate Gradient.

    Used to compute the damped natural gradient ``x ≈ (F + δI)⁻¹ g`` from the
    Fisher (= GGN) matrix-vector oracle alone — no matrix is ever formed.  The
    operator ``A + δI`` is symmetric positive-definite (``A = F`` is PSD and
    ``δ > 0``), which is exactly the setting CG requires.

    Each iteration costs **one** call to ``matvec`` (the FVP) plus O(P) vector
    arithmetic.  CG resolves the high-curvature part of the spectrum first, so
    a small ``iters`` (≈5-20, fewer with warm-starting) already gives a good
    direction.

    Parameters
    ----------
    matvec : Callable[[Tensor], Tensor]
        The Fisher-vector oracle ``v → F v`` (e.g. from :func:`make_fisher_vp`).
        The ``δ v`` damping term is added internally, so ``matvec`` should
        return the *undamped* ``F v``.
    b : torch.Tensor
        Right-hand side (the gradient ``g``), shape ``(P,)``.
    damping : float
        ``δ`` added to the diagonal of ``F``.  Must be > 0.
    iters : int
        Maximum number of CG iterations.
    tol : float
        Relative residual tolerance; CG stops early once
        ``‖r‖ ≤ tol · ‖b‖``.  Default ``1e-4``.
    x0 : torch.Tensor, optional
        Warm-start solution (e.g. the previous step's natural gradient).  When
        ``None`` (default) CG starts from zero.

    Returns
    -------
    x : torch.Tensor
        Approximate solution ``(F + δI)⁻¹ b``, shape ``(P,)``.
    n_iter : int
        Number of CG iterations actually performed (≤ ``iters``).
    """
    def _apply(v: torch.Tensor) -> torch.Tensor:
        return matvec(v) + damping * v

    b_norm = torch.linalg.norm(b)
    if b_norm < 1e-12:
        return torch.zeros_like(b), 0

    if x0 is None:
        x = torch.zeros_like(b)
        r = b.clone()
    else:
        x = x0.clone()
        r = b - _apply(x)

    p = r.clone()
    rs = torch.dot(r, r)
    thresh = (tol * b_norm) ** 2  # compare squared norms to avoid a sqrt/iter

    n_iter = 0
    for n_iter in range(1, iters + 1):
        Ap = _apply(p)
        pAp = torch.dot(p, Ap)
        if pAp <= 1e-12:
            # Non-positive curvature (numerical) — A+δI is PD, so this only
            # happens from round-off; stop with the current iterate.
            break
        alpha = rs / pAp
        x = x + alpha * p
        r = r - alpha * Ap
        rs_new = torch.dot(r, r)
        if rs_new <= thresh:
            break
        p = r + (rs_new / rs) * p
        rs = rs_new

    return x, n_iter
