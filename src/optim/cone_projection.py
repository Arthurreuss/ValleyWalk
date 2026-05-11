"""Dangerous-subspace deflation and cone projection for CACL.

This module implements the two geometric operations that form the heart of
CACL's direction-finding step:

1. **Deflation** — removes the projection of the joint gradient onto the
   *dangerous subspace* V_danger (the span of the top-d replay-Hessian
   eigenvectors).  The result, ``g_deflated``, points in a direction that
   reduces the joint loss while avoiding the axes of maximum replay curvature.

2. **Cone projection** — ensures the final step direction ``d_star`` does not
   deviate more than ``alpha_deg`` from the joint gradient direction
   ``g_hat_joint``.  If ``g_deflated`` already lies within the cone, it is
   returned normalised.  Otherwise a binary search finds the cone-boundary
   direction (Equation 14 of the paper).

Usage (inside CACL's ``observe()`` method)::

    from src.optim.cone_projection import deflate, cone_project

    # g_hat_joint: normalised joint gradient (1-D, shape (P,))
    # V_danger:    top-d Lanczos eigenvectors  (shape (P, d))

    g_deflated        = deflate(g_hat_joint, V_danger)
    d_star, fallback  = cone_project(g_hat_joint, g_deflated, alpha_deg=45.0)
    params_update     = trust_radius * d_star

References
----------
Equation numbering follows the CACL paper draft.  Eq. 14 defines the
interpolated cone-boundary direction used in ``cone_project``.
"""

from __future__ import annotations

import math
from typing import Tuple

import torch


def deflate(
    g_hat_joint: torch.Tensor,
    V_danger: torch.Tensor,
) -> torch.Tensor:
    """Remove the projection of *g_hat_joint* onto the dangerous subspace.

    Computes::

        g_deflated = g_hat_joint − V_danger (V_dangerᵀ g_hat_joint)
                   = g_hat_joint − Σᵢ (vᵢᵀ g_hat_joint) vᵢ

    where the columns of *V_danger* are the top-d eigenvectors of the replay
    Hessian (the "dangerous directions").  The result is orthogonal to every
    column of *V_danger*.

    Args:
        g_hat_joint:
            Normalised (unit) joint-loss gradient, 1-D tensor of shape ``(P,)``.
        V_danger:
            Dangerous-subspace basis matrix of shape ``(P, d)``.  Columns
            must be mutually orthonormal (as produced by :func:`lanczos`).

    Returns:
        g_deflated:
            1-D tensor of shape ``(P,)`` with dangerous components removed.
            The returned vector is **not** normalised; its norm satisfies
            ``‖g_deflated‖ ≤ 1`` and equals 0 only when *g_hat_joint* lies
            entirely in the dangerous subspace.

    Notes:
        - If *V_danger* has shape ``(P, 0)`` (no dangerous directions), the
          function returns *g_hat_joint* unchanged.
        - The projection exploits the orthonormality of V_danger's columns;
          no matrix inversion is required.
    """
    if V_danger.shape[1] == 0:
        return g_hat_joint.clone()

    # Scalar projections onto each dangerous direction: shape (d,)
    projections = V_danger.T @ g_hat_joint        # (d,)
    # Subtract the component in the dangerous subspace
    g_deflated = g_hat_joint - V_danger @ projections   # (P,)
    return g_deflated


def cone_project(
    g_hat_joint: torch.Tensor,
    g_deflated: torch.Tensor,
    alpha_deg: float,
    _binary_search_iters: int = 64,
) -> Tuple[torch.Tensor, bool]:
    """Project the deflated gradient onto the cone centred on *g_hat_joint*.

    The cone is defined by all unit vectors ``d`` satisfying::

        cos(∠(d, g_hat_joint)) ≥ cos(alpha_deg)
        ⟺  d · g_hat_joint ≥ cos(alpha_deg)

    **Procedure:**

    1. Handle degenerate input (``‖g_deflated‖ < 1e-8``): the joint gradient
       lies entirely in the dangerous subspace.  Return ``g_hat_joint``
       directly and signal a full-conflict fallback.

    2. Normalise *g_deflated* → ``ĝ_deflated``.

    3. **Cone check**: if ``ĝ_deflated · g_hat_joint ≥ cos(alpha_deg)``,
       the deflated direction is already within the cone.  Return
       ``(ĝ_deflated, False)`` — no fallback.

    4. **Binary search** (Equation 14): find β ∈ [0, 1] such that the
       interpolated direction::

           d*(β) = normalise(β · g_hat_joint + (1−β) · ĝ_deflated)

       satisfies ``d*(β) · g_hat_joint = cos(alpha_deg)`` (on the cone
       boundary).  Return ``(d*(β_star), True)``.

       The search is valid because ``f(β) = d*(β) · g_hat_joint`` is
       monotone increasing from ``f(0) = ĝ_deflated · g_hat_joint``
       (< cos α, outside cone) to ``f(1) = 1`` (inside cone).

    Args:
        g_hat_joint:
            Normalised joint-loss gradient, 1-D tensor of shape ``(P,)``.
        g_deflated:
            Output of :func:`deflate`; unnormalised, shape ``(P,)``.
        alpha_deg:
            Cone half-angle in degrees.  Valid range: [0, 90].
            ``alpha_deg=0`` degenerates to returning ``g_hat_joint``.
            ``alpha_deg=90`` always returns ``ĝ_deflated`` (the full
            hemisphere is within the cone).
        _binary_search_iters:
            Number of bisection iterations.  Default 64 gives ~1e-19 precision
            in β.  (Internal parameter; not part of the public API.)

    Returns:
        d_star:
            Unit-norm step direction, 1-D tensor of shape ``(P,)``.
            Satisfies the cone constraint within numerical tolerance.
        fallback_triggered:
            ``True`` if the deflated gradient was outside the cone (or the
            full-conflict degenerate case occurred); ``False`` if
            ``ĝ_deflated`` was already within the cone.

    Notes:
        - The binary search is stable for all alpha in (0°, 180°).
        - When ``‖g_deflated‖ < 1e-8`` (full conflict), the return is
          ``(g_hat_joint, True)``.  Log this event in production.
        - The trust-ratio denominator in :class:`TrustRegion` uses
          ``g_hat_joint · d_star``; this can be near zero when α ≈ 90°.
          Guard against that in the calling code.
    """
    # ── Special case: alpha=0 → cone degenerates to a single ray ───────────
    # Return g_hat_joint directly; binary search would converge to β=1 anyway,
    # but float32 precision causes it to stop at β≈1-ε, yielding a tiny
    # residual error.
    if alpha_deg == 0.0:
        return g_hat_joint.clone(), True

    # ── Degenerate: g_deflated ≈ 0 (joint gradient in dangerous subspace) ──
    # In float32, QR decomposition leaves residuals ~1e-7 even when g_hat lies
    # exactly in the dangerous subspace.  Using 1e-7 as the threshold reliably
    # catches this case without false positives on valid deflated gradients.
    norm_g_deflated = torch.linalg.norm(g_deflated)
    if norm_g_deflated < 1e-7:
        # Full-conflict: skip deflation, return joint direction
        return g_hat_joint.clone(), True

    g_deflated_norm: torch.Tensor = g_deflated / norm_g_deflated   # unit vector

    cos_alpha: float = math.cos(math.radians(alpha_deg))

    # ── Cone check ────────────────────────────────────────────────────────────
    cos_angle: float = torch.dot(g_deflated_norm, g_hat_joint).item()
    if cos_angle >= cos_alpha:
        # Deflated direction is inside (or on) the cone boundary.
        return g_deflated_norm, False

    # ── Binary search for β (Equation 14) ────────────────────────────────────
    # We want f(β) = cos_angle_of(β) = cos_alpha.
    # f(0) = cos_angle  < cos_alpha   (outside cone)
    # f(1) = 1.0        ≥ cos_alpha   (g_hat_joint at cone centre)
    # f is monotone increasing in β.

    def _cos_of_angle(beta: float) -> float:
        """Cosine of angle between d*(β) and g_hat_joint."""
        d = beta * g_hat_joint + (1.0 - beta) * g_deflated_norm
        norm_d = torch.linalg.norm(d)
        if norm_d < 1e-12:
            return 0.0
        return (torch.dot(d, g_hat_joint) / norm_d).item()

    beta_lo: float = 0.0   # f(beta_lo) < cos_alpha  (outside)
    beta_hi: float = 1.0   # f(beta_hi) ≥ cos_alpha  (inside or on boundary)

    for _ in range(_binary_search_iters):
        beta_mid = 0.5 * (beta_lo + beta_hi)
        if _cos_of_angle(beta_mid) < cos_alpha:
            beta_lo = beta_mid
        else:
            beta_hi = beta_mid

    # Use the hi side (guaranteed to be inside or on boundary)
    beta_star: float = beta_hi
    d_raw = beta_star * g_hat_joint + (1.0 - beta_star) * g_deflated_norm
    d_star: torch.Tensor = d_raw / torch.linalg.norm(d_raw)

    return d_star, True
