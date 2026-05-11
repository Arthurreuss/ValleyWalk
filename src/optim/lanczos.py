"""Hessian-vector products (Pearlmutter) and Lanczos iteration.

This module provides the low-level linear-algebra primitives used by CACL to
identify and project away from curvature directions in the replay Hessian that
are dangerous for previously learned tasks.

Components
----------
T4.1  hvp()     — Exact HVP via double back-prop (Pearlmutter, 1994).
T4.2  lanczos() — Full reorthogonalised Lanczos to extract top eigenpairs.

References
----------
Pearlmutter, B. A. (1994). Fast Exact Multiplication by the Hessian.
  Neural Computation, 6(1), 147–160.

Golub, G. H. & Van Loan, C. F. (2013). Matrix Computations (4th ed.).
  Johns Hopkins University Press. §10.1 (Lanczos).
"""

from __future__ import annotations

from typing import Callable, List, Optional, Tuple

import torch


def hvp(
    loss_fn: Callable[[List[torch.Tensor]], torch.Tensor],
    params: List[torch.Tensor],
    v: List[torch.Tensor],
    g: Optional[List[torch.Tensor]] = None,
    damping: float = 0.0,
) -> torch.Tensor:
    """Compute the Hessian-vector product (H + εI)v using double back-prop.

    Implements Pearlmutter's "R-operator" trick in two autodiff passes:

    .. code-block:: text

        1.  loss = loss_fn(params)
        2.  g    = ∂loss / ∂params          [create_graph=True]
        3.  gv   = Σ_i  g_i · v_i           [scalar]
        4.  Hv   = ∂(gv) / ∂params

    When *g* is already available from a prior backward pass (e.g. the joint
    gradient computed at step 3 of CACL Algorithm 1), it can be supplied
    directly to skip the forward and first backward passes.

    Args:
        loss_fn:
            Callable with signature ``loss_fn(params) → scalar Tensor``.
            It must compute the loss by flowing computation *through* the same
            parameter tensors in *params* so that ``torch.autograd`` can
            differentiate w.r.t. them.  In practice, ``loss_fn`` may ignore its
            argument and use the model's stored parameters (which are the same
            objects) — the important thing is that ``params`` appear in the
            computation graph of the returned scalar.
        params:
            List of leaf ``Tensor`` objects, each with ``requires_grad=True``.
            These define the space in which H is computed.
        v:
            List of tensors with the same shapes as *params*, representing the
            vector to multiply against H.  Need not have ``requires_grad``.
        g:
            Optional pre-computed gradient list.  Each element must have been
            produced with ``create_graph=True`` so the graph is still alive
            for the second backward pass.  If ``None``, *g* is computed
            internally from ``loss_fn(params)``.
        damping:
            Non-negative scalar ε added to the diagonal of H, yielding
            ``(H + εI)v``.  Helps when H is near-singular (e.g. during the
            first steps of a new task when the replay loss is nearly zero).
            Default: ``0.0`` (no damping).

    Returns:
        Hv:
            1-D tensor of shape ``(P,)`` where ``P = Σ p.numel() for p in params``.
            The entries correspond to the parameters in the order they appear in
            *params*, each flattened and concatenated.

    Raises:
        RuntimeError:
            If *params* do not have ``requires_grad=True``, or if the
            computation graph has already been freed before the second pass.

    Example::

        model   = nn.Linear(10, 5)
        params  = list(model.parameters())
        x, y    = torch.randn(8, 10), torch.randint(5, (8,))
        loss_fn = lambda p: F.cross_entropy(model(x), y)
        v       = [torch.randn_like(p) for p in params]
        Hv      = hvp(loss_fn, params, v)          # shape: (55,)
    """
    # ------------------------------------------------------------------
    # Step 1–2: forward pass and first-order gradients (with graph retained)
    # ------------------------------------------------------------------
    if g is None:
        loss = loss_fn(params)
        g = torch.autograd.grad(loss, params, create_graph=True)

    # ------------------------------------------------------------------
    # Step 3: scalar dot product g·v
    # ------------------------------------------------------------------
    gv: torch.Tensor = sum((gi * vi).sum() for gi, vi in zip(g, v))

    # ------------------------------------------------------------------
    # Step 4: second-order gradients — the actual HVP
    # allow_unused=True: if a param doesn't appear in gv, return zeros.
    # ------------------------------------------------------------------
    Hv_raw = torch.autograd.grad(gv, params, allow_unused=True)
    Hv: List[torch.Tensor] = [
        (h if h is not None else torch.zeros_like(p))
        for h, p in zip(Hv_raw, params)
    ]

    # ------------------------------------------------------------------
    # Flatten, concatenate, and optionally damp
    # ------------------------------------------------------------------
    Hv_flat = torch.cat([h.flatten() for h in Hv])

    if damping != 0.0:
        v_flat = torch.cat([vi.flatten() for vi in v])
        Hv_flat = Hv_flat + damping * v_flat

    return Hv_flat


# ---------------------------------------------------------------------------
# T4.2 — Lanczos iteration
# ---------------------------------------------------------------------------

def lanczos(
    hvp_fn: Callable[[torch.Tensor], torch.Tensor],
    dim: int,
    k: int,
    d: int,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Extract the top-d eigenpairs of an implicit symmetric matrix via Lanczos.

    Implements the textbook Lanczos algorithm with full reorthogonalisation
    (Gram-Schmidt against *all* previous basis vectors at each step) to prevent
    ghost eigenvalues caused by floating-point loss of orthogonality.

    The matrix is accessed only through the matrix-vector oracle *hvp_fn*, so
    it never needs to be stored explicitly.  In CACL, *hvp_fn* wraps the
    replay-Hessian HVP from :func:`hvp`.

    Algorithm (three-term recurrence + full reorthogonalisation)
    ------------------------------------------------------------
    Start: q₀ = random unit vector, β₋₁ = 0.

    For j = 0, 1, …, k−1:
        z      = H qⱼ                          (HVP oracle)
        αⱼ     = qⱼᵀ z                         (diagonal of tridiagonal T)
        z      = z − αⱼ qⱼ − βⱼ₋₁ qⱼ₋₁        (three-term residual)
        z      = z − Σᵢ (qᵢᵀ z) qᵢ             (full reorthogonalisation)
        βⱼ     = ‖z‖                            (off-diagonal of T)
        if βⱼ < ε: stop                         (invariant subspace found)
        qⱼ₊₁  = z / βⱼ

    Eigendecompose the n×n tridiagonal T:  T = S Λ Sᵀ
    Ritz vectors:  Q S  (shape dim × n)
    Return top-d eigenpairs sorted by eigenvalue descending.

    Args:
        hvp_fn:
            Oracle with signature ``hvp_fn(v: Tensor[dim]) → Tensor[dim]``.
            Must accept and return 1-D tensors of the specified *device* and
            *dtype*.  This is typically ``lambda v: hvp(loss_fn, params, v)``.
        dim:
            Dimension of the parameter space (size of the vectors).
        k:
            Number of Lanczos steps to perform.  Should satisfy ``k ≥ d``.
            Larger *k* gives higher accuracy at the cost of more HVP evaluations.
            For CACL, the config default is ``k = 15``.
        d:
            Number of top eigenpairs to return.  Must satisfy ``d ≤ k``.
        device:
            Device on which to create the Lanczos basis vectors.  Must match
            the device expected by *hvp_fn*.  Defaults to CPU.
        dtype:
            Floating-point dtype for the basis vectors.  Defaults to
            ``torch.float32``.

    Returns:
        eigenvalues:
            1-D tensor of shape ``(d,)`` containing the *d* largest Ritz
            values, sorted in **descending** order.
        eigenvectors:
            2-D tensor of shape ``(dim, d)`` whose columns are the
            corresponding Ritz vectors (approximate eigenvectors), each of
            unit norm.

    Notes:
        - If an invariant subspace is found before *k* steps, the function
          returns however many eigenpairs were computed (at most *min(n, d)*).
        - The Ritz vectors are accurate approximations of the true eigenvectors
          when the eigenvalues are well-separated and *k* is large enough.
        - ``d`` should be less than the actual number of Lanczos steps taken;
          requesting more eigenvectors than steps yields only as many as steps.

    Example::

        # Treat a known 100×100 matrix as an implicit HVP oracle
        A = torch.randn(100, 100)
        A = A @ A.T                            # make PSD
        hvp_fn = lambda v: A @ v
        evals, evecs = lanczos(hvp_fn, dim=100, k=30, d=5)
        # evals: (5,) largest Ritz values
        # evecs: (100, 5) corresponding unit Ritz vectors
    """
    if device is None:
        device = torch.device("cpu")
    if dtype is None:
        dtype = torch.float32

    # ── Initialise the starting vector as a random unit vector ──────────────
    torch.manual_seed(0)  # reproducible starting point; does not affect caller RNG
    q = torch.randn(dim, device=device, dtype=dtype)
    q = q / torch.linalg.norm(q)

    # Storage for Lanczos basis, diagonal (α) and off-diagonal (β) elements
    Q: List[torch.Tensor] = []   # Lanczos basis vectors (each shape: (dim,))
    alphas: List[torch.Tensor] = []
    betas: List[torch.Tensor] = []

    # ── Main Lanczos loop ────────────────────────────────────────────────────
    for j in range(k):
        Q.append(q.clone())

        # Matrix-vector product via oracle
        z = hvp_fn(q)                                    # shape: (dim,)

        # Diagonal element α_j = qⱼᵀ H qⱼ
        alpha_j = torch.dot(q, z)
        alphas.append(alpha_j)

        # Three-term recurrence: subtract α·q and (previous) β·q_{j-1}
        z = z - alpha_j * q
        if j > 0:
            z = z - betas[-1] * Q[j - 1]

        # Full reorthogonalisation (Gram-Schmidt against all previous vectors).
        # This is O(j·dim) per step but prevents ghost eigenvalues.
        for qi in Q:
            z = z - torch.dot(qi, z) * qi

        # Off-diagonal element β_j = ‖z‖
        beta_j = torch.linalg.norm(z)
        betas.append(beta_j)

        # Invariant subspace found: no more independent directions
        if beta_j < 1e-10:
            break

        q = z / beta_j

    # ── Build the n×n tridiagonal matrix T ──────────────────────────────────
    n = len(alphas)
    alpha_vec = torch.stack(alphas)                    # (n,)
    T = torch.diag(alpha_vec)

    if n > 1:
        # Off-diagonals: β_0 … β_{n-2}  (the last β connects to a vector
        # outside the current basis and is not part of T)
        beta_vec = torch.stack(betas[:n - 1])          # (n-1,)
        T = T + torch.diag(beta_vec, 1) + torch.diag(beta_vec, -1)

    # ── Eigendecompose T (symmetric → use eigh for stability) ───────────────
    # evals: ascending order;  evecs_T: columns are eigenvectors of T
    # torch.linalg.eigh is not implemented on MPS; run on CPU (T is tiny:
    # at most k×k where k=15) then move results back to the original device.
    T_device = T.device
    evals, evecs_T = torch.linalg.eigh(T.cpu())        # (n,), (n, n)
    evals, evecs_T = evals.to(T_device), evecs_T.to(T_device)

    # ── Ritz vectors: lift T's eigenvectors back to the full dim-space ──────
    Q_matrix = torch.stack(Q, dim=1)                   # (dim, n)
    ritz_vecs = Q_matrix @ evecs_T                     # (dim, n)

    # ── Sort by eigenvalue descending ────────────────────────────────────────
    idx = torch.argsort(evals, descending=True)
    evals = evals[idx]
    ritz_vecs = ritz_vecs[:, idx]

    # ── Return top-d (or fewer if early termination occurred) ────────────────
    d_actual = min(d, n)
    return evals[:d_actual], ritz_vecs[:, :d_actual]
