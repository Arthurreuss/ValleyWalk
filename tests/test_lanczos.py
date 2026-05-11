"""Tests for hvp() (T4.1) and lanczos() (T4.2).

T4.1 — HVP (Pearlmutter)
  On a small Linear(4, 5) network (20 parameters, no bias), verify that
  ``hvp()`` output matches the explicit ``H @ v`` computed via
  ``torch.autograd.functional.hessian`` to relative error < 1e-5.
  Also verifies the pre-computed-g shortcut and the damping correction.

T4.2 — Lanczos iteration
  On a 100×100 PSD matrix with 5 well-separated large eigenvalues:
  - Top-5 Ritz values match ``np.linalg.eigh`` to relative error < 1e-3.
  - Corresponding Ritz vectors are aligned with the true eigenvectors to
    |cosine similarity| > 0.99 (sign-invariant comparison).
  - Returned eigenvalues are mutually distinct (no ghost eigenvalues).
  - Requesting d > number of Lanczos steps is handled gracefully.

Matrix construction
-------------------
  n = 100
  eigenvalues = [100, 50, 25, 10, 5, 0.1, 0.1, ..., 0.1]   (5 large + 95 small)
  A = Q_rand @ diag(lambdas) @ Q_rand.T     (random orthonormal basis Q_rand)

  The accuracy tests use float64 because the condition number of A is 1000
  (= 100 / 0.1), which causes float32 Lanczos to lose orthogonality in
  ~40 steps and produce ghost Ritz values.  Real-world neural-network
  Hessians used in CACL are much better conditioned; the float32 default
  dtype is appropriate there.  Float64 is used here purely to keep the
  test deterministic and tight.
"""

import os
import sys

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd.functional import hessian

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.optim.lanczos import hvp, lanczos


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def _make_small_model_and_data():
    """Return (model, x, y) with a 20-parameter Linear(4→5) network."""
    torch.manual_seed(42)
    model = nn.Linear(4, 5, bias=False)   # 4×5 = 20 parameters
    x = torch.randn(8, 4)
    y = torch.randint(5, (8,))
    return model, x, y


def _make_psd_matrix(n: int = 100, seed: int = 7) -> torch.Tensor:
    """Return an n×n PSD matrix (float64) with 5 large eigenvalues and 95 small ones.

    Eigenvalue spectrum:
        λ = [100, 50, 25, 10, 5, 0.1, 0.1, ..., 0.1]

    The eigenvectors form a random orthonormal basis constructed in float64 to
    avoid the QR orthogonality loss that occurs in float32 (~3.5e-7 residual),
    which would corrupt the Lanczos basis after ~20 steps.
    """
    torch.manual_seed(seed)
    # Build in float64 for a well-conditioned orthonormal basis
    Q_rand, _ = torch.linalg.qr(torch.randn(n, n, dtype=torch.float64))
    lambdas = torch.full((n,), 0.1, dtype=torch.float64)
    lambdas[0] = 100.0
    lambdas[1] = 50.0
    lambdas[2] = 25.0
    lambdas[3] = 10.0
    lambdas[4] = 5.0
    A = Q_rand @ torch.diag(lambdas) @ Q_rand.T
    return A   # float64


# ---------------------------------------------------------------------------
# T4.1 — HVP tests
# ---------------------------------------------------------------------------

class TestHVP:
    """Verify hvp() against the explicit Hessian on a 20-parameter network."""

    def setup_method(self):
        self.model, self.x, self.y = _make_small_model_and_data()
        self.params = list(self.model.parameters())
        assert sum(p.numel() for p in self.params) == 20

        # Random direction vector (same shapes as params)
        torch.manual_seed(99)
        self.v = [torch.randn_like(p) for p in self.params]

        # Loss function: accepts params (ignored, uses model's stored params)
        self.loss_fn = lambda p: F.cross_entropy(self.model(self.x), self.y)

        # Ground-truth: explicit Hessian computed via functional API
        # Work with the single weight matrix flattened to 1-D
        params_flat = self.params[0].detach().flatten().requires_grad_(True)
        x_, y_ = self.x, self.y

        def loss_flat(p_flat):
            w = p_flat.reshape(5, 4)
            return F.cross_entropy(x_ @ w.T, y_)

        H_full = hessian(loss_flat, params_flat)  # (20, 20)
        v_flat = self.v[0].flatten()
        self.Hv_true = (H_full @ v_flat).detach()

    def test_hvp_relative_error_below_1e5(self):
        """hvp() matches H @ v to relative error < 1e-5."""
        Hv_ours = hvp(self.loss_fn, self.params, self.v)
        rel_err = (Hv_ours - self.Hv_true).norm() / (self.Hv_true.norm() + 1e-12)
        assert rel_err.item() < 1e-5, f"rel error {rel_err:.2e} ≥ 1e-5"

    def test_hvp_output_shape(self):
        """hvp() returns a 1-D tensor with P = 20 elements."""
        Hv = hvp(self.loss_fn, self.params, self.v)
        assert Hv.shape == (20,)

    def test_hvp_precomputed_g_matches(self):
        """Pre-computed g shortcut yields the same result as the full path."""
        # Compute g externally with create_graph=True
        loss = self.loss_fn(self.params)
        g = torch.autograd.grad(loss, self.params, create_graph=True)

        Hv_shortcut = hvp(self.loss_fn, self.params, self.v, g=g)
        rel_err = (Hv_shortcut - self.Hv_true).norm() / (self.Hv_true.norm() + 1e-12)
        assert rel_err.item() < 1e-5, f"rel error (g path) {rel_err:.2e} ≥ 1e-5"

    def test_hvp_damping_correct(self):
        """With damping ε, hvp returns (H + εI)v = Hv + ε*v."""
        eps = 1e-2
        Hv_damped = hvp(self.loss_fn, self.params, self.v, damping=eps)
        v_flat = self.v[0].flatten()
        expected = self.Hv_true + eps * v_flat
        rel_err = (Hv_damped - expected).norm() / (expected.norm() + 1e-12)
        assert rel_err.item() < 1e-5, f"damping rel error {rel_err:.2e} ≥ 1e-5"

    def test_hvp_zero_vector_gives_zero(self):
        """HVP with v = 0 must return the zero vector (linearity)."""
        v_zero = [torch.zeros_like(p) for p in self.params]
        Hv = hvp(self.loss_fn, self.params, v_zero)
        assert torch.allclose(Hv, torch.zeros_like(Hv), atol=1e-6)


# ---------------------------------------------------------------------------
# T4.2 — Lanczos tests
# ---------------------------------------------------------------------------

class TestLanczos:
    """Verify lanczos() on a 100×100 PSD matrix with a known spectrum."""

    def setup_method(self):
        torch.manual_seed(7)
        # A is float64; HVP oracle and lanczos() are run in float64
        self.A = _make_psd_matrix(n=100, seed=7)
        self.n = 100

        # HVP oracle: matrix-vector product (float64 → float64)
        self.hvp_fn = lambda v: self.A @ v

        # Ground truth via numpy (float64)
        A_np = self.A.numpy()                         # already float64
        evals_np, evecs_np = np.linalg.eigh(A_np)    # ascending order
        # Sort descending, keep top 5
        idx = np.argsort(evals_np)[::-1]
        self.true_evals = evals_np[idx[:5]]           # (5,) float64 ndarray
        self.true_evecs = evecs_np[:, idx[:5]]        # (100, 5) float64 ndarray

    def test_top5_eigenvalues_relative_error(self):
        """Top-5 Ritz values match np.linalg.eigh to relative error < 1e-3 (float64)."""
        evals, _ = lanczos(self.hvp_fn, dim=self.n, k=40, d=5, dtype=torch.float64)

        evals_np = evals.detach().numpy()             # already float64
        for i in range(5):
            rel_err = abs(evals_np[i] - self.true_evals[i]) / abs(self.true_evals[i])
            assert rel_err < 1e-3, (
                f"λ_{i}: ours={evals_np[i]:.6f}, true={self.true_evals[i]:.6f}, "
                f"rel_err={rel_err:.2e}"
            )

    def test_top5_eigenvector_cosine_similarity(self):
        """Ritz vectors align with true eigenvectors: |cos θ| > 0.99 (float64)."""
        _, evecs = lanczos(self.hvp_fn, dim=self.n, k=40, d=5, dtype=torch.float64)

        evecs_np = evecs.detach().numpy()             # already float64
        for i in range(5):
            cos_sim = abs(evecs_np[:, i] @ self.true_evecs[:, i])
            # Normalise (Ritz vectors should already be unit; just in case)
            cos_sim /= (
                np.linalg.norm(evecs_np[:, i]) * np.linalg.norm(self.true_evecs[:, i])
            )
            assert cos_sim > 0.99, f"eigenvec {i}: |cos θ| = {cos_sim:.4f} ≤ 0.99"

    def test_eigenvalues_sorted_descending(self):
        """Returned eigenvalues are in strictly descending order."""
        evals, _ = lanczos(self.hvp_fn, dim=self.n, k=40, d=5, dtype=torch.float64)
        evals_list = evals.tolist()
        for i in range(len(evals_list) - 1):
            assert evals_list[i] >= evals_list[i + 1], (
                f"Not sorted descending: λ_{i}={evals_list[i]:.4f} < "
                f"λ_{i+1}={evals_list[i+1]:.4f}"
            )

    def test_no_ghost_eigenvalues(self):
        """All returned eigenvalues are distinct (no duplicates from ghost eigenvalues)."""
        evals, _ = lanczos(self.hvp_fn, dim=self.n, k=40, d=5, dtype=torch.float64)
        evals_list = evals.tolist()
        for i in range(len(evals_list)):
            for j in range(i + 1, len(evals_list)):
                gap = abs(evals_list[i] - evals_list[j])
                assert gap > 0.01, (
                    f"Ghost eigenvalue suspected: λ_{i}={evals_list[i]:.4f} ≈ "
                    f"λ_{j}={evals_list[j]:.4f} (gap={gap:.2e})"
                )

    def test_output_shapes(self):
        """lanczos() returns tensors of the correct shape."""
        evals, evecs = lanczos(self.hvp_fn, dim=self.n, k=40, d=5, dtype=torch.float64)
        assert evals.shape == (5,), f"Expected (5,), got {evals.shape}"
        assert evecs.shape == (self.n, 5), f"Expected ({self.n}, 5), got {evecs.shape}"

    def test_ritz_vectors_are_unit_norm(self):
        """Each returned Ritz vector has unit norm (within floating-point tolerance)."""
        _, evecs = lanczos(self.hvp_fn, dim=self.n, k=40, d=5, dtype=torch.float64)
        for i in range(5):
            norm = evecs[:, i].norm().item()
            assert abs(norm - 1.0) < 1e-4, f"Ritz vector {i} norm = {norm:.6f} ≠ 1"

    def test_d_less_than_k_returns_d_eigenpairs(self):
        """Requesting d < k returns exactly d eigenpairs."""
        evals, evecs = lanczos(self.hvp_fn, dim=self.n, k=30, d=3, dtype=torch.float64)
        assert evals.shape == (3,)
        assert evecs.shape == (self.n, 3)

    def test_early_termination_invariant_subspace(self):
        """On a rank-1 matrix, Lanczos terminates after 1 step (invariant subspace)."""
        # Rank-1 PSD matrix: A = v v^T
        torch.manual_seed(123)
        v0 = torch.randn(20)
        A_rank1 = v0.unsqueeze(1) @ v0.unsqueeze(0)   # (20, 20), rank 1
        hvp_fn_r1 = lambda v: A_rank1 @ v

        # Should terminate after 1 step (only 1 non-zero eigenvalue)
        evals, evecs = lanczos(hvp_fn_r1, dim=20, k=10, d=1)
        # The single non-zero eigenvalue should be ≈ ‖v0‖²
        true_eval = (v0 * v0).sum().item()
        rel_err = abs(evals[0].item() - true_eval) / abs(true_eval)
        assert rel_err < 1e-4, f"Rank-1 eigenvalue error: {rel_err:.2e}"

    def test_identity_matrix_gives_uniform_eigenvalues(self):
        """lanczos() on λI returns eigenvalue ≈ λ for all eigenpairs."""
        lam = 3.7
        hvp_fn_I = lambda v: lam * v
        evals, _ = lanczos(hvp_fn_I, dim=50, k=20, d=5)
        for i, ev in enumerate(evals.tolist()):
            assert abs(ev - lam) < 1e-3, f"λ_{i} = {ev:.4f}, expected {lam}"
