"""Tests for PrecondER — natural-gradient experience replay (CG solve).

Coverage:
  (a) Fisher oracle: PSD, symmetric, matches a dense GGN reference.
  (b) CG solver: solves (A+δI)x=b on a known matrix; warm-start; zero rhs.
  (c) Structural: PrecondER subclasses BaseMethod; bad config raises.
  (d) Task-0 fallback: observe() before any end_task is plain SGD.
  (e) Natural-gradient step: damped solve matches a dense reference; large
      damping → plain ER; never amplifies.
  (f) Buffer population: end_task fills the buffer with correct task_ids.
  (g) Diagnostics shape and values; warm-start reset at task boundary.
  (h) Forgetting prevention: after 2 tasks, task-0 accuracy stays reasonable.
"""

import sys
import os
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.methods.precond_er import PrecondER
from src.methods.base_method import BaseMethod
from src.optim.fisher import make_fisher_vp, conjugate_gradient
from src.data.memory_buffer import ReservoirBuffer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_cfg(
    target: str = "joint",
    damping: float = 1.0,
    cg_iters: int = 50,
    cg_tol: float = 1e-8,
    warm_start: bool = True,
    replay_batch_size=None,
    lr: float = 0.05,
) -> SimpleNamespace:
    return SimpleNamespace(
        method=SimpleNamespace(
            name="precond_er",
            replay_batch_size=replay_batch_size,
            fisher=SimpleNamespace(target=target, damping=damping),
            cg=SimpleNamespace(iters=cg_iters, tol=cg_tol, warm_start=warm_start),
        ),
        training=SimpleNamespace(lr=lr, weight_decay=0.0),
    )


class TinyMLP(nn.Module):
    def __init__(self, in_dim: int = 12, hidden: int = 10, out_dim: int = 4) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _make_loader(n: int = 48, in_dim: int = 12, n_classes: int = 4, seed: int = 0) -> DataLoader:
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, in_dim, generator=g)
    y = torch.randint(0, n_classes, (n,), generator=g)
    return DataLoader(TensorDataset(x, y), batch_size=8, shuffle=False)


def _flat_params(params):
    return torch.cat([p.detach().reshape(-1) for p in params])


def _dense_fisher(model, x, params):
    """Reference Fisher = (1/B) Σ_b Jᵀ_b (diag(p_b) − p_b p_bᵀ) J_b, built densely."""
    n = sum(p.numel() for p in params)
    B = x.shape[0]
    logits = model(x)
    C = logits.shape[1]
    probs = torch.softmax(logits, dim=1).detach()

    rows = []
    for b in range(B):
        for c in range(C):
            grads = torch.autograd.grad(logits[b, c], params, retain_graph=True)
            rows.append(torch.cat([g.reshape(-1) for g in grads]))
    J = torch.stack(rows).reshape(B, C, n)

    F_mat = torch.zeros(n, n)
    for b in range(B):
        p = probs[b]
        H = torch.diag(p) - torch.outer(p, p)
        Jb = J[b]
        F_mat += Jb.t() @ H @ Jb
    return F_mat / B


# ---------------------------------------------------------------------------
# (a) Fisher oracle correctness
# ---------------------------------------------------------------------------

class TestFisherOracle:
    def _setup(self):
        torch.manual_seed(0)
        model = TinyMLP()
        params = [p for p in model.parameters() if p.requires_grad]
        x = torch.randn(6, 12)
        y = torch.randint(0, 4, (6,))
        return model, params, x, y

    def test_matches_dense_ggn(self):
        model, params, x, y = self._setup()
        n = sum(p.numel() for p in params)
        fvp = make_fisher_vp(model, x, y, params)
        F_ref = _dense_fisher(model, x, params)
        for i in range(0, n, max(1, n // 8)):
            e = torch.zeros(n)
            e[i] = 1.0
            assert torch.allclose(fvp(e), F_ref[:, i], atol=1e-4), f"col {i}"

    def test_psd(self):
        model, params, x, y = self._setup()
        n = sum(p.numel() for p in params)
        fvp = make_fisher_vp(model, x, y, params)
        for s in range(5):
            v = torch.randn(n, generator=torch.Generator().manual_seed(s))
            assert torch.dot(v, fvp(v)).item() >= -1e-5

    def test_symmetric(self):
        model, params, x, y = self._setup()
        n = sum(p.numel() for p in params)
        fvp = make_fisher_vp(model, x, y, params)
        u = torch.randn(n, generator=torch.Generator().manual_seed(1))
        v = torch.randn(n, generator=torch.Generator().manual_seed(2))
        assert abs(torch.dot(u, fvp(v)).item() - torch.dot(v, fvp(u)).item()) < 1e-4

    def test_ignores_labels(self):
        model, params, x, y = self._setup()
        n = sum(p.numel() for p in params)
        v = torch.randn(n, generator=torch.Generator().manual_seed(3))
        f1 = make_fisher_vp(model, x, y, params)(v)
        f2 = make_fisher_vp(model, x, (y + 1) % 4, params)(v)
        assert torch.allclose(f1, f2, atol=1e-6)


# ---------------------------------------------------------------------------
# (b) Conjugate-gradient solver
# ---------------------------------------------------------------------------

class TestCG:
    def _spd_system(self, n=20, seed=0):
        # float64: keeps CG's "converges within n steps" property exact (float32
        # loses basis orthogonality and can need extra iters for tight tols).
        g = torch.Generator().manual_seed(seed)
        M = torch.randn(n, n, generator=g, dtype=torch.float64)
        A = M @ M.t()                       # PSD
        b = torch.randn(n, generator=g, dtype=torch.float64)
        return A, b

    def test_solves_known_system(self):
        A, b = self._spd_system()
        delta = 0.5
        x, n_iter = conjugate_gradient(lambda v: A @ v, b, damping=delta, iters=200, tol=1e-10)
        x_ref = torch.linalg.solve(A + delta * torch.eye(A.shape[0], dtype=A.dtype), b)
        assert torch.allclose(x, x_ref, atol=1e-4)
        # CG converges in ~n steps (a few extra from finite-precision orthogonality
        # loss), and well before the 200-iteration cap.
        assert n_iter <= 2 * A.shape[0]

    def test_zero_rhs_returns_zero(self):
        A, _ = self._spd_system()
        x, n_iter = conjugate_gradient(lambda v: A @ v, torch.zeros(A.shape[0]),
                                       damping=1.0, iters=50)
        assert torch.allclose(x, torch.zeros(A.shape[0]))
        assert n_iter == 0

    def test_warm_start_reduces_iters(self):
        A, b = self._spd_system()
        delta = 0.5
        x_full, iters_cold = conjugate_gradient(lambda v: A @ v, b, damping=delta,
                                                iters=200, tol=1e-8)
        # Warm-start from the (near-exact) solution → converges almost immediately.
        _, iters_warm = conjugate_gradient(lambda v: A @ v, b, damping=delta,
                                           iters=200, tol=1e-8, x0=x_full)
        assert iters_warm <= iters_cold

    def test_respects_tolerance(self):
        A, b = self._spd_system()
        delta = 1.0
        x, _ = conjugate_gradient(lambda v: A @ v, b, damping=delta, iters=200, tol=1e-6)
        res = torch.linalg.norm((A + delta * torch.eye(A.shape[0])) @ x - b)
        assert res <= 1e-6 * torch.linalg.norm(b) * 10  # small slack


# ---------------------------------------------------------------------------
# (c) Structural checks
# ---------------------------------------------------------------------------

class TestStructure:
    def test_is_base_method_subclass(self):
        assert issubclass(PrecondER, BaseMethod)

    def test_instantiates(self):
        assert isinstance(PrecondER(TinyMLP(), _make_cfg(), ReservoirBuffer(100)), PrecondER)

    def test_bad_target_raises(self):
        with pytest.raises(ValueError, match="fisher.target"):
            PrecondER(TinyMLP(), _make_cfg(target="bogus"), ReservoirBuffer(100))

    def test_nonpositive_damping_raises(self):
        with pytest.raises(ValueError, match="damping"):
            PrecondER(TinyMLP(), _make_cfg(damping=0.0), ReservoirBuffer(100))

    def test_bad_cg_iters_raises(self):
        with pytest.raises(ValueError, match="cg.iters"):
            PrecondER(TinyMLP(), _make_cfg(cg_iters=0), ReservoirBuffer(100))


# ---------------------------------------------------------------------------
# (d) Task-0 fallback is plain SGD
# ---------------------------------------------------------------------------

class TestTask0Fallback:
    def test_task0_updates_params(self):
        model = TinyMLP()
        m = PrecondER(model, _make_cfg(), ReservoirBuffer(100))
        before = _flat_params(m._params).clone()
        out = m.observe(torch.randn(8, 12), torch.randint(0, 4, (8,)), task_id=0)
        assert "loss" in out
        assert not torch.allclose(before, _flat_params(m._params))

    def test_task0_diagnostics_are_zero(self):
        m = PrecondER(TinyMLP(), _make_cfg(), ReservoirBuffer(100))
        m.observe(torch.randn(8, 12), torch.randint(0, 4, (8,)), task_id=0)
        diag = m.get_step_diagnostics()
        assert diag["cg_iters"] == 0
        assert diag["precond_ratio"] == 1.0


# ---------------------------------------------------------------------------
# (e) Natural-gradient step semantics
# ---------------------------------------------------------------------------

class TestNaturalGradient:
    @staticmethod
    def _fixed_batches():
        xc = torch.randn(8, 12, generator=torch.Generator().manual_seed(1))
        yc = torch.randint(0, 4, (8,), generator=torch.Generator().manual_seed(2))
        xr = torch.randn(8, 12, generator=torch.Generator().manual_seed(3))
        yr = torch.randint(0, 4, (8,), generator=torch.Generator().manual_seed(4))
        return xc, yc, xr, yr

    @staticmethod
    def _restore(params, flat):
        with torch.no_grad():
            off = 0
            for p in params:
                p.copy_(flat[off : off + p.numel()].view_as(p))
                off += p.numel()

    def _joint_grad(self, model, params, xc, yc, xr, yr):
        for p in params:
            p.grad = None
        loss = nn.CrossEntropyLoss()(model(xc), yc) + nn.CrossEntropyLoss()(model(xr), yr)
        loss.backward()
        return torch.cat([p.grad.reshape(-1) for p in params])

    def test_step_matches_dense_natural_gradient(self):
        """The applied update equals lr · δ · (F+δI)⁻¹ g, computed densely."""
        torch.manual_seed(0)
        model = TinyMLP()
        buf = ReservoirBuffer(500)
        delta, lr = 1.0, 0.1
        m = PrecondER(model, _make_cfg(target="replay", damping=delta, lr=lr,
                                       cg_iters=400, cg_tol=1e-12, warm_start=False), buf)
        m.end_task(0, _make_loader(n=32, seed=0))

        xc, yc, xr, yr = self._fixed_batches()
        buf.sample = lambda bs: (xr, yr, [0] * bs)  # deterministic replay draw

        params = m._params
        before = _flat_params(params).clone()
        m.observe(xc, yc, task_id=1)
        applied = before - _flat_params(params)     # SGD: θ -= lr·d ⇒ applied = lr·d

        # Reconstruct g and (replay-target) F at the original weights.
        self._restore(params, before)
        g = self._joint_grad(model, params, xc, yc, xr, yr)
        F = _dense_fisher(model, xr, params)
        n = F.shape[0]
        d_ref = delta * torch.linalg.solve(F + delta * torch.eye(n), g)
        assert torch.allclose(applied, lr * d_ref, atol=1e-4)

    def test_large_damping_recovers_plain_er(self):
        """δ → ∞ ⇒ δ·(F+δI)⁻¹ g → g, so the update equals a plain ER step."""
        torch.manual_seed(0)
        model = TinyMLP()
        buf = ReservoirBuffer(500)
        lr = 0.1
        m = PrecondER(model, _make_cfg(target="joint", damping=1e8, lr=lr,
                                       cg_iters=50, cg_tol=1e-12, warm_start=False), buf)
        m.end_task(0, _make_loader(n=32, seed=0))

        xc, yc, xr, yr = self._fixed_batches()
        buf.sample = lambda bs: (xr, yr, [0] * bs)

        params = m._params
        before = _flat_params(params).clone()
        m.observe(xc, yc, task_id=1)
        applied = before - _flat_params(params)

        self._restore(params, before)
        g = self._joint_grad(model, params, xc, yc, xr, yr)
        assert torch.allclose(applied, lr * g, atol=1e-3)  # ≈ plain ER step lr·g

    def test_precond_ratio_not_above_one(self):
        torch.manual_seed(0)
        m = PrecondER(TinyMLP(), _make_cfg(damping=0.1), ReservoirBuffer(200))
        m.end_task(0, _make_loader(seed=0))
        m.observe(torch.randn(8, 12), torch.randint(0, 4, (8,)), task_id=1)
        # δ·(F+δI)⁻¹ has spectral norm ≤ 1 ⇒ ‖d‖ ≤ ‖g‖.
        assert m.get_step_diagnostics()["precond_ratio"] <= 1.0 + 1e-4


# ---------------------------------------------------------------------------
# (f) Buffer population
# ---------------------------------------------------------------------------

class TestBuffer:
    def test_end_task_fills_buffer(self):
        buf = ReservoirBuffer(1000)
        PrecondER(TinyMLP(), _make_cfg(), buf).end_task(0, _make_loader(n=48, seed=0))
        assert len(buf) == 48
        assert set(buf._task_ids) == {0}

    def test_end_task_tags_task_ids(self):
        buf = ReservoirBuffer(1000)
        m = PrecondER(TinyMLP(), _make_cfg(), buf)
        m.end_task(0, _make_loader(n=24, seed=0))
        m.end_task(1, _make_loader(n=24, seed=1))
        assert set(buf._task_ids) == {0, 1}


# ---------------------------------------------------------------------------
# (g) Diagnostics & warm-start state
# ---------------------------------------------------------------------------

class TestDiagnostics:
    def test_diagnostics_keys_and_types(self):
        torch.manual_seed(0)
        m = PrecondER(TinyMLP(), _make_cfg(), ReservoirBuffer(200))
        m.end_task(0, _make_loader(seed=0))
        m.observe(torch.randn(8, 12), torch.randint(0, 4, (8,)), task_id=1)
        diag = m.get_step_diagnostics()
        assert set(diag) == {"cg_iters", "cg_residual", "precond_ratio"}
        assert isinstance(diag["cg_iters"], int) and diag["cg_iters"] >= 1
        assert diag["cg_residual"] >= 0.0
        assert diag["precond_ratio"] <= 1.0 + 1e-4

    def test_warm_start_cached_then_reset(self):
        torch.manual_seed(0)
        m = PrecondER(TinyMLP(), _make_cfg(warm_start=True), ReservoirBuffer(200))
        m.end_task(0, _make_loader(seed=0))
        assert m._prev_x is None
        m.observe(torch.randn(8, 12), torch.randint(0, 4, (8,)), task_id=1)
        assert m._prev_x is not None                 # cached after a step
        m.end_task(1, _make_loader(seed=1))
        assert m._prev_x is None                      # reset at task boundary

    def test_warm_start_disabled_keeps_none(self):
        torch.manual_seed(0)
        m = PrecondER(TinyMLP(), _make_cfg(warm_start=False), ReservoirBuffer(200))
        m.end_task(0, _make_loader(seed=0))
        m.observe(torch.randn(8, 12), torch.randint(0, 4, (8,)), task_id=1)
        assert m._prev_x is None


# ---------------------------------------------------------------------------
# (h) End-to-end forgetting check
# ---------------------------------------------------------------------------

class TestForgetting:
    def test_two_task_run_retains_task0(self):
        torch.manual_seed(0)
        model = TinyMLP()
        buf = ReservoirBuffer(500)
        m = PrecondER(model, _make_cfg(lr=0.1, damping=1.0, cg_iters=10), buf)

        loader0 = _make_loader(n=64, seed=0)
        loader1 = _make_loader(n=64, seed=1)

        for _ in range(5):
            for x, y in loader0:
                m.observe(x, y, task_id=0)
        m.end_task(0, loader0)
        acc0_before = m.evaluate(loader0)

        for _ in range(5):
            for x, y in loader1:
                m.observe(x, y, task_id=1)
        m.end_task(1, loader1)
        acc0_after = m.evaluate(loader0)

        assert acc0_after > 0.3
        assert acc0_before > 0.3
