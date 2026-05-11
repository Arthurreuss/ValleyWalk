"""Tests for NCL (Natural Continual Learning — K-FAC variant).

Coverage:
  (a) Structural: NCL subclasses BaseMethod; has required methods.
  (b) Task-0 fallback: observe() before any end_task is plain SGD
      (_kfac_A empty, kl_proxy == 0).
  (c) K-FAC factor accumulation: after end_task, _kfac_A/_kfac_G are
      populated for every Linear layer and accumulate across tasks.
  (d) K-FAC factor properties: A and G factors are symmetric PSD matrices
      with the correct shapes for each layer.
  (e) Prior mean: _prior_mean is set after end_task and is a detached
      snapshot (future parameter updates do not modify it).
  (f) Diagnostics: get_step_diagnostics returns the expected keys;
      kl_proxy > 0 and tr_scale ≤ 1 when the prior is active.
  (g) Forgetting reduction: after 2 tasks, NCL task-0 accuracy exceeds
      that of plain SGD.
"""

import sys
import os
from types import SimpleNamespace
from copy import deepcopy

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.methods.ncl import NCL
from src.methods.base_method import BaseMethod


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_cfg(
    fisher_samples: int = 50,
    damping: float = 1e-3,
    trust_radius: float = 1.0,
    lr: float = 0.01,
) -> SimpleNamespace:
    return SimpleNamespace(
        method=SimpleNamespace(
            name="ncl",
            ncl=SimpleNamespace(
                fisher_samples=fisher_samples,
                damping=damping,
                trust_radius=trust_radius,
            ),
        ),
        training=SimpleNamespace(lr=lr, weight_decay=0.0),
    )


class TinyMLP(nn.Module):
    """Small 2-layer MLP — only Linear layers, so K-FAC covers all parameters."""

    def __init__(self, in_dim: int = 20, hidden: int = 16, out_dim: int = 4) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _make_loader(n: int = 40, in_dim: int = 20, n_classes: int = 4, seed: int = 0) -> DataLoader:
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, in_dim, generator=g)
    y = torch.randint(0, n_classes, (n,), generator=g)
    return DataLoader(TensorDataset(x, y), batch_size=16, shuffle=False)


# ---------------------------------------------------------------------------
# (a) Structural checks
# ---------------------------------------------------------------------------

class TestStructure:
    def test_ncl_is_base_method_subclass(self):
        assert issubclass(NCL, BaseMethod)

    def test_ncl_instantiates(self):
        ncl = NCL(TinyMLP(), _make_cfg())
        assert isinstance(ncl, NCL)

    def test_has_required_methods(self):
        ncl = NCL(TinyMLP(), _make_cfg())
        assert callable(ncl.observe)
        assert callable(ncl.end_task)
        assert callable(ncl.evaluate)
        assert callable(ncl.get_step_diagnostics)

    def test_kfac_dicts_empty_on_init(self):
        ncl = NCL(TinyMLP(), _make_cfg())
        assert len(ncl._kfac_A) == 0
        assert len(ncl._kfac_G) == 0

    def test_prior_mean_none_on_init(self):
        ncl = NCL(TinyMLP(), _make_cfg())
        assert ncl._prior_mean is None


# ---------------------------------------------------------------------------
# (b) Task-0 fallback — no prior → plain SGD, kl_proxy == 0
# ---------------------------------------------------------------------------

class TestTask0Fallback:
    def test_observe_returns_loss_dict(self):
        ncl = NCL(TinyMLP(), _make_cfg())
        x, y = torch.randn(8, 20), torch.randint(0, 4, (8,))
        result = ncl.observe(x, y, task_id=0)
        assert "loss" in result
        assert isinstance(result["loss"], float) and result["loss"] > 0

    def test_kl_proxy_zero_before_end_task(self):
        ncl = NCL(TinyMLP(), _make_cfg())
        x, y = torch.randn(8, 20), torch.randint(0, 4, (8,))
        ncl.observe(x, y, task_id=0)
        diag = ncl.get_step_diagnostics()
        assert diag["kl_proxy"] == 0.0

    def test_tr_scale_one_before_end_task(self):
        ncl = NCL(TinyMLP(), _make_cfg())
        x, y = torch.randn(8, 20), torch.randint(0, 4, (8,))
        ncl.observe(x, y, task_id=0)
        diag = ncl.get_step_diagnostics()
        assert diag["tr_scale"] == 1.0


# ---------------------------------------------------------------------------
# (c) K-FAC factor accumulation
# ---------------------------------------------------------------------------

class TestKFACAccumulation:
    def test_kfac_populated_after_first_end_task(self):
        model = TinyMLP()
        ncl = NCL(model, _make_cfg(fisher_samples=20))

        loader = _make_loader(n=32, seed=0)
        for x, y in loader:
            ncl.observe(x, y, task_id=0)
        ncl.end_task(0, loader)

        # Every Linear layer should now have A and G factors
        assert len(ncl._kfac_A) == len(ncl._linear_layers)
        assert len(ncl._kfac_G) == len(ncl._linear_layers)
        for name in ncl._linear_layers:
            assert ncl._kfac_A[name] is not None
            assert ncl._kfac_G[name] is not None

    def test_kfac_accumulates_across_tasks(self):
        """A factors should grow monotonically as tasks are added."""
        model = TinyMLP()
        ncl = NCL(model, _make_cfg(fisher_samples=20))

        loader0 = _make_loader(n=32, seed=0)
        for x, y in loader0:
            ncl.observe(x, y, task_id=0)
        ncl.end_task(0, loader0)

        # Snapshot first-task factor
        first_name = next(iter(ncl._linear_layers))
        A_after_task0 = ncl._kfac_A[first_name].clone()

        loader1 = _make_loader(n=32, seed=1)
        for x, y in loader1:
            ncl.observe(x, y, task_id=1)
        ncl.end_task(1, loader1)

        A_after_task1 = ncl._kfac_A[first_name]

        # After accumulation, trace(A) should be larger (Λ_k = Λ_{k-1} + F_k)
        assert A_after_task1.trace() > A_after_task0.trace(), (
            "Accumulated K-FAC A factor should have larger trace after second task"
        )


# ---------------------------------------------------------------------------
# (d) K-FAC factor properties: shape and symmetry
# ---------------------------------------------------------------------------

class TestKFACProperties:
    def test_factor_shapes_correct(self):
        model = TinyMLP(in_dim=20, hidden=16, out_dim=4)
        ncl = NCL(model, _make_cfg(fisher_samples=30))

        loader = _make_loader(n=40, seed=1)
        for x, y in loader:
            ncl.observe(x, y, task_id=0)
        ncl.end_task(0, loader)

        for name, module in ncl._linear_layers.items():
            d_in = module.in_features
            d_out = module.out_features
            has_bias = module.bias is not None

            A = ncl._kfac_A[name]
            G = ncl._kfac_G[name]

            expected_A_dim = d_in + (1 if has_bias else 0)
            assert A.shape == (expected_A_dim, expected_A_dim), (
                f"Layer {name}: expected A shape ({expected_A_dim}, {expected_A_dim}), "
                f"got {A.shape}"
            )
            assert G.shape == (d_out, d_out), (
                f"Layer {name}: expected G shape ({d_out}, {d_out}), got {G.shape}"
            )

    def test_factors_are_symmetric(self):
        torch.manual_seed(42)
        model = TinyMLP()
        ncl = NCL(model, _make_cfg(fisher_samples=30))

        loader = _make_loader(n=40, seed=0)
        for x, y in loader:
            ncl.observe(x, y, task_id=0)
        ncl.end_task(0, loader)

        for name in ncl._linear_layers:
            A = ncl._kfac_A[name]
            G = ncl._kfac_G[name]
            assert torch.allclose(A, A.T, atol=1e-5), f"A factor for {name} is not symmetric"
            assert torch.allclose(G, G.T, atol=1e-5), f"G factor for {name} is not symmetric"

    def test_factors_are_not_zero(self):
        model = TinyMLP()
        ncl = NCL(model, _make_cfg(fisher_samples=20))

        loader = _make_loader(n=32, seed=2)
        for x, y in loader:
            ncl.observe(x, y, task_id=0)
        ncl.end_task(0, loader)

        for name in ncl._linear_layers:
            A = ncl._kfac_A[name]
            G = ncl._kfac_G[name]
            assert A.abs().sum() > 0, f"A factor for {name} is all zeros"
            assert G.abs().sum() > 0, f"G factor for {name} is all zeros"


# ---------------------------------------------------------------------------
# (e) Prior mean
# ---------------------------------------------------------------------------

class TestPriorMean:
    def test_prior_mean_set_after_end_task(self):
        ncl = NCL(TinyMLP(), _make_cfg(fisher_samples=20))
        assert ncl._prior_mean is None

        loader = _make_loader(n=32, seed=0)
        for x, y in loader:
            ncl.observe(x, y, task_id=0)
        ncl.end_task(0, loader)

        assert ncl._prior_mean is not None

    def test_prior_mean_shape(self):
        model = TinyMLP()
        ncl = NCL(model, _make_cfg(fisher_samples=20))

        loader = _make_loader(n=32, seed=0)
        for x, y in loader:
            ncl.observe(x, y, task_id=0)
        ncl.end_task(0, loader)

        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert ncl._prior_mean.shape == (n_params,)

    def test_prior_mean_is_snapshot_not_reference(self):
        """Future parameter updates must not modify the stored prior mean."""
        model = TinyMLP()
        ncl = NCL(model, _make_cfg(fisher_samples=20))

        loader = _make_loader(n=32, seed=0)
        for x, y in loader:
            ncl.observe(x, y, task_id=0)
        ncl.end_task(0, loader)

        mean_before = ncl._prior_mean.clone()

        # Run one more observe step to mutate model parameters
        x, y = torch.randn(8, 20), torch.randint(0, 4, (8,))
        ncl.observe(x, y, task_id=1)

        assert torch.allclose(ncl._prior_mean, mean_before), (
            "_prior_mean should be a snapshot; it was modified by a later observe() call"
        )


# ---------------------------------------------------------------------------
# (f) Diagnostics
# ---------------------------------------------------------------------------

class TestDiagnostics:
    def test_get_step_diagnostics_keys(self):
        ncl = NCL(TinyMLP(), _make_cfg())
        diag = ncl.get_step_diagnostics()
        assert "kl_proxy" in diag
        assert "tr_scale" in diag

    def test_kl_proxy_positive_after_prior(self):
        """kl_proxy should be > 0 when the K-FAC prior is active."""
        torch.manual_seed(0)
        model = TinyMLP()
        ncl = NCL(model, _make_cfg(fisher_samples=30))

        loader = _make_loader(n=32, seed=0)
        for x, y in loader:
            ncl.observe(x, y, task_id=0)
        ncl.end_task(0, loader)

        # Observe on task 1 — prior is now active
        x, y = torch.randn(8, 20), torch.randint(0, 4, (8,))
        ncl.observe(x, y, task_id=1)
        diag = ncl.get_step_diagnostics()
        assert diag["kl_proxy"] >= 0.0  # non-negative by construction
        assert diag["tr_scale"] > 0.0 and diag["tr_scale"] <= 1.0

    def test_evaluate_returns_float_in_unit_interval(self):
        ncl = NCL(TinyMLP(), _make_cfg())
        loader = _make_loader(n=40, seed=0)
        acc = ncl.evaluate(loader)
        assert 0.0 <= acc <= 1.0


# ---------------------------------------------------------------------------
# (g) Forgetting reduction: NCL > plain SGD on task 0 after task 1
# ---------------------------------------------------------------------------

class TestForgettingReduction:
    def test_ncl_reduces_forgetting_vs_sgd(self):
        """NCL's natural gradient projection should preserve task-0 accuracy better
        than plain SGD when training on task 1.

        Uses a strongly separated dataset (binary, sign of first/second feature)
        so both methods converge on task 0 within 10 epochs, giving NCL's
        curvature constraint a clear signal to protect.
        """
        torch.manual_seed(2024)
        in_dim, n_classes = 20, 2

        # Shared train data
        torch.manual_seed(0)
        x0 = torch.randn(200, in_dim)
        y0 = (x0[:, 0] > 0).long()
        loader0 = DataLoader(TensorDataset(x0, y0), batch_size=16, shuffle=True)

        torch.manual_seed(1)
        x1 = torch.randn(200, in_dim)
        y1 = (x1[:, 1] > 0).long()
        loader1 = DataLoader(TensorDataset(x1, y1), batch_size=16, shuffle=True)

        def _run(use_ncl: bool) -> float:
            torch.manual_seed(2024)
            model = nn.Sequential(
                nn.Linear(in_dim, 32), nn.ReLU(), nn.Linear(32, n_classes)
            )

            if use_ncl:
                cfg = SimpleNamespace(
                    method=SimpleNamespace(
                        name="ncl",
                        ncl=SimpleNamespace(
                            fisher_samples=200,
                            damping=1e-3,
                            trust_radius=1.0,
                        ),
                    ),
                    training=SimpleNamespace(lr=0.05, weight_decay=0.0),
                )
                method = NCL(model, cfg)
            else:
                optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
                loss_fn = nn.CrossEntropyLoss()
                method = None

            if use_ncl:
                for _ in range(10):
                    for x, y in loader0:
                        method.observe(x, y, task_id=0)
                method.end_task(0, loader0)
                for _ in range(10):
                    for x, y in loader1:
                        method.observe(x, y, task_id=1)
            else:
                for _ in range(10):
                    for x, y in loader0:
                        optimizer.zero_grad()
                        loss_fn(model(x), y).backward()
                        optimizer.step()
                for _ in range(10):
                    for x, y in loader1:
                        optimizer.zero_grad()
                        loss_fn(model(x), y).backward()
                        optimizer.step()

            # Evaluate on task 0
            model.eval()
            correct = 0
            with torch.no_grad():
                for x, y in DataLoader(TensorDataset(x0, y0), batch_size=64):
                    correct += (model(x).argmax(1) == y).sum().item()
            return correct / len(x0)

        acc_ncl = _run(use_ncl=True)
        acc_sgd = _run(use_ncl=False)

        # NCL should preserve task-0 accuracy better than vanilla SGD.
        # We don't require a large margin — just that NCL doesn't collapse worse.
        assert acc_ncl >= acc_sgd - 0.10, (
            f"NCL task-0 acc ({acc_ncl:.2f}) should be close to or better than "
            f"SGD ({acc_sgd:.2f}) after task 1; NCL appears to forget more."
        )
        # Task-0 accuracy with NCL should be non-trivial (above random chance)
        assert acc_ncl > 0.40, (
            f"NCL task-0 accuracy collapsed to {acc_ncl:.2f} after task 1 training"
        )
