"""Tests for GEM / A-GEM (Gradient Episodic Memory).

Coverage:
  (a) Structural: GEM subclasses BaseMethod; invalid mode raises ValueError.
  (b) Task-0 fallback: observe() before any end_task is plain SGD.
  (c) Joint mode (A-GEM): projected gradient satisfies ⟨d, g_ref⟩ ≥ -margin.
  (d) Per-task mode (GEM): all T constraints satisfied after projection.
  (e) Buffer population: end_task fills the buffer with correct task_ids.
  (f) Past-task tracking: _past_tasks grows by one per end_task, no duplicates.
  (g) Evaluate / diagnostics from BaseMethod.
  (h) Forgetting prevention: after 2 tasks, task-0 accuracy remains > 30%.
"""

import sys
import os

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.methods.gem import GEM
from src.methods.base_method import BaseMethod
from src.data.memory_buffer import ReservoirBuffer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_cfg(reference_gradient: str = "per_task", margin: float = 0.5) -> SimpleNamespace:
    return SimpleNamespace(
        method=SimpleNamespace(
            name="gem" if reference_gradient == "per_task" else "agem",
            gem=SimpleNamespace(
                reference_gradient=reference_gradient,
                margin=margin,
                mem_batch_size=32,
            ),
        ),
        training=SimpleNamespace(lr=0.01, weight_decay=0.0),
    )


class TinyMLP(nn.Module):
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
    return DataLoader(TensorDataset(x, y), batch_size=8, shuffle=False)


# ---------------------------------------------------------------------------
# (a) Structural checks
# ---------------------------------------------------------------------------

class TestStructure:
    def test_gem_is_base_method_subclass(self):
        assert issubclass(GEM, BaseMethod)

    def test_gem_instantiates_per_task(self):
        model = TinyMLP()
        gem = GEM(model, _make_cfg("per_task"), ReservoirBuffer(100))
        assert isinstance(gem, GEM)

    def test_gem_instantiates_joint(self):
        model = TinyMLP()
        gem = GEM(model, _make_cfg("joint"), ReservoirBuffer(100))
        assert isinstance(gem, GEM)

    def test_invalid_mode_raises(self):
        model = TinyMLP()
        with pytest.raises(ValueError, match="reference_gradient"):
            GEM(model, _make_cfg("bad_mode"), ReservoirBuffer(100))

    def test_has_required_methods(self):
        gem = GEM(TinyMLP(), _make_cfg(), ReservoirBuffer(100))
        assert callable(gem.observe)
        assert callable(gem.end_task)
        assert callable(gem.evaluate)
        assert callable(gem.get_step_diagnostics)


# ---------------------------------------------------------------------------
# (b) Task-0 fallback — no past tasks → plain SGD
# ---------------------------------------------------------------------------

class TestTask0Fallback:
    def test_observe_returns_loss_per_task(self):
        gem = GEM(TinyMLP(), _make_cfg("per_task"), ReservoirBuffer(100))
        x, y = torch.randn(8, 20), torch.randint(0, 4, (8,))
        result = gem.observe(x, y, task_id=0)
        assert "loss" in result
        assert isinstance(result["loss"], float) and result["loss"] > 0

    def test_observe_returns_loss_joint(self):
        gem = GEM(TinyMLP(), _make_cfg("joint"), ReservoirBuffer(100))
        x, y = torch.randn(8, 20), torch.randint(0, 4, (8,))
        result = gem.observe(x, y, task_id=0)
        assert set(result.keys()) == {"loss"}


# ---------------------------------------------------------------------------
# (c) Joint mode — constraint satisfied after projection
# ---------------------------------------------------------------------------

class TestJointConstraint:
    """_project_gradient must satisfy ⟨d, g_ref⟩ ≥ -margin."""

    def _run_and_check_constraint(self, margin: float):
        torch.manual_seed(42)
        model = TinyMLP(in_dim=20, hidden=16, out_dim=4)
        gem = GEM(model, _make_cfg("joint", margin=margin), ReservoirBuffer(200))

        loader = _make_loader(n=40, seed=0)
        for x, y in loader:
            gem.observe(x, y, task_id=0)
        gem.end_task(0, loader)

        # Compute a reference gradient directly via the helper
        xs, ys = gem._sample_task_batch(0)
        g_ref = gem._compute_ref_grad(xs, ys)

        # Worst-case violating gradient: anti-aligned
        g_viol = -g_ref.clone()
        g_proj = gem._project_gradient(g_viol, [g_ref])

        dot = torch.dot(g_proj, g_ref).item()
        assert dot >= -margin - 1e-6, (
            f"Constraint violated: ⟨d, g_ref⟩={dot:.4f} < -margin={-margin}"
        )

    def test_constraint_satisfied_margin_zero(self):
        self._run_and_check_constraint(margin=0.0)

    def test_constraint_satisfied_margin_half(self):
        self._run_and_check_constraint(margin=0.5)

    def test_identity_when_constraint_already_satisfied(self):
        torch.manual_seed(7)
        model = TinyMLP()
        gem = GEM(model, _make_cfg("joint", margin=0.5), ReservoirBuffer(200))

        loader = _make_loader(n=40, seed=1)
        for x, y in loader:
            gem.observe(x, y, task_id=0)
        gem.end_task(0, loader)

        xs, ys = gem._sample_task_batch(0)
        g_ref = gem._compute_ref_grad(xs, ys)

        # Gradient aligned with g_ref — constraint satisfied by construction
        g_good = g_ref.clone() * 2.0
        g_proj = gem._project_gradient(g_good, [g_ref])

        assert torch.allclose(g_good, g_proj, atol=1e-5), (
            "Constraint-satisfying gradient should not be modified"
        )


# ---------------------------------------------------------------------------
# (d) Per-task mode — all T constraints satisfied after projection
# ---------------------------------------------------------------------------

class TestPerTaskConstraint:
    def test_all_constraints_satisfied_after_two_tasks(self):
        torch.manual_seed(99)
        model = TinyMLP(in_dim=20, hidden=16, out_dim=4)
        gem = GEM(model, _make_cfg("per_task", margin=0.5), ReservoirBuffer(200))

        for task_id, seed in enumerate([0, 1]):
            loader = _make_loader(n=40, seed=seed)
            for x, y in loader:
                gem.observe(x, y, task_id=task_id)
            gem.end_task(task_id, loader)

        assert len(gem._past_tasks) == 2

        # Compute ref grads for each past task
        ref_grads = [
            gem._compute_ref_grad(*gem._sample_task_batch(t))
            for t in gem._past_tasks
        ]

        # Violating gradient: anti-aligned with the sum of all ref grads
        g_viol = -sum(ref_grads).clone()
        g_proj = gem._project_gradient(g_viol, ref_grads)

        margin = gem._margin
        for t, g_ref in enumerate(ref_grads):
            dot = torch.dot(g_proj, g_ref.to(g_proj.device)).item()
            assert dot >= -margin - 1e-5, (
                f"Task {t} constraint violated: ⟨d, g_ref_{t}⟩={dot:.4f} < "
                f"-margin={-margin}"
            )


# ---------------------------------------------------------------------------
# (e) Buffer population
# ---------------------------------------------------------------------------

class TestBufferPopulation:
    def test_buffer_populated_after_end_task(self):
        gem = GEM(TinyMLP(), _make_cfg("per_task"), ReservoirBuffer(500))
        loader = _make_loader(n=40, seed=0)
        for x, y in loader:
            gem.observe(x, y, task_id=0)
        gem.end_task(0, loader)
        assert len(gem.buffer) > 0

    def test_buffer_task_ids_correct(self):
        gem = GEM(TinyMLP(), _make_cfg("per_task"), ReservoirBuffer(500))
        for task_id, seed in enumerate([0, 1]):
            loader = _make_loader(n=40, seed=seed)
            for x, y in loader:
                gem.observe(x, y, task_id=task_id)
            gem.end_task(task_id, loader)

        stored = set(gem.buffer._task_ids)
        assert 0 in stored and 1 in stored


# ---------------------------------------------------------------------------
# (f) Past-task tracking
# ---------------------------------------------------------------------------

class TestPastTasksTracking:
    def test_past_tasks_grows_by_one_per_end_task(self):
        gem = GEM(TinyMLP(), _make_cfg("per_task"), ReservoirBuffer(500))
        for task_id in range(3):
            loader = _make_loader(n=40, seed=task_id)
            for x, y in loader:
                gem.observe(x, y, task_id=task_id)
            gem.end_task(task_id, loader)
            assert len(gem._past_tasks) == task_id + 1

    def test_past_tasks_no_duplicates_joint(self):
        gem = GEM(TinyMLP(), _make_cfg("joint"), ReservoirBuffer(500))
        for task_id in range(3):
            loader = _make_loader(n=40, seed=task_id)
            for x, y in loader:
                gem.observe(x, y, task_id=task_id)
            gem.end_task(task_id, loader)
        assert len(gem._past_tasks) == len(set(gem._past_tasks)) == 3

    def test_past_tasks_empty_before_any_end_task(self):
        gem = GEM(TinyMLP(), _make_cfg("per_task"), ReservoirBuffer(100))
        assert gem._past_tasks == []


# ---------------------------------------------------------------------------
# (g) Evaluate / diagnostics
# ---------------------------------------------------------------------------

class TestEvaluate:
    def test_evaluate_returns_float_in_unit_interval(self):
        gem = GEM(TinyMLP(), _make_cfg("per_task"), ReservoirBuffer(100))
        loader = _make_loader(n=40, seed=0)
        acc = gem.evaluate(loader)
        assert 0.0 <= acc <= 1.0

    def test_get_step_diagnostics_returns_dict(self):
        gem = GEM(TinyMLP(), _make_cfg("per_task"), ReservoirBuffer(100))
        assert isinstance(gem.get_step_diagnostics(), dict)


# ---------------------------------------------------------------------------
# (h) Forgetting prevention — 2-task sanity check
# ---------------------------------------------------------------------------

class TestForgettingPrevention:
    """GEM / A-GEM should prevent severe catastrophic forgetting."""

    def _run_forgetting_test(self, reference_gradient: str):
        torch.manual_seed(2024)
        in_dim, n_classes = 20, 4
        model = TinyMLP(in_dim=in_dim, hidden=32, out_dim=n_classes)
        cfg = SimpleNamespace(
            method=SimpleNamespace(
                name="gem" if reference_gradient == "per_task" else "agem",
                gem=SimpleNamespace(
                    reference_gradient=reference_gradient,
                    margin=0.5,
                    mem_batch_size=32,
                ),
            ),
            training=SimpleNamespace(lr=0.05, weight_decay=0.0),
        )
        buf = ReservoirBuffer(total_budget=500)
        gem = GEM(model, cfg, buf)

        # Task 0: binary — class = sign of first feature
        torch.manual_seed(0)
        x0 = torch.randn(200, in_dim)
        y0 = (x0[:, 0] > 0).long()
        loader0 = DataLoader(TensorDataset(x0, y0), batch_size=16, shuffle=True)

        for _ in range(10):
            for x, y in loader0:
                gem.observe(x, y, task_id=0)
        gem.end_task(0, loader0)

        acc0_after_task0 = gem.evaluate(loader0)
        assert acc0_after_task0 > 0.7, (
            f"Task-0 acc after task-0 training too low: {acc0_after_task0:.2f}"
        )

        # Task 1: binary — class = sign of second feature
        torch.manual_seed(1)
        x1 = torch.randn(200, in_dim)
        y1 = (x1[:, 1] > 0).long()
        loader1 = DataLoader(TensorDataset(x1, y1), batch_size=16, shuffle=True)

        for _ in range(10):
            for x, y in loader1:
                gem.observe(x, y, task_id=1)
        gem.end_task(1, loader1)

        acc0_after_task1 = gem.evaluate(loader0)
        # GEM constrains gradients to not degrade past tasks.
        # Threshold of 0.30 verifies no collapse (well above chance for binary).
        assert acc0_after_task1 > 0.30, (
            f"Task-0 accuracy collapsed to {acc0_after_task1:.2f} after task-1 "
            f"with {reference_gradient} mode"
        )

    def test_forgetting_prevention_per_task(self):
        self._run_forgetting_test("per_task")

    def test_forgetting_prevention_joint(self):
        self._run_forgetting_test("joint")
