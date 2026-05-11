"""Tests for ER's λ-curriculum schedule and curriculum-on gradient composition.

Covers the four schedules exposed by ``ER._curriculum_lambda``:

  - linear:    λ = progress
  - cosine:    λ = 0.5·(1 − cos(π·progress))
  - step:      λ = 0 for first N steps, then 1
  - adaptive:  λ = clip(EMA(‖g_replay‖/‖g_new‖), 0, 1)

And four invariants of ``ER._standard_step`` / ``ER._balanced_step``:

  (i)  λ = 1 reproduces vanilla ER (joint gradient = g_new + g_replay)
  (ii) λ_min floors the schedule output across schedules
  (iii) curriculum off (lambda_curriculum.enabled=false) returns λ = 1.0
  (iv) task_id = 0 returns λ = 1.0 regardless of schedule
"""

from __future__ import annotations

import math
import os
import sys
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

# Allow running from repo root without installation.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data.memory_buffer import ReservoirBuffer
from src.methods.er import ER


# ---------------------------------------------------------------------------
# Minimal config builder — only the fields ER actually reads.
# ---------------------------------------------------------------------------

def _make_cfg(
    *,
    mode: str = "standard",
    lc_enabled: bool = False,
    lc_schedule: str = "linear",
    lc_ramp_steps: int = 0,
    lc_ema_alpha: float = 0.05,
    lc_lambda_min: float = 0.0,
    lr_warmup_enabled: bool = False,
    lr_warmup_steps: int = 0,
    momentum: float = 0.0,
    lr: float = 0.1,
    normalize_components: bool = True,
    task_weighted: bool = False,
    reset_momentum_steps: int = 0,
    replay_batch_size=None,
    replay_full_buffer: bool = False,
):
    """Build a SimpleNamespace config matching the structure ER expects.

    SimpleNamespace supports attribute access AND .get() on the nested dicts
    we wrap in another helper namespace, which is sufficient for the fields
    accessed via ``cfg.method.get(...)`` in ER.__init__.
    """

    class _NS(SimpleNamespace):
        def get(self, key, default=None):
            return getattr(self, key, default)

    grad_balance = _NS(
        normalize_components=normalize_components,
        task_weighted=task_weighted,
    )
    lambda_curriculum = _NS(
        enabled=lc_enabled,
        ramp_steps=lc_ramp_steps,
        schedule=lc_schedule,
        ema_alpha=lc_ema_alpha,
        lambda_min=lc_lambda_min,
    )
    lr_warmup = _NS(
        enabled=lr_warmup_enabled,
        warmup_steps=lr_warmup_steps,
    )
    method = _NS(
        name="er",
        mode=mode,
        reset_momentum_steps=reset_momentum_steps,
        replay_batch_size=replay_batch_size,
        replay_full_buffer=replay_full_buffer,
        grad_balance=grad_balance,
        lambda_curriculum=lambda_curriculum,
        lr_warmup=lr_warmup,
    )
    training = _NS(
        lr=lr,
        momentum=momentum,
        weight_decay=0.0,
    )
    return _NS(method=method, training=training)


def _make_method(buffer_budget: int = 8, **cfg_kwargs) -> ER:
    """Tiny linear model + empty buffer + ER under the requested config."""
    model = nn.Linear(4, 2, bias=False)
    buffer = ReservoirBuffer(total_budget=buffer_budget)
    cfg = _make_cfg(**cfg_kwargs)
    return ER(model, cfg, buffer)


# ---------------------------------------------------------------------------
# Time-indexed schedules
# ---------------------------------------------------------------------------

class TestLinearSchedule:
    """λ(t) = clip(t / N, 0, 1)."""

    def test_progress_equals_task_step_over_ramp_steps(self):
        m = _make_method(
            lc_enabled=True, lc_schedule="linear", lc_ramp_steps=10,
        )
        # _task_step is incremented by observe(); here we set it manually
        # and call the private resolver directly.  We test 1-indexed positions
        # because ER uses (task_step - 1) / N for progress.
        for t in range(1, 11):
            m._task_step = t
            lam = m._curriculum_lambda(task_id=1)
            expected = (t - 1) / 10.0
            assert lam == pytest.approx(expected, abs=1e-12)

    def test_plateaus_at_one_after_ramp(self):
        m = _make_method(
            lc_enabled=True, lc_schedule="linear", lc_ramp_steps=10,
        )
        for t in (11, 50, 1_000_000):
            m._task_step = t
            assert m._curriculum_lambda(task_id=1) == 1.0

    def test_starts_at_zero(self):
        m = _make_method(
            lc_enabled=True, lc_schedule="linear", lc_ramp_steps=10,
        )
        m._task_step = 1  # the very first step of a new task
        assert m._curriculum_lambda(task_id=1) == 0.0


class TestCosineSchedule:
    """λ(t) = 0.5 · (1 − cos(π · progress))."""

    def test_endpoints(self):
        m = _make_method(
            lc_enabled=True, lc_schedule="cosine", lc_ramp_steps=10,
        )
        m._task_step = 1
        assert m._curriculum_lambda(task_id=1) == pytest.approx(0.0, abs=1e-12)
        m._task_step = 11
        assert m._curriculum_lambda(task_id=1) == 1.0  # post-ramp plateau

    def test_midpoint_is_half(self):
        # progress = 0.5  →  λ = 0.5 · (1 − cos(π/2)) = 0.5
        m = _make_method(
            lc_enabled=True, lc_schedule="cosine", lc_ramp_steps=10,
        )
        m._task_step = 6  # progress = (6 - 1) / 10 = 0.5
        assert m._curriculum_lambda(task_id=1) == pytest.approx(0.5, abs=1e-12)


class TestStepSchedule:
    """λ = 0 during the first N steps, then 1."""

    def test_zero_before_ramp_end(self):
        m = _make_method(
            lc_enabled=True, lc_schedule="step", lc_ramp_steps=10,
        )
        for t in range(1, 11):
            m._task_step = t
            assert m._curriculum_lambda(task_id=1) == 0.0

    def test_one_after_ramp(self):
        m = _make_method(
            lc_enabled=True, lc_schedule="step", lc_ramp_steps=10,
        )
        m._task_step = 11
        assert m._curriculum_lambda(task_id=1) == 1.0


# ---------------------------------------------------------------------------
# Adaptive schedule
# ---------------------------------------------------------------------------

class TestAdaptiveSchedule:
    """λ = clip(EMA(‖g_replay‖ / ‖g_new‖), 0, 1)."""

    def test_starts_near_zero_when_replay_gradient_vanishes(self):
        # At θ_0* the replay gradient is ≈ 0, so the instantaneous ratio is
        # ≈ 0 and the EMA tracks it.  λ should start near 0.
        m = _make_method(
            lc_enabled=True, lc_schedule="adaptive", lc_ema_alpha=0.5,
        )
        m._task_step = 1
        lam = m._curriculum_lambda(
            task_id=1, g_new_norm=1.0, g_replay_norm=1e-9,
        )
        assert lam == pytest.approx(0.5 * 1e-9, abs=1e-8)

    def test_climbs_to_one_as_gradients_rebalance(self):
        # Feed a stream of r = 1.0 inputs.  EMA → 1, λ → 1.
        m = _make_method(
            lc_enabled=True, lc_schedule="adaptive", lc_ema_alpha=0.5,
        )
        for _ in range(50):
            lam = m._curriculum_lambda(
                task_id=1, g_new_norm=1.0, g_replay_norm=1.0,
            )
        assert lam == pytest.approx(1.0, abs=1e-6)

    def test_clipped_above_at_one(self):
        # If g_replay > g_new, raw ratio > 1.  λ must clip to 1.
        m = _make_method(
            lc_enabled=True, lc_schedule="adaptive", lc_ema_alpha=1.0,
        )
        lam = m._curriculum_lambda(
            task_id=1, g_new_norm=1.0, g_replay_norm=5.0,
        )
        assert lam == 1.0

    def test_clipped_below_at_zero(self):
        # If g_replay = 0 exactly, EMA stays at 0 ⇒ λ = 0 (before floor).
        m = _make_method(
            lc_enabled=True, lc_schedule="adaptive", lc_ema_alpha=1.0,
        )
        lam = m._curriculum_lambda(
            task_id=1, g_new_norm=1.0, g_replay_norm=0.0,
        )
        assert lam == 0.0


# ---------------------------------------------------------------------------
# λ_min floor
# ---------------------------------------------------------------------------

class TestLambdaMinFloor:
    """max(λ_min, schedule(t)) — applies to all schedules."""

    def test_floors_linear_below_floor(self):
        m = _make_method(
            lc_enabled=True, lc_schedule="linear",
            lc_ramp_steps=10, lc_lambda_min=0.2,
        )
        m._task_step = 1   # raw λ = 0
        assert m._curriculum_lambda(task_id=1) == pytest.approx(0.2)
        m._task_step = 2   # raw λ = 0.1
        assert m._curriculum_lambda(task_id=1) == pytest.approx(0.2)
        m._task_step = 4   # raw λ = 0.3 > 0.2 — floor does not bind
        assert m._curriculum_lambda(task_id=1) == pytest.approx(0.3)

    def test_floors_adaptive_when_ratio_is_tiny(self):
        m = _make_method(
            lc_enabled=True, lc_schedule="adaptive",
            lc_ema_alpha=0.5, lc_lambda_min=0.1,
        )
        m._task_step = 1
        lam = m._curriculum_lambda(
            task_id=1, g_new_norm=1.0, g_replay_norm=1e-9,
        )
        assert lam == 0.1  # raw ≈ 5e-10 → floored to 0.1


# ---------------------------------------------------------------------------
# Disabled-path invariants
# ---------------------------------------------------------------------------

class TestDisabledPaths:
    """When the curriculum is off or task_id=0, λ must be 1.0."""

    def test_returns_one_when_disabled(self):
        m = _make_method(lc_enabled=False, lc_schedule="linear", lc_ramp_steps=10)
        m._task_step = 1
        assert m._curriculum_lambda(task_id=1) == 1.0

    def test_returns_one_on_task_zero_even_when_enabled(self):
        m = _make_method(lc_enabled=True, lc_schedule="linear", lc_ramp_steps=10)
        m._task_step = 1
        assert m._curriculum_lambda(task_id=0) == 1.0


# ---------------------------------------------------------------------------
# Joint-gradient composition: λ = 1 must reproduce vanilla ER exactly
# ---------------------------------------------------------------------------

class TestJointGradientAtLambdaOne:
    """When λ = 1, the curriculum-on standard step must produce the same
    joint gradient as plain ER: g_joint = g_new + g_replay.

    Because ReservoirBuffer.sample uses Python's ``random`` module (not
    torch), seeding torch alone doesn't make the two buffers draw the same
    replay batch.  We patch ``sample`` on both methods to return a fixed,
    identical (x, y, ids) tuple so the only varying factor is the
    curriculum-on vs curriculum-off code path.
    """

    @staticmethod
    def _fixed_sample(x_replay, y_replay):
        """Return a closure that ignores its size argument and returns the
        same (x, y, ids) each call — used to pin the replay batch."""
        def _sample(_batch_size):
            return x_replay.clone(), y_replay.clone(), [0] * x_replay.size(0)
        return _sample

    def test_lambda_one_matches_vanilla_er_gradient(self):
        torch.manual_seed(42)
        method_a = _make_method(
            lc_enabled=True, lc_schedule="linear", lc_ramp_steps=0,
        )
        method_b = _make_method(lc_enabled=False)

        # Force identical initial weights.
        with torch.no_grad():
            for p_a, p_b in zip(method_a.model.parameters(), method_b.model.parameters()):
                p_b.copy_(p_a)

        # Inject one dummy element into each buffer so len(buffer) > 0 and
        # ER takes the replay path instead of the task-0 SGD shortcut.
        method_a.buffer.add(torch.zeros(4), torch.tensor(0), task_id=0)
        method_b.buffer.add(torch.zeros(4), torch.tensor(0), task_id=0)

        # Pin both buffers' .sample to return the same fixed replay batch.
        x_replay = torch.randn(8, 4)
        y_replay = torch.randint(0, 2, (8,))
        method_a.buffer.sample = self._fixed_sample(x_replay, y_replay)
        method_b.buffer.sample = self._fixed_sample(x_replay, y_replay)

        # Identical current-task batch.
        x = torch.randn(8, 4)
        y = torch.randint(0, 2, (8,))

        method_a.observe(x.clone(), y.clone(), task_id=1)
        method_b.observe(x.clone(), y.clone(), task_id=1)

        # Compare resulting parameters — should match to numerical precision.
        for p_a, p_b in zip(method_a.model.parameters(), method_b.model.parameters()):
            assert torch.allclose(p_a, p_b, atol=1e-6), (
                "curriculum-on at λ=1 must reproduce vanilla ER"
            )


# ---------------------------------------------------------------------------
# Adaptive EMA reset between tasks
# ---------------------------------------------------------------------------

class TestAdaptiveEMAReset:
    """The adaptive EMA must reset at every task boundary so λ starts at 0
    on the first step of every new task (mirrors the slow-start behaviour of
    the time-indexed schedules).
    """

    def test_ema_resets_on_new_task(self):
        m = _make_method(lc_enabled=True, lc_schedule="adaptive", lc_ema_alpha=0.5)

        # Warm up the EMA on task 1.
        for _ in range(20):
            m._curriculum_lambda(task_id=1, g_new_norm=1.0, g_replay_norm=1.0)
        assert m._lc_r_ema == pytest.approx(1.0, abs=1e-6)

        # Manually trigger the task-boundary reset that observe() performs.
        m._current_task_id = 2
        m._task_step = 0
        m._lc_r_ema = 0.0

        # First step of task 2 with g_replay ≈ 0: λ should restart near 0.
        m._task_step = 1
        lam = m._curriculum_lambda(task_id=2, g_new_norm=1.0, g_replay_norm=1e-9)
        assert lam < 1e-6


# ---------------------------------------------------------------------------
# Full-buffer replay (G2 / G4 conditions)
# ---------------------------------------------------------------------------

class TestSampleAll:
    """ReservoirBuffer.sample_all() returns every stored sample exactly once
    — the basis for the exact-gradient G2/G4 conditions in §4.4.
    """

    def test_returns_every_sample_once(self):
        buf = ReservoirBuffer(total_budget=8)
        # Insert four unique samples, identified by the value in y.
        xs = [torch.full((4,), float(i)) for i in range(4)]
        ys = [torch.tensor(i) for i in range(4)]
        for i in range(4):
            buf.add(xs[i], ys[i], task_id=0)

        x_all, y_all, tids = buf.sample_all()
        assert x_all.shape == (4, 4)
        assert y_all.shape == (4,)
        # The y values are 0..3 (any order doesn't matter since we don't shuffle).
        assert sorted(y_all.tolist()) == [0, 1, 2, 3]
        # Each tensor's value matches its y label (we constructed x[i] = full(i)).
        for x_row, y in zip(x_all, y_all):
            assert torch.allclose(x_row, torch.full((4,), float(y.item())))
        assert tids == [0, 0, 0, 0]

    def test_size_grows_with_buffer(self):
        buf = ReservoirBuffer(total_budget=100)
        for i in range(50):
            buf.add(torch.zeros(4), torch.tensor(0), task_id=0)
        x_all, _, _ = buf.sample_all()
        assert x_all.size(0) == 50  # not total_budget, not 0 — exactly stored count

    def test_raises_when_empty(self):
        buf = ReservoirBuffer(total_budget=8)
        with pytest.raises(ValueError):
            buf.sample_all()


class TestReplayFullBuffer:
    """When ``method.replay_full_buffer=true``, ER's observe() step uses the
    *entire* buffer as the replay batch each step.  We verify this end-to-end
    by checking that the gradient computed during observe() matches what we
    would have computed by a forward+backward on the full-buffer cross-entropy
    directly, with no stochastic dependence on Python's random state.
    """

    def test_replay_loss_equals_full_buffer_loss(self):
        torch.manual_seed(0)
        # Buffer budget ≥ number of samples added → every sample is retained,
        # so sample_all() returns exactly the inserted xs / ys.
        m = _make_method(buffer_budget=32, replay_full_buffer=True)

        # Populate buffer with 16 samples.
        xs = torch.randn(16, 4)
        ys = torch.randint(0, 2, (16,))
        for i in range(16):
            m.buffer.add(xs[i], ys[i], task_id=0)

        # Reference replay loss: cross-entropy on the entire buffer.
        with torch.no_grad():
            ref_loss = nn.CrossEntropyLoss()(m.model(xs), ys).item()

        # Replay loss observed during observe() — replay-batch path must use
        # every sample exactly once, so it must equal the reference exactly.
        x_curr = torch.randn(8, 4)
        y_curr = torch.randint(0, 2, (8,))
        result = m.observe(x_curr, y_curr, task_id=1)
        assert "replay_loss" in result
        assert result["replay_loss"] == pytest.approx(ref_loss, abs=1e-6)

    def test_with_replacement_default_differs(self):
        """Sanity check: default ER (no replay_full_buffer) draws a small
        random batch and therefore produces a *different* replay loss than
        the full-buffer reference.  This guards against the flag being a
        silent no-op.
        """
        torch.manual_seed(0)
        # replay_full_buffer=False (default), replay_batch_size=None → 8.
        m = _make_method(buffer_budget=32, replay_full_buffer=False)

        xs = torch.randn(16, 4)
        ys = torch.randint(0, 2, (16,))
        for i in range(16):
            m.buffer.add(xs[i], ys[i], task_id=0)

        with torch.no_grad():
            ref_loss = nn.CrossEntropyLoss()(m.model(xs), ys).item()

        # Hit a non-default random state so the 8-sample draw is unlikely to
        # equal the full-buffer mean by coincidence.
        import random as _random
        _random.seed(7)

        x_curr = torch.randn(8, 4)
        y_curr = torch.randint(0, 2, (8,))
        result = m.observe(x_curr, y_curr, task_id=1)
        assert result["replay_loss"] != pytest.approx(ref_loss, abs=1e-6)
