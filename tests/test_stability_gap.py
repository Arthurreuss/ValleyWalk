"""Tests for StabilityGapTracker (T3.2).

Synthetic scenario: a small linear classifier on a balanced two-class dataset.

Setup
-----
* Model: single Linear(4 → 2) layer with hand-crafted weights that yield
  100 % test accuracy.
* Dataset: 100 samples, 4 features.
  - Class 0: all features = -1
  - Class 1: all features = +1
* Weights: row 0 = [-1,-1,-1,-1], row 1 = [+1,+1,+1,+1]
  → class-0 score = +4 for negatives, class-1 score = +4 for positives → 100 %.

Synthetic degrade-then-recover scenario
-----------------------------------------
1. Tracker is created → pre-task accuracy = 1.0 is recorded.
2. Model weights are zeroed → accuracy drops to 0.5 (all logits = 0,
   argmax returns index 0, correct only for class-0 samples).
3. record() is called at global steps 0, 10, …, 190 (all within 200 steps).
4. Weights are restored → accuracy = 1.0.
5. record() is called at global step 200.

Expected results:
  max_drop()         ≈ 0.50   (1.0 − 0.5)
  recovery_steps()   == 200   (first step where acc ≥ 0.95 × 1.0)
"""

import os
import sys

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.eval.stability_gap import StabilityGapTracker


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_model_and_loader():
    """Return (model, loader, good_weights) with 100 % initial accuracy.

    * 100 balanced samples (50 per class).
    * Linear(4, 2), no bias.
    * Weights hand-crafted so every sample is classified correctly.
    """
    torch.manual_seed(0)
    n_per_class = 50
    x0 = -torch.ones(n_per_class, 4)
    x1 =  torch.ones(n_per_class, 4)
    X = torch.cat([x0, x1], dim=0)
    Y = torch.cat([
        torch.zeros(n_per_class, dtype=torch.long),
        torch.ones( n_per_class, dtype=torch.long),
    ])

    model = nn.Linear(4, 2, bias=False)
    good_weights = torch.tensor([[-1., -1., -1., -1.],
                                  [ 1.,  1.,  1.,  1.]])
    with torch.no_grad():
        model.weight.copy_(good_weights)

    loader = DataLoader(TensorDataset(X, Y), batch_size=100, shuffle=False)
    return model, loader, good_weights.clone()


# ---------------------------------------------------------------------------
# Test class
# ---------------------------------------------------------------------------

class TestStabilityGapTracker:
    """Verify StabilityGapTracker on a controlled degrade-then-recover scenario."""

    def setup_method(self):
        self.model, loader, self.good_weights = _make_model_and_loader()
        self.test_loaders = {0: loader}

        # Tracker constructed with good model → pre-task acc[0] = 1.0
        self.tracker = StabilityGapTracker(
            eval_freq_steps=10,
            model=self.model,
            test_loaders=self.test_loaders,
        )

    # -----------------------------------------------------------------------
    # Pre-task accuracy
    # -----------------------------------------------------------------------

    def test_pre_task_accuracy_is_one(self):
        """Baseline is captured correctly at construction (perfect model → 1.0)."""
        assert pytest.approx(self.tracker._pre_task_acc[0], abs=1e-6) == 1.0

    # -----------------------------------------------------------------------
    # Core scenario: degrade then recover
    # -----------------------------------------------------------------------

    def test_max_drop_degrade_then_recover(self):
        """max_drop() ≈ 0.50 after zeroing weights for the first 200 steps."""
        with torch.no_grad():
            self.model.weight.zero_()           # acc drops to 0.5
        for step in range(0, 200, 10):          # steps 0..190 (within window)
            self.tracker.record(step, self.test_loaders)

        with torch.no_grad():
            self.model.weight.copy_(self.good_weights)  # acc restored to 1.0
        self.tracker.record(200, self.test_loaders)      # step 200, still within window

        assert pytest.approx(self.tracker.max_drop(), abs=0.01) == 0.5

    def test_recovery_steps_degrade_then_recover(self):
        """recovery_steps() == 200 (first step where acc ≥ 0.95)."""
        with torch.no_grad():
            self.model.weight.zero_()
        for step in range(0, 200, 10):
            self.tracker.record(step, self.test_loaders)

        with torch.no_grad():
            self.model.weight.copy_(self.good_weights)
        self.tracker.record(200, self.test_loaders)

        assert self.tracker.recovery_steps() == 200

    def test_no_recovery_returns_none(self):
        """recovery_steps() is None when the model never regains 95 % accuracy."""
        with torch.no_grad():
            self.model.weight.zero_()           # acc = 0.5 throughout
        for step in range(0, 100, 10):
            self.tracker.record(step, self.test_loaders)
        # Model stays degraded → threshold never crossed
        assert self.tracker.recovery_steps() is None

    # -----------------------------------------------------------------------
    # 200-step window boundary for max_drop
    # -----------------------------------------------------------------------

    def test_max_drop_ignores_steps_beyond_200(self):
        """Accuracy drops occurring at step > 200 are excluded from max_drop()."""
        # Record with good model at step 0 (within window, drop = 0)
        self.tracker.record(0, self.test_loaders)

        # Degrade model and record at step 210 (outside window → must be ignored)
        with torch.no_grad():
            self.model.weight.zero_()
        self.tracker.record(210, self.test_loaders)

        # Only step 0 counts; acc = 1.0 → drop = 1.0 - 1.0 = 0.0
        assert pytest.approx(self.tracker.max_drop(), abs=1e-6) == 0.0

    def test_max_drop_step_200_is_included_in_window(self):
        """Step 200 is within the ≤ 200 window and contributes to max_drop()."""
        with torch.no_grad():
            self.model.weight.zero_()
        self.tracker.record(200, self.test_loaders)   # step_within_task = 200

        # acc = 0.5 at step 200 → drop = 0.5
        assert pytest.approx(self.tracker.max_drop(), abs=0.01) == 0.5

    # -----------------------------------------------------------------------
    # List interface
    # -----------------------------------------------------------------------

    def test_list_test_loaders_accepted(self):
        """test_loaders can be a list; index becomes the task id key."""
        loader = DataLoader(
            TensorDataset(
                -torch.ones(50, 4),
                torch.zeros(50, dtype=torch.long),
            ),
            batch_size=50,
        )
        tracker = StabilityGapTracker(
            eval_freq_steps=10,
            model=self.model,
            test_loaders=[loader],          # list, not dict
        )
        assert 0 in tracker._pre_task_acc
