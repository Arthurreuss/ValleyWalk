"""Tests for ReservoirBuffer.

Three checks as specified in T1.1:
  (a) After adding N > budget items, buffer holds exactly `budget` items.
  (b) Over many independent trials, each item appears with frequency ≈
      budget/N  (chi-squared goodness-of-fit, p > 0.01).
  (c) state_dict → load_state_dict round-trip preserves every stored item.
"""

import sys
import os

import numpy as np
import pytest
import scipy.stats
import torch

# Allow running from repo root without installation.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data.memory_buffer import ReservoirBuffer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_item(item_id: int, task_id: int = 0):
    """Return (x, y, task_id) tensors for a uniquely-identified sample.

    y encodes item_id so we can recover item identity from the buffer.
    """
    x = torch.tensor([float(item_id)])
    y = torch.tensor(item_id)          # scalar int tensor — unique identifier
    return x, y, task_id


def _items_in_buffer(buf: ReservoirBuffer) -> list:
    """Extract all item_ids currently stored (via state_dict, not internals)."""
    state = buf.state_dict()
    return [int(y.item()) for y in state["ys"]]


# ---------------------------------------------------------------------------
# (a) Exact budget size
# ---------------------------------------------------------------------------

class TestBufferSize:
    def test_never_exceeds_budget(self):
        """Buffer size never exceeds total_budget regardless of how many items
        are added."""
        budget = 50
        buf = ReservoirBuffer(total_budget=budget)

        for i in range(budget * 3):
            x, y, tid = _make_item(i)
            buf.add(x, y, tid)
            assert len(buf) <= budget

    def test_exactly_budget_after_overflow(self):
        """After adding N > budget items, buffer holds exactly budget items."""
        budget = 100
        N = budget * 5
        buf = ReservoirBuffer(total_budget=budget)

        for i in range(N):
            x, y, tid = _make_item(i)
            buf.add(x, y, tid)

        assert len(buf) == budget, (
            f"Expected {budget} items, got {len(buf)}"
        )

    def test_partial_fill(self):
        """Buffer size equals number of items added when N < budget."""
        budget = 200
        N = 50
        buf = ReservoirBuffer(total_budget=budget)

        for i in range(N):
            x, y, tid = _make_item(i)
            buf.add(x, y, tid)

        assert len(buf) == N


# ---------------------------------------------------------------------------
# (b) Uniformity via chi-squared test
# ---------------------------------------------------------------------------

class TestUniformity:
    def test_reservoir_is_uniform(self):
        """Each item from a stream of 10× budget has probability ≈ budget/N.

        Strategy: repeat the full-fill experiment n_trials times, count how
        often each item id appears across all resulting buffers, then run a
        chi-squared goodness-of-fit test against the uniform expected counts.
        """
        budget = 50
        N = budget * 10       # 500 distinct items
        n_trials = 400        # repetitions to accumulate stable counts

        counts = np.zeros(N, dtype=np.int64)

        for _ in range(n_trials):
            buf = ReservoirBuffer(total_budget=budget)
            for item_id in range(N):
                x, y, tid = _make_item(item_id)
                buf.add(x, y, tid)

            # Record which item_ids ended up in the buffer.
            for item_id in _items_in_buffer(buf):
                counts[item_id] += 1

        # Expected: each item appears in n_trials * budget/N trials on average.
        expected_per_item = n_trials * budget / N          # = 40.0
        expected = np.full(N, expected_per_item)

        stat, p_value = scipy.stats.chisquare(counts, f_exp=expected)
        assert p_value > 0.01, (
            f"Chi-squared test rejected uniformity (p={p_value:.4f}, "
            f"stat={stat:.2f}). Reservoir sampling may be biased."
        )


# ---------------------------------------------------------------------------
# (c) state_dict → load_state_dict round-trip
# ---------------------------------------------------------------------------

class TestSerialisation:
    def test_round_trip_preserves_items(self):
        """state_dict → load_state_dict restores every stored item exactly."""
        budget = 30
        buf_orig = ReservoirBuffer(total_budget=budget)

        for i in range(budget * 2):
            x = torch.randn(4)          # multi-element tensor
            y = torch.tensor(float(i))
            buf_orig.add(x, y, task_id=i % 3)

        snap = buf_orig.state_dict()

        buf_restored = ReservoirBuffer(total_budget=1)  # wrong budget; gets overwritten
        buf_restored.load_state_dict(snap)

        # Same length.
        assert len(buf_restored) == len(buf_orig)

        # Same total_budget and _n_seen.
        assert buf_restored.total_budget == buf_orig.total_budget
        assert buf_restored._n_seen == buf_orig._n_seen

        # Every stored tensor is bit-for-bit identical.
        orig_state = buf_orig.state_dict()
        rest_state = buf_restored.state_dict()

        for i in range(len(buf_orig)):
            assert torch.equal(orig_state["xs"][i], rest_state["xs"][i]), (
                f"x mismatch at slot {i}"
            )
            assert torch.equal(orig_state["ys"][i], rest_state["ys"][i]), (
                f"y mismatch at slot {i}"
            )
            assert orig_state["task_ids"][i] == rest_state["task_ids"][i], (
                f"task_id mismatch at slot {i}"
            )

    def test_round_trip_via_torch_save(self, tmp_path):
        """state_dict survives torch.save → torch.load."""
        budget = 20
        buf = ReservoirBuffer(total_budget=budget)
        for i in range(budget):
            buf.add(torch.randn(3), torch.tensor(float(i)), task_id=0)

        path = tmp_path / "buffer.pt"
        torch.save(buf.state_dict(), path)

        buf2 = ReservoirBuffer(total_budget=1)
        buf2.load_state_dict(torch.load(path, weights_only=False))

        assert len(buf2) == len(buf)
        assert buf2.total_budget == budget
