"""Tests for ER's ``balanced_direction`` mode (D3'/D4' conditions).

Invariants of ``ER._balanced_direction_step``:

  (i)   step length equals vanilla's: ‖Δθ‖ = lr · ‖g_new + g_replay‖
  (ii)  direction is the equal-magnitude mix: Δθ ∝ −(ĝ_new + ĝ_rep)
  (iii) when ‖g_new‖ = ‖g_replay‖ the update reduces to vanilla ER exactly
  (iv)  task 0 (empty buffer) falls back to plain SGD
  (v)   an unknown mode string raises at construction
"""

from __future__ import annotations

import os
import sys

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data.memory_buffer import ReservoirBuffer
from src.methods.er import ER
from test_er_curriculum import _make_cfg


LR = 0.1


def _flat_params(model: nn.Module) -> torch.Tensor:
    return torch.cat([p.detach().flatten().clone() for p in model.parameters()])


def _flat_grad_of(model: nn.Module, loss: torch.Tensor) -> torch.Tensor:
    grads = torch.autograd.grad(loss, list(model.parameters()), retain_graph=False)
    return torch.cat([g.flatten() for g in grads])


def _build(seed: int = 0):
    """ER in balanced_direction mode with a deterministic replay batch.

    ``replay_full_buffer=True`` makes the method use ``buffer.sample_all()``,
    so the replay batch is the entire (small) buffer in storage order — no
    sampling randomness, and the expected update can be recomputed exactly.
    """
    torch.manual_seed(seed)
    model = nn.Linear(4, 2, bias=False)
    buffer = ReservoirBuffer(total_budget=6)
    cfg = _make_cfg(mode="balanced_direction", lr=LR, replay_full_buffer=True)
    method = ER(model, cfg, buffer)

    for i in range(6):
        buffer.add(torch.randn(4), torch.tensor(i % 2), task_id=0)

    x = torch.randn(8, 4)
    y = torch.randint(0, 2, (8,))
    return method, model, buffer, x, y


def _expected_gradients(method, model, buffer, x, y):
    """Recompute g_new / g_replay at the current parameters."""
    x_replay, y_replay, _ = buffer.sample_all()
    g_new = _flat_grad_of(model, method.loss_fn(model(x), y))
    g_rep = _flat_grad_of(model, method.loss_fn(model(x_replay), y_replay))
    return g_new, g_rep


class TestBalancedDirectionStep:
    def test_step_length_matches_vanilla(self):
        method, model, buffer, x, y = _build()
        g_new, g_rep = _expected_gradients(method, model, buffer, x, y)
        theta_before = _flat_params(model)

        method.observe(x, y, task_id=1)

        delta = _flat_params(model) - theta_before
        expected_len = LR * torch.linalg.norm(g_new + g_rep)
        assert torch.linalg.norm(delta).item() == pytest.approx(
            expected_len.item(), rel=1e-5
        )

    def test_direction_is_equal_magnitude_mix(self):
        method, model, buffer, x, y = _build()
        g_new, g_rep = _expected_gradients(method, model, buffer, x, y)
        theta_before = _flat_params(model)

        method.observe(x, y, task_id=1)

        delta = _flat_params(model) - theta_before
        d_dir = g_new / g_new.norm() + g_rep / g_rep.norm()
        cos = torch.dot(delta, -d_dir) / (delta.norm() * d_dir.norm())
        assert cos.item() == pytest.approx(1.0, abs=1e-5)

    def test_reduces_to_vanilla_when_norms_equal(self):
        # Analytic check on the update formula: when ‖g_new‖ = ‖g_rep‖,
        #   d = (ĝ_new + ĝ_rep) · ‖g_raw‖ / ‖ĝ_new + ĝ_rep‖ = g_new + g_rep.
        g_new = torch.tensor([3.0, 0.0])
        g_rep = torch.tensor([0.0, 3.0])  # same norm, orthogonal
        g_raw = g_new + g_rep
        d_dir = g_new / g_new.norm() + g_rep / g_rep.norm()
        d = d_dir * g_raw.norm() / d_dir.norm()
        assert torch.allclose(d, g_raw, atol=1e-6)

    def test_task0_falls_back_to_plain_sgd(self):
        torch.manual_seed(0)
        model = nn.Linear(4, 2, bias=False)
        buffer = ReservoirBuffer(total_budget=6)  # left empty
        cfg = _make_cfg(mode="balanced_direction", lr=LR)
        method = ER(model, cfg, buffer)

        x = torch.randn(8, 4)
        y = torch.randint(0, 2, (8,))
        g = _flat_grad_of(model, method.loss_fn(model(x), y))
        theta_before = _flat_params(model)

        method.observe(x, y, task_id=0)

        delta = _flat_params(model) - theta_before
        assert torch.allclose(delta, -LR * g, atol=1e-6)

    def test_unknown_mode_raises(self):
        model = nn.Linear(4, 2, bias=False)
        buffer = ReservoirBuffer(total_budget=6)
        cfg = _make_cfg(mode="bogus")
        with pytest.raises(ValueError, match="method.mode"):
            ER(model, cfg, buffer)
