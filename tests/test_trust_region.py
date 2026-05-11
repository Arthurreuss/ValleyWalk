"""Tests for TrustRegion (T4.4).

Verifies all five "Done when" conditions from the spec:

  (a) rho = 1.0 when loss change exactly equals the predicted reduction.
  (b) rho > expand_thresh (0.75) doubles the trust radius.
  (c) rho < contract_thresh (0.25) halves the trust radius.
  (d) Radius is clamped to [min_r, max_r] regardless of multiplier.
  (e) state_dict / load_state_dict round-trip preserves the radius exactly.

Additional tests:
  - rho = None when |denominator| < 1e-8 (skip-update sentinel).
  - Neutral band (0.25 ≤ rho ≤ 0.75) → radius unchanged.
  - Negative rho (loss increased) → radius contracts.
  - get_step_size returns the current radius.
  - update_radius(None) holds radius constant.
  - Sequential expand-then-contract cycle.
  - Full end-to-end compute_ratio → update_radius cycles.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.optim.trust_region import TrustRegion


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _default_tr(**kwargs) -> TrustRegion:
    """Return a TrustRegion with standard CACL defaults, optionally overriding."""
    defaults = dict(
        initial_radius=0.1,
        expand_thresh=0.75,
        contract_thresh=0.25,
        min_r=1e-4,
        max_r=1.0,
    )
    defaults.update(kwargs)
    return TrustRegion(**defaults)


# ---------------------------------------------------------------------------
# (a) rho = 1.0 when loss change equals the predicted reduction
# ---------------------------------------------------------------------------

class TestComputeRatio:
    """Verify compute_ratio() implements Equation (16) correctly."""

    def test_rho_equals_one_when_prediction_exact(self):
        """rho = 1.0 when actual reduction == predicted reduction."""
        tr = _default_tr()
        # actual = 1.0 - 0.9 = 0.1; predicted = -(-0.1) * 1.0 = 0.1
        rho = tr.compute_ratio(loss_new=0.9, loss_old=1.0,
                               directional_deriv=-0.1, step_size=1.0)
        assert abs(rho - 1.0) < 1e-10, f"Expected rho=1.0, got {rho}"

    def test_rho_quarter_when_actual_is_quarter_of_predicted(self):
        """rho = 0.25 when actual = (1/4) × predicted."""
        tr = _default_tr()
        # predicted = 0.2; actual = 0.05 → rho = 0.25
        rho = tr.compute_ratio(loss_new=0.95, loss_old=1.0,
                               directional_deriv=-0.2, step_size=1.0)
        assert abs(rho - 0.25) < 1e-10, f"Expected rho=0.25, got {rho}"

    def test_rho_greater_than_one_when_overshoot(self):
        """rho > 1 when actual improvement exceeds the linear prediction."""
        tr = _default_tr()
        # predicted = 0.1; actual = 0.2 → rho = 2.0
        rho = tr.compute_ratio(loss_new=0.8, loss_old=1.0,
                               directional_deriv=-0.1, step_size=1.0)
        assert abs(rho - 2.0) < 1e-10, f"Expected rho=2.0, got {rho}"

    def test_rho_negative_when_loss_increases(self):
        """rho < 0 when the step causes the loss to increase."""
        tr = _default_tr()
        # actual = 1.0 - 1.2 = -0.2; predicted = 0.1 → rho = -2.0
        rho = tr.compute_ratio(loss_new=1.2, loss_old=1.0,
                               directional_deriv=-0.1, step_size=1.0)
        assert rho < 0, f"Expected negative rho for loss increase, got {rho}"
        assert abs(rho - (-2.0)) < 1e-10

    def test_rho_zero_when_actual_reduction_zero(self):
        """rho = 0 when the step leaves the loss unchanged."""
        tr = _default_tr()
        rho = tr.compute_ratio(loss_new=1.0, loss_old=1.0,
                               directional_deriv=-0.1, step_size=1.0)
        assert abs(rho - 0.0) < 1e-10

    def test_rho_scales_correctly_with_step_size(self):
        """compute_ratio accounts for step_size in the denominator."""
        tr = _default_tr()
        # directional_deriv=-1.0, step_size=0.1 → predicted=0.1; actual=0.1 → rho=1.0
        rho = tr.compute_ratio(loss_new=0.9, loss_old=1.0,
                               directional_deriv=-1.0, step_size=0.1)
        assert abs(rho - 1.0) < 1e-10

    def test_returns_none_when_directional_deriv_tiny(self):
        """Returns None when |directional_deriv * step_size| < 1e-8."""
        tr = _default_tr()
        # predicted = -(-1e-10) * 1.0 = 1e-10 < 1e-8 → None
        rho = tr.compute_ratio(loss_new=0.9, loss_old=1.0,
                               directional_deriv=-1e-10, step_size=1.0)
        assert rho is None

    def test_returns_none_when_step_size_tiny(self):
        """Returns None when step_size is nearly zero (degenerate step)."""
        tr = _default_tr()
        rho = tr.compute_ratio(loss_new=0.5, loss_old=1.0,
                               directional_deriv=-1.0, step_size=1e-10)
        assert rho is None

    def test_returns_none_when_directional_deriv_zero(self):
        """Returns None when directional_deriv is exactly zero."""
        tr = _default_tr()
        rho = tr.compute_ratio(loss_new=0.9, loss_old=1.0,
                               directional_deriv=0.0, step_size=1.0)
        assert rho is None


# ---------------------------------------------------------------------------
# (b) rho > expand_thresh → radius doubles
# ---------------------------------------------------------------------------

class TestUpdateRadiusExpand:
    """rho > expand_thresh (0.75) must double the radius."""

    def test_expand_on_high_rho(self):
        """rho = 0.8 > 0.75 → radius doubles from 0.1 to 0.2."""
        tr = _default_tr(initial_radius=0.1)
        new_r = tr.update_radius(0.8)
        assert abs(new_r - 0.2) < 1e-12, f"Expected 0.2, got {new_r}"
        assert abs(tr.radius - 0.2) < 1e-12

    def test_expand_on_rho_one(self):
        """rho = 1.0 (perfect step) → doubles radius."""
        tr = _default_tr(initial_radius=0.1)
        tr.update_radius(1.0)
        assert abs(tr.radius - 0.2) < 1e-12

    def test_expand_on_rho_greater_than_one(self):
        """rho > 1 (overshoot) still doubles (same rule applies)."""
        tr = _default_tr(initial_radius=0.1)
        tr.update_radius(2.0)
        assert abs(tr.radius - 0.2) < 1e-12

    def test_no_expand_at_exact_threshold(self):
        """rho = expand_thresh exactly: strict inequality → no expansion."""
        tr = _default_tr(initial_radius=0.1, expand_thresh=0.75)
        tr.update_radius(0.75)   # not strictly > 0.75
        assert abs(tr.radius - 0.1) < 1e-12

    def test_update_radius_returns_new_radius(self):
        """update_radius returns the updated value."""
        tr = _default_tr(initial_radius=0.1)
        returned = tr.update_radius(1.0)
        assert abs(returned - tr.radius) < 1e-12


# ---------------------------------------------------------------------------
# (c) rho < contract_thresh → radius halves
# ---------------------------------------------------------------------------

class TestUpdateRadiusContract:
    """rho < contract_thresh (0.25) must halve the radius."""

    def test_contract_on_low_rho(self):
        """rho = 0.1 < 0.25 → radius halves from 0.1 to 0.05."""
        tr = _default_tr(initial_radius=0.1)
        new_r = tr.update_radius(0.1)
        assert abs(new_r - 0.05) < 1e-12, f"Expected 0.05, got {new_r}"
        assert abs(tr.radius - 0.05) < 1e-12

    def test_contract_on_negative_rho(self):
        """rho < 0 (loss increased) → halves radius."""
        tr = _default_tr(initial_radius=0.2)
        tr.update_radius(-1.5)
        assert abs(tr.radius - 0.1) < 1e-12

    def test_contract_on_rho_zero(self):
        """rho = 0 (no improvement) → halves radius."""
        tr = _default_tr(initial_radius=0.1)
        tr.update_radius(0.0)
        assert abs(tr.radius - 0.05) < 1e-12

    def test_no_contract_at_exact_threshold(self):
        """rho = contract_thresh exactly: strict inequality → no contraction."""
        tr = _default_tr(initial_radius=0.1, contract_thresh=0.25)
        tr.update_radius(0.25)   # not strictly < 0.25
        assert abs(tr.radius - 0.1) < 1e-12

    @pytest.mark.parametrize("rho", [0.25, 0.4, 0.5, 0.6, 0.75])
    def test_neutral_band_radius_unchanged(self, rho: float):
        """0.25 ≤ rho ≤ 0.75 → radius is unchanged."""
        tr = _default_tr(initial_radius=0.1)
        tr.update_radius(rho)
        assert abs(tr.radius - 0.1) < 1e-12, (
            f"rho={rho}: expected radius=0.1, got {tr.radius}"
        )


# ---------------------------------------------------------------------------
# (d) Radius is clamped to [min_r, max_r]
# ---------------------------------------------------------------------------

class TestRadiusClamping:
    """Radius must never exceed max_r or fall below min_r."""

    def test_expansion_capped_at_max_radius(self):
        """Expanding near max_r stops at max_r."""
        tr = _default_tr(initial_radius=0.9, max_r=1.0)
        tr.update_radius(1.0)   # would double to 1.8, capped at 1.0
        assert abs(tr.radius - 1.0) < 1e-12

    def test_expansion_at_max_radius_stays_at_max(self):
        """Expanding when already at max_r stays at max_r."""
        tr = _default_tr(initial_radius=1.0, max_r=1.0)
        tr.update_radius(1.0)
        assert abs(tr.radius - 1.0) < 1e-12

    def test_contraction_floored_at_min_radius(self):
        """Contracting near min_r stops at min_r."""
        tr = _default_tr(initial_radius=1.5e-4, min_r=1e-4)
        tr.update_radius(0.0)   # would halve to 7.5e-5, floored at 1e-4
        assert tr.radius >= 1e-4
        assert abs(tr.radius - 1e-4) < 1e-15

    def test_contraction_at_min_radius_stays_at_min(self):
        """Contracting when already at min_r stays at min_r."""
        tr = _default_tr(initial_radius=1e-4, min_r=1e-4)
        tr.update_radius(0.0)
        assert abs(tr.radius - 1e-4) < 1e-15

    def test_repeated_expansions_bounded_by_max(self):
        """Many consecutive expand updates never exceed max_r."""
        tr = _default_tr(initial_radius=0.1, max_r=1.0)
        for _ in range(20):
            tr.update_radius(1.0)
        assert tr.radius <= 1.0

    def test_repeated_contractions_bounded_by_min(self):
        """Many consecutive contract updates never go below min_r."""
        tr = _default_tr(initial_radius=0.5, min_r=1e-4)
        for _ in range(50):
            tr.update_radius(0.0)
        assert tr.radius >= 1e-4


# ---------------------------------------------------------------------------
# (e) state_dict / load_state_dict round-trip
# ---------------------------------------------------------------------------

class TestStateDictRoundTrip:
    """state_dict() → mutate → load_state_dict() must restore the original state."""

    def test_round_trip_preserves_radius(self):
        """Saving and reloading state_dict restores radius exactly."""
        tr = _default_tr(initial_radius=0.3)
        saved = tr.state_dict()
        tr.update_radius(1.0)   # doubles radius to 0.6
        assert abs(tr.radius - 0.6) < 1e-12
        tr.load_state_dict(saved)
        assert abs(tr.radius - 0.3) < 1e-12, (
            f"Expected radius=0.3 after load_state_dict, got {tr.radius}"
        )

    def test_state_dict_contains_required_keys(self):
        """state_dict includes all five required keys."""
        sd = _default_tr().state_dict()
        for key in ("radius", "expand_thresh", "contract_thresh", "min_r", "max_r"):
            assert key in sd, f"Key '{key}' missing from state_dict"

    def test_load_state_dict_restores_all_hyperparameters(self):
        """load_state_dict restores every field, not just radius."""
        tr1 = TrustRegion(
            initial_radius=0.05,
            expand_thresh=0.8,
            contract_thresh=0.3,
            min_r=1e-3,
            max_r=0.5,
        )
        sd = tr1.state_dict()
        tr2 = _default_tr()   # different defaults
        tr2.load_state_dict(sd)
        assert abs(tr2.radius - 0.05) < 1e-12
        assert abs(tr2.expand_thresh - 0.8) < 1e-12
        assert abs(tr2.contract_thresh - 0.3) < 1e-12
        assert abs(tr2.min_r - 1e-3) < 1e-12
        assert abs(tr2.max_r - 0.5) < 1e-12

    def test_state_dict_is_independent_copy(self):
        """Mutating the returned dict does not affect the TrustRegion."""
        tr = _default_tr(initial_radius=0.1)
        sd = tr.state_dict()
        sd["radius"] = 999.0
        assert abs(tr.radius - 0.1) < 1e-12


# ---------------------------------------------------------------------------
# Additional: get_step_size, None sentinel, integration cycles
# ---------------------------------------------------------------------------

class TestGetStepSizeAndIntegration:
    """Miscellaneous get_step_size and end-to-end cycle tests."""

    def test_get_step_size_returns_initial_radius(self):
        """get_step_size() returns the current radius."""
        tr = _default_tr(initial_radius=0.1)
        assert abs(tr.get_step_size() - 0.1) < 1e-12

    def test_get_step_size_reflects_updates(self):
        """get_step_size() returns the updated value after update_radius."""
        tr = _default_tr(initial_radius=0.1)
        tr.update_radius(1.0)   # expands to 0.2
        assert abs(tr.get_step_size() - 0.2) < 1e-12

    def test_update_radius_none_holds_radius_constant(self):
        """update_radius(None) skips the update; radius is unchanged."""
        tr = _default_tr(initial_radius=0.1)
        new_r = tr.update_radius(None)
        assert abs(new_r - 0.1) < 1e-12
        assert abs(tr.radius - 0.1) < 1e-12

    def test_sequential_expand_then_contract(self):
        """Expand to 0.2, then contract back to 0.1."""
        tr = _default_tr(initial_radius=0.1)
        tr.update_radius(1.0)    # 0.1 → 0.2
        tr.update_radius(0.0)    # 0.2 → 0.1
        assert abs(tr.radius - 0.1) < 1e-12

    def test_end_to_end_expand_cycle(self):
        """Full cycle: compute_ratio gives rho=1.0 → update_radius doubles."""
        tr = _default_tr(initial_radius=0.1)
        # predicted = -(-0.1) * 1.0 = 0.1; actual = 1.0 - 0.9 = 0.1 → rho = 1.0
        rho = tr.compute_ratio(loss_new=0.9, loss_old=1.0,
                               directional_deriv=-0.1, step_size=1.0)
        assert abs(rho - 1.0) < 1e-10
        new_r = tr.update_radius(rho)
        assert abs(new_r - 0.2) < 1e-12

    def test_end_to_end_contract_cycle(self):
        """Full cycle: no improvement → rho=0.0 → update_radius halves."""
        tr = _default_tr(initial_radius=0.1)
        rho = tr.compute_ratio(loss_new=1.0, loss_old=1.0,
                               directional_deriv=-0.1, step_size=1.0)
        assert abs(rho - 0.0) < 1e-10
        new_r = tr.update_radius(rho)
        assert abs(new_r - 0.05) < 1e-12

    def test_end_to_end_none_sentinel_cycle(self):
        """Degenerate directional_deriv → None → radius unchanged."""
        tr = _default_tr(initial_radius=0.1)
        rho = tr.compute_ratio(loss_new=0.5, loss_old=1.0,
                               directional_deriv=0.0, step_size=1.0)
        assert rho is None
        new_r = tr.update_radius(rho)
        assert abs(new_r - 0.1) < 1e-12
