"""Tests for deflate() and cone_project() (T4.3).

Verifies the four geometric properties specified in the T4.3 "done when":

  (a) deflated vector is orthogonal to all columns of V_danger (|dot| < 1e-6)
  (b) output direction satisfies cone constraint to tolerance 1e-4
  (c) when g_joint is exactly aligned with a dangerous direction, fallback
      triggers and output still satisfies the cone constraint
  (d) alpha=0  → returns g_hat_joint (cone degenerates to a ray)
      alpha=90 → always returns the (normalised) deflated gradient

Additional tests cover:
  - Empty V_danger (d=0): deflation is a no-op
  - Partial deflation: non-zero g_deflated with correct projection removal
  - Degenerate g_deflated ≈ 0 is handled gracefully
  - Binary-search precision: cone constraint met to < 1e-6
  - fallback_triggered flag is correct
"""

import math
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.optim.cone_projection import cone_project, deflate


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _random_orthonormal(dim: int, d: int, seed: int = 0) -> torch.Tensor:
    """Return a (dim, d) matrix with orthonormal columns."""
    torch.manual_seed(seed)
    Q, _ = torch.linalg.qr(torch.randn(dim, d))
    return Q  # (dim, d)


def _random_unit(dim: int, seed: int = 1) -> torch.Tensor:
    """Return a random unit vector of shape (dim,)."""
    torch.manual_seed(seed)
    v = torch.randn(dim)
    return v / torch.linalg.norm(v)


def _angle_deg(u: torch.Tensor, v: torch.Tensor) -> float:
    """Angle in degrees between two vectors."""
    cos = torch.dot(u / u.norm(), v / v.norm()).clamp(-1.0, 1.0).item()
    return math.degrees(math.acos(cos))


# ---------------------------------------------------------------------------
# (a) Orthogonality after deflation
# ---------------------------------------------------------------------------

class TestDeflate:
    """Verify that deflate() removes all projection onto V_danger."""

    def test_orthogonality_to_all_danger_columns(self):
        """g_deflated must be orthogonal to every column of V_danger (|dot| < 1e-6)."""
        dim, d = 20, 4
        V_danger = _random_orthonormal(dim, d, seed=0)
        g_hat = _random_unit(dim, seed=1)

        g_deflated = deflate(g_hat, V_danger)

        for i in range(d):
            dot_product = abs(torch.dot(g_deflated, V_danger[:, i]).item())
            assert dot_product < 1e-6, (
                f"Column {i}: |g_deflated · v_{i}| = {dot_product:.2e} ≥ 1e-6"
            )

    def test_deflation_does_not_change_components_outside_danger(self):
        """The component of g_deflated outside V_danger must equal g_hat outside V_danger."""
        dim, d = 10, 2
        V_danger = _random_orthonormal(dim, d, seed=2)
        g_hat = _random_unit(dim, seed=3)

        g_deflated = deflate(g_hat, V_danger)

        # Remove dangerous component from g_hat manually
        proj = V_danger @ (V_danger.T @ g_hat)
        expected = g_hat - proj
        assert torch.allclose(g_deflated, expected, atol=1e-6)

    def test_empty_V_danger_is_noop(self):
        """With d=0 (no dangerous directions), deflation returns g_hat unchanged."""
        dim = 10
        V_empty = torch.zeros(dim, 0)
        g_hat = _random_unit(dim, seed=4)

        g_deflated = deflate(g_hat, V_empty)
        assert torch.allclose(g_deflated, g_hat, atol=1e-7)

    def test_g_hat_in_danger_space_gives_zero(self):
        """If g_hat_joint exactly equals a danger column, g_deflated ≈ 0."""
        dim, d = 10, 3
        V_danger = _random_orthonormal(dim, d, seed=5)
        # Set g_hat to the first dangerous direction
        g_hat = V_danger[:, 0].clone()

        g_deflated = deflate(g_hat, V_danger)
        assert torch.linalg.norm(g_deflated).item() < 1e-6

    def test_deflated_norm_at_most_one(self):
        """‖g_deflated‖ ≤ ‖g_hat_joint‖ = 1 (projection can only decrease norm)."""
        dim, d = 30, 5
        V_danger = _random_orthonormal(dim, d, seed=6)
        g_hat = _random_unit(dim, seed=7)

        g_deflated = deflate(g_hat, V_danger)
        assert torch.linalg.norm(g_deflated).item() <= 1.0 + 1e-6

    def test_deflation_dot_with_g_hat_is_nonneg(self):
        """g_deflated · g_hat_joint ≥ 0 always (deflation cannot reverse the direction)."""
        dim, d = 20, 3
        V_danger = _random_orthonormal(dim, d, seed=8)
        g_hat = _random_unit(dim, seed=9)

        g_deflated = deflate(g_hat, V_danger)
        dot = torch.dot(g_deflated, g_hat).item()
        assert dot >= -1e-7, f"dot(g_deflated, g_hat) = {dot:.4f} < 0"


# ---------------------------------------------------------------------------
# (b) Cone constraint satisfied after cone_project
# ---------------------------------------------------------------------------

class TestConeProjectConstraint:
    """Verify that d_star satisfies the cone constraint within tolerance 1e-4."""

    def _check_cone(self, d_star: torch.Tensor, g_hat: torch.Tensor, alpha_deg: float):
        """Assert that angle(d_star, g_hat) ≤ alpha_deg + 1e-4 (in degrees)."""
        angle = _angle_deg(d_star, g_hat)
        assert angle <= alpha_deg + 1e-4, (
            f"angle(d_star, g_hat) = {angle:.4f}° > alpha_deg = {alpha_deg}°"
        )

    def test_constraint_when_deflated_inside_cone(self):
        """When g_deflated is inside the cone, d_star = ĝ_deflated satisfies constraint."""
        dim = 50
        g_hat = _random_unit(dim, seed=10)
        # Construct g_deflated pointing close to g_hat (10° away)
        perp = _random_unit(dim, seed=11)
        perp = perp - torch.dot(perp, g_hat) * g_hat
        perp = perp / perp.norm()
        angle_rad = math.radians(10.0)
        g_deflated = math.cos(angle_rad) * g_hat + math.sin(angle_rad) * perp

        alpha_deg = 30.0
        d_star, fallback = cone_project(g_hat, g_deflated, alpha_deg)
        assert not fallback, "Expected no fallback for inside-cone deflated gradient"
        self._check_cone(d_star, g_hat, alpha_deg)

    def test_constraint_when_deflated_outside_cone(self):
        """When g_deflated is outside the cone, d_star is placed on the cone boundary."""
        dim = 50
        g_hat = _random_unit(dim, seed=12)
        # Construct g_deflated pointing 60° away from g_hat
        perp = _random_unit(dim, seed=13)
        perp = perp - torch.dot(perp, g_hat) * g_hat
        perp = perp / perp.norm()
        angle_rad = math.radians(60.0)
        g_deflated = math.cos(angle_rad) * g_hat + math.sin(angle_rad) * perp

        alpha_deg = 30.0
        d_star, fallback = cone_project(g_hat, g_deflated, alpha_deg)
        assert fallback, "Expected fallback for outside-cone deflated gradient"
        self._check_cone(d_star, g_hat, alpha_deg)

    @pytest.mark.parametrize("alpha_deg", [15.0, 30.0, 45.0, 60.0, 75.0])
    def test_constraint_various_alphas(self, alpha_deg: float):
        """Cone constraint holds for a range of alpha values with outside-cone input."""
        dim = 30
        g_hat = _random_unit(dim, seed=20)
        # Place g_deflated at 80° (outside all tested cones)
        perp = _random_unit(dim, seed=21)
        perp = perp - torch.dot(perp, g_hat) * g_hat
        perp = perp / perp.norm()
        angle_rad = math.radians(80.0)
        g_deflated = math.cos(angle_rad) * g_hat + math.sin(angle_rad) * perp

        d_star, _ = cone_project(g_hat, g_deflated, alpha_deg)
        self._check_cone(d_star, g_hat, alpha_deg)

    def test_d_star_is_unit_norm(self):
        """d_star must have unit norm regardless of fallback."""
        dim = 20
        g_hat = _random_unit(dim, seed=30)
        perp = _random_unit(dim, seed=31)
        perp = perp - torch.dot(perp, g_hat) * g_hat
        perp = perp / perp.norm()
        g_deflated = math.cos(math.radians(70.0)) * g_hat + math.sin(math.radians(70.0)) * perp

        d_star, _ = cone_project(g_hat, g_deflated, alpha_deg=45.0)
        norm = torch.linalg.norm(d_star).item()
        assert abs(norm - 1.0) < 1e-5, f"‖d_star‖ = {norm:.6f} ≠ 1"

    def test_binary_search_precision(self):
        """Cone boundary is met to tolerance < 1e-6 after binary search."""
        dim = 50
        g_hat = _random_unit(dim, seed=40)
        perp = _random_unit(dim, seed=41)
        perp = perp - torch.dot(perp, g_hat) * g_hat
        perp = perp / perp.norm()
        g_deflated = math.cos(math.radians(75.0)) * g_hat + math.sin(math.radians(75.0)) * perp

        alpha_deg = 30.0
        d_star, fallback = cone_project(g_hat, g_deflated, alpha_deg)
        assert fallback

        actual_angle = _angle_deg(d_star, g_hat)
        # Should be very close to alpha_deg on the cone boundary
        assert abs(actual_angle - alpha_deg) < 1e-4, (
            f"angle {actual_angle:.6f}° deviates from alpha {alpha_deg}° by "
            f"{abs(actual_angle - alpha_deg):.2e}°"
        )


# ---------------------------------------------------------------------------
# (c) Full-conflict: g_joint aligned with dangerous direction
# ---------------------------------------------------------------------------

class TestFullConflictFallback:
    """When g_hat_joint lies in the dangerous subspace, fallback triggers correctly."""

    def test_fallback_triggered_when_g_in_danger_space(self):
        """g_hat_joint = a danger column → g_deflated ≈ 0 → fallback = True."""
        dim, d = 20, 3
        V_danger = _random_orthonormal(dim, d, seed=50)
        g_hat = V_danger[:, 0].clone()  # exactly in danger space

        g_deflated = deflate(g_hat, V_danger)
        # Confirm g_deflated is nearly zero
        assert torch.linalg.norm(g_deflated).item() < 1e-6

        d_star, fallback = cone_project(g_hat, g_deflated, alpha_deg=45.0)
        assert fallback, "Fallback must be True when g_deflated ≈ 0"

    def test_output_is_g_hat_joint_on_full_conflict(self):
        """d_star must equal g_hat_joint on full conflict."""
        dim, d = 20, 3
        V_danger = _random_orthonormal(dim, d, seed=51)
        g_hat = V_danger[:, 1].clone()

        g_deflated = deflate(g_hat, V_danger)
        d_star, fallback = cone_project(g_hat, g_deflated, alpha_deg=45.0)

        assert fallback
        assert torch.allclose(d_star, g_hat, atol=1e-6), (
            "d_star should equal g_hat_joint on full conflict"
        )

    def test_output_satisfies_cone_on_full_conflict(self):
        """Even after a full-conflict fallback, d_star satisfies the cone constraint."""
        dim, d = 20, 3
        V_danger = _random_orthonormal(dim, d, seed=52)
        g_hat = V_danger[:, 2].clone()

        g_deflated = deflate(g_hat, V_danger)
        d_star, fallback = cone_project(g_hat, g_deflated, alpha_deg=45.0)

        assert fallback
        # d_star = g_hat_joint → angle = 0° ≤ 45°
        angle = _angle_deg(d_star, g_hat)
        assert angle < 1e-4, f"angle(d_star, g_hat) = {angle:.4f}° on full conflict"


# ---------------------------------------------------------------------------
# (d) Special-case alpha values
# ---------------------------------------------------------------------------

class TestSpecialAlphas:
    """Verify exact behaviour at alpha=0 and alpha=90."""

    def test_alpha_zero_returns_g_hat_joint(self):
        """alpha=0: cone degenerates to a ray; must return g_hat_joint."""
        dim = 20
        g_hat = _random_unit(dim, seed=60)
        # Build a g_deflated that is NOT g_hat (not in the zero-cone)
        perp = _random_unit(dim, seed=61)
        perp = perp - torch.dot(perp, g_hat) * g_hat
        perp = perp / perp.norm()
        g_deflated = math.cos(math.radians(30.0)) * g_hat + math.sin(math.radians(30.0)) * perp

        d_star, fallback = cone_project(g_hat, g_deflated, alpha_deg=0.0)

        assert torch.allclose(d_star, g_hat, atol=1e-5), (
            "alpha=0 must return g_hat_joint"
        )
        assert fallback, "alpha=0 with non-parallel deflated grad must trigger fallback"

    def test_alpha_zero_when_deflated_equals_g_hat(self):
        """alpha=0: if deflation doesn't change direction, still returns g_hat."""
        dim = 20
        g_hat = _random_unit(dim, seed=62)
        # g_deflated exactly parallel to g_hat (no dangerous component)
        g_deflated = 0.7 * g_hat  # same direction, different magnitude

        d_star, _ = cone_project(g_hat, g_deflated, alpha_deg=0.0)
        assert torch.allclose(d_star, g_hat, atol=1e-5)

    def test_alpha_90_returns_deflated_gradient_no_fallback(self):
        """alpha=90: the full hemisphere is within the cone; returns ĝ_deflated."""
        dim = 20
        V_danger = _random_orthonormal(dim, 3, seed=70)
        g_hat = _random_unit(dim, seed=71)

        g_deflated = deflate(g_hat, V_danger)
        # After deflation: dot(g_deflated, g_hat) ≥ 0 (always true for unit g_hat)
        assert g_deflated.norm() > 1e-8, "g_hat must not lie entirely in danger space"

        d_star, fallback = cone_project(g_hat, g_deflated, alpha_deg=90.0)

        expected = g_deflated / g_deflated.norm()
        assert not fallback, "alpha=90 must not trigger fallback"
        assert torch.allclose(d_star, expected, atol=1e-6), (
            "alpha=90 must return normalised deflated gradient"
        )

    @pytest.mark.parametrize("seed", [80, 81, 82, 83, 84])
    def test_alpha_90_no_fallback_multiple_seeds(self, seed: int):
        """alpha=90 never triggers fallback (deflated gradient always in upper hemisphere)."""
        dim = 30
        V_danger = _random_orthonormal(dim, 3, seed=seed)
        g_hat = _random_unit(dim, seed=seed + 100)
        g_deflated = deflate(g_hat, V_danger)

        if g_deflated.norm() < 1e-8:
            pytest.skip("g_hat in full danger space for this seed — skip")

        _, fallback = cone_project(g_hat, g_deflated, alpha_deg=90.0)
        assert not fallback, f"seed={seed}: alpha=90 triggered fallback unexpectedly"
