"""Adaptive trust-region radius for CACL.

Implements Equations (16) and (17) from the CACL paper:

  (16) ρ = (ℓ_old − ℓ_new) / (−g_joint · d* · η)

       The numerator is the *actual* loss reduction; the denominator is the
       *predicted* reduction under a first-order linear model of the loss.
       ρ = 1 when the model is perfect, ρ > 1 when the step over-performs,
       ρ < 0 when the loss actually increases.

  (17) η_{t+1} = clip(η_t × m, η_min, η_max)

       where  m = 2    if ρ > expand_thresh  (step was good — trust more)
              m = 0.5  if ρ < contract_thresh (step was poor — trust less)
              m = 1    otherwise              (neither expand nor contract)

Usage inside ``CACL.observe()``::

    eta     = trust_region.get_step_size()
    params -= eta * d_star                    # apply unit-direction step
    rho     = trust_region.compute_ratio(loss_new, loss_old,
                                         directional_deriv, eta)
    trust_region.update_radius(rho)

References
----------
Equation numbering follows the CACL paper draft.  The ratio formula is the
standard trust-region acceptance criterion (Nocedal & Wright, §4.1).
"""

from __future__ import annotations

from typing import Optional


class TrustRegion:
    """Adaptive trust-region radius controller for CACL step-size adaptation.

    Args:
        initial_radius: Starting trust radius η₀.
        expand_thresh: Ratio threshold above which the radius is doubled.
            Corresponds to ``expand_threshold`` in ``configs/method/cacl.yaml``
            (default 0.75).
        contract_thresh: Ratio threshold below which the radius is halved.
            Corresponds to ``contract_threshold`` in config (default 0.25).
        min_r: Minimum allowed radius; prevents the radius from collapsing
            to zero when the step keeps failing.
        max_r: Maximum allowed radius; prevents unboundedly large steps.
    """

    def __init__(
        self,
        initial_radius: float,
        expand_thresh: float,
        contract_thresh: float,
        min_r: float,
        max_r: float,
    ) -> None:
        self.radius: float = float(initial_radius)
        self.expand_thresh: float = float(expand_thresh)
        self.contract_thresh: float = float(contract_thresh)
        self.min_r: float = float(min_r)
        self.max_r: float = float(max_r)

    # ------------------------------------------------------------------
    # Core methods
    # ------------------------------------------------------------------

    def compute_ratio(
        self,
        loss_new: float,
        loss_old: float,
        directional_deriv: float,
        step_size: float,
    ) -> Optional[float]:
        """Compute the trust-region acceptance ratio ρ (Equation 16).

        ``ρ = actual_reduction / predicted_reduction``

        where::

            actual_reduction    = ℓ_old − ℓ_new
            predicted_reduction = −(g_joint · d*) × η
                                = −directional_deriv × step_size

        *directional_deriv* is ``g_joint · d*``, which is negative for a
        descent direction (making ``predicted_reduction`` positive).

        Args:
            loss_new: Scalar loss evaluated **after** the step ``params -= η d*``.
            loss_old: Scalar loss evaluated **before** the step (at the current
                iterate; this is the joint loss used to compute ``d*``).
            directional_deriv: The inner product ``g_joint · d*``.  Typically
                negative for descent directions.
            step_size: The step size η used in ``params -= η d*``.  In CACL
                this is the value returned by :meth:`get_step_size` at that
                iteration.

        Returns:
            ρ as a Python float, or ``None`` if the predicted reduction is
            too small to be meaningful (``|directional_deriv × step_size| < 1e-8``).
            Pass ``None`` directly to :meth:`update_radius`, which will skip
            the update and hold the radius constant.

        Notes:
            When ``d*`` is nearly orthogonal to ``g_joint`` (possible when
            ``alpha_deg`` approaches 90°), ``directional_deriv`` ≈ 0 and the
            denominator is undefined.  The 1e-8 guard catches this case; log
            the event in production code.
        """
        actual_reduction: float = float(loss_old) - float(loss_new)
        predicted_reduction: float = -float(directional_deriv) * float(step_size)

        if abs(predicted_reduction) < 1e-8:
            # Denominator is effectively zero — skip trust update.
            return None

        return actual_reduction / predicted_reduction

    def update_radius(self, rho: Optional[float]) -> float:
        """Update the trust radius according to Equation (17).

        Args:
            rho: Trust ratio from :meth:`compute_ratio`, or ``None`` to skip
                the update and hold the radius constant.

        Returns:
            The new (possibly unchanged) trust radius, also stored as
            ``self.radius``.
        """
        if rho is None:
            # Degenerate denominator: hold radius constant.
            return self.radius

        if rho > self.expand_thresh:
            self.radius = min(self.radius * 2.0, self.max_r)
        elif rho < self.contract_thresh:
            self.radius = max(self.radius / 2.0, self.min_r)
        # Else: contract_thresh ≤ rho ≤ expand_thresh — radius unchanged.

        return self.radius

    def get_step_size(self) -> float:
        """Return the current trust radius as the step-size η.

        In CACL, the parameter update is ``params -= η · d*`` where ``d*``
        is a unit vector, so ``η`` equals the actual Euclidean step length.

        Returns:
            Current trust radius.
        """
        return self.radius

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def state_dict(self) -> dict:
        """Return a serialisable snapshot of the trust-region state.

        Captures both the mutable radius and the fixed hyperparameters so
        that a fully restored :class:`TrustRegion` is identical to the
        original.  Used by CACL's checkpoint-save routine (T6.2).

        Returns:
            Plain Python dict with keys: ``radius``, ``expand_thresh``,
            ``contract_thresh``, ``min_r``, ``max_r``.
        """
        return {
            "radius": self.radius,
            "expand_thresh": self.expand_thresh,
            "contract_thresh": self.contract_thresh,
            "min_r": self.min_r,
            "max_r": self.max_r,
        }

    def load_state_dict(self, d: dict) -> None:
        """Restore state from a previously saved :meth:`state_dict`.

        Args:
            d: Dictionary as returned by :meth:`state_dict`.  All five keys
               must be present; extra keys are ignored.
        """
        self.radius = float(d["radius"])
        self.expand_thresh = float(d["expand_thresh"])
        self.contract_thresh = float(d["contract_thresh"])
        self.min_r = float(d["min_r"])
        self.max_r = float(d["max_r"])
