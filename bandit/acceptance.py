"""
Simulated annealing acceptance engine.
"""

import math
from bandit.schemas import (
    ArmState, BanditState, AcceptanceDecision,
)
from bandit.temperature import TemperatureDeriver
from bandit.surrogate_bridge import SurrogateModulationCalculator


class AnnealingAcceptanceEngine:
    """Decides whether to accept a candidate solution using simulated annealing."""

    def __init__(self):
        self._temp_deriver = TemperatureDeriver()
        self._surrogate_calc = SurrogateModulationCalculator()

    def decide(
        self,
        delta: float,
        arm_state: ArmState,
        state: BanditState,
        surrogate_predicted_delta: float = None,
        rng=None,
    ) -> AcceptanceDecision:
        """Make an acceptance decision.

        Args:
            delta: Performance change (negative = improvement, positive = regression).
            arm_state: The arm's current state.
            state: Full bandit state.
            surrogate_predicted_delta: Surrogate model's prediction (optional).
            rng: Random number generator with .random() method.

        Returns:
            AcceptanceDecision with full details.
        """
        surrogate_modulation = self._surrogate_calc.compute_modulation(
            surrogate_predicted_delta, delta
        )

        T_arm = self._temp_deriver.compute(arm_state, state.T_base, state.min_temperature)
        T_effective = self._temp_deriver.effective_temperature(
            T_arm, arm_state.constraint_density, surrogate_modulation
        )

        # Improvement: always accept
        if delta <= 0:
            return AcceptanceDecision(
                accepted=True,
                accepted_by="improvement",
                probability=1.0,
                random_draw=None,
                T_effective=T_effective,
                delta=delta,
                surrogate_predicted_delta=surrogate_predicted_delta,
                surrogate_modulation_factor=surrogate_modulation,
            )

        # Regression: stochastic acceptance via Boltzmann criterion
        if T_effective > 0:
            p = math.exp(-delta / T_effective)
        else:
            p = 0.0

        u = rng.random() if rng is not None else 1.0  # default reject without rng

        accepted = u < p

        return AcceptanceDecision(
            accepted=accepted,
            accepted_by="annealing" if accepted else "rejected",
            probability=p,
            random_draw=u,
            T_effective=T_effective,
            delta=delta,
            surrogate_predicted_delta=surrogate_predicted_delta,
            surrogate_modulation_factor=surrogate_modulation,
        )
