"""
Temperature derivation from Beta posterior uncertainty.
"""

import math
from bandit.schemas import ArmState, BanditState


class TemperatureDeriver:
    """Derives per-arm temperatures from Beta posterior variance."""

    def compute(self, arm_state: ArmState, T_base: float, min_temperature: float) -> float:
        """Compute temperature for a single arm from its posterior uncertainty.

        Formula: sigma = sqrt(alpha * beta / ((alpha + beta)^2 * (alpha + beta + 1)))
                 T_arm = max(min_temperature, T_base * sigma)
        """
        a = arm_state.alpha
        b = arm_state.beta
        ab = a + b
        sigma = math.sqrt(a * b / (ab * ab * (ab + 1)))
        T_arm = T_base * sigma
        return max(min_temperature, T_arm)

    def compute_all(self, state: BanditState) -> dict:
        """Compute temperatures for all arms in the bandit state.

        Returns dict mapping arm_id -> temperature.
        """
        result = {}
        for arm_id, arm in state.arms.items():
            if not isinstance(arm, ArmState):
                continue
            result[arm_id] = self.compute(arm, state.T_base, state.min_temperature)
        return result

    def effective_temperature(
        self,
        T_arm: float,
        constraint_density: float,
        surrogate_modulation_factor: float,
    ) -> float:
        """Compute effective temperature with constraint and surrogate modulation.

        T_effective = T_arm * (1 + constraint_density) * surrogate_modulation_factor
        """
        return T_arm * (1 + constraint_density) * surrogate_modulation_factor
