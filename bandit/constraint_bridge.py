"""
Constraint density calculation for per-arm temperature modulation.
"""

from bandit.schemas import ArmState, BanditState


class ConstraintDensityCalculator:
    """Computes per-arm constraint density from a constraint list and taxonomy."""

    def compute_densities(
        self,
        constraint_list: list,
        taxonomy: dict,
    ) -> dict:
        """Compute per-arm constraint density.

        Each constraint should have an 'arm_id' or 'category' field.
        Density = (arm constraint count) / total constraints, normalized so sum ~= 1.0.

        Args:
            constraint_list: List of constraint dicts with 'arm_id' or 'category' key.
            taxonomy: Dict mapping arm_id -> ArmDefinition or similar.

        Returns:
            Dict mapping arm_id -> constraint density (float).
        """
        if not constraint_list:
            return {arm_id: 0.0 for arm_id in taxonomy}

        counts = {arm_id: 0 for arm_id in taxonomy}
        total = 0

        for constraint in constraint_list:
            arm_id = constraint.get("arm_id", constraint.get("category", ""))
            if arm_id in counts:
                counts[arm_id] += 1
                total += 1

        if total == 0:
            return {arm_id: 0.0 for arm_id in taxonomy}

        densities = {}
        for arm_id in taxonomy:
            densities[arm_id] = counts[arm_id] / total

        return densities

    def effective_min_temperature(
        self,
        density: float,
        base_min_temperature: float,
    ) -> float:
        """Compute effective minimum temperature for high-density arms.

        High-density arms get a higher minimum temperature to ensure
        they maintain exploration despite many constraints.
        """
        # Arms with density > 0.3 get a boosted minimum temperature
        if density > 0.3:
            return base_min_temperature * (1 + density)
        return base_min_temperature
