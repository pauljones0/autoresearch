"""Joint optimization of interacting parameter pairs."""

from typing import List, Tuple

from meta.schemas import MetaExperimentResult


class JointOptimizer:
    """Finds optimal joint settings for two interacting dimensions."""

    def optimize_joint(
        self,
        dim_i: str,
        dim_j: str,
        experiment_history: List[MetaExperimentResult],
    ) -> tuple:
        """Build a 2D grid and find the cell with highest IR.

        Returns (best_value_i, best_value_j, best_ir).
        """
        # Build grid: (variant_i, variant_j) -> list of IR observations
        grid: dict = {}

        for exp in experiment_history:
            varied = {}
            for diff in exp.config_diff:
                if isinstance(diff, dict):
                    pid = diff.get("param_id", "")
                    new_val = diff.get("new_value")
                else:
                    pid = getattr(diff, "param_id", "")
                    new_val = getattr(diff, "new_value", None)
                if pid:
                    varied[pid] = new_val

            # Determine value for dim_i and dim_j
            val_i = varied.get(dim_i, "__default__")
            val_j = varied.get(dim_j, "__default__")

            key = (str(val_i), str(val_j))
            grid.setdefault(key, []).append(exp.improvement_rate)

        if not grid:
            return (None, None, 0.0)

        # Find cell with highest mean IR
        best_key = None
        best_ir = -float("inf")

        for key, ir_values in grid.items():
            mean_ir = sum(ir_values) / len(ir_values)
            if mean_ir > best_ir:
                best_ir = mean_ir
                best_key = key

        if best_key is None:
            return (None, None, 0.0)

        return (best_key[0], best_key[1], best_ir)
