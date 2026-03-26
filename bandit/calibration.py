"""
Temperature calibration for target acceptance probability.
"""

import math


class TemperatureCalibrator:
    """Calibrates T_base to achieve a target acceptance probability."""

    def calibrate(
        self,
        target_acceptance_prob: float = 0.25,
        target_delta: float = 0.01,
    ) -> float:
        """Analytically derive T_base for a target acceptance probability.

        Uses the initial Beta(1,1) posterior where sigma_initial = sqrt(1/12).
        Formula: T_base = -target_delta / (sigma_initial * ln(target_acceptance_prob))
        """
        sigma_initial = math.sqrt(1.0 / 12.0)  # ~0.2887
        ln_p = math.log(target_acceptance_prob)
        T_base = -target_delta / (sigma_initial * ln_p)
        return T_base

    def sensitivity_analysis(
        self,
        deltas: list,
        T_base_values: list,
    ) -> dict:
        """Grid of acceptance probabilities for (delta, T_base) pairs.

        Returns dict with keys:
          - "deltas": list of delta values
          - "T_base_values": list of T_base values
          - "grid": list of lists, grid[i][j] = acceptance prob for deltas[i], T_base_values[j]
        """
        sigma_initial = math.sqrt(1.0 / 12.0)
        grid = []
        for delta in deltas:
            row = []
            for T_base in T_base_values:
                T_eff = T_base * sigma_initial
                if T_eff > 0 and delta > 0:
                    p = math.exp(-delta / T_eff)
                elif delta <= 0:
                    p = 1.0
                else:
                    p = 0.0
                row.append(p)
            grid.append(row)
        return {
            "deltas": deltas,
            "T_base_values": T_base_values,
            "grid": grid,
        }
