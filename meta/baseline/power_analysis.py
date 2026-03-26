"""
Minimum detectable effect size calculator — power analysis.
"""

import math
from meta.schemas import MDESResult


# Standard normal quantiles
_Z_ALPHA_005 = 1.645   # one-sided alpha=0.05
_Z_POWER_08 = 0.842    # power=0.8


class MinimumDetectableEffectCalculator:
    """Power analysis for meta-experiment design."""

    def compute_mdes(self, aggregate_ir, meta_experiment_length: int = 50,
                     alpha: float = 0.05, power: float = 0.8,
                     window_size: int = 20) -> MDESResult:
        n_windows = max(1, meta_experiment_length // window_size)
        std_ir = aggregate_ir.std_ir if aggregate_ir.std_ir > 0 else 0.001

        z_alpha = _Z_ALPHA_005 if alpha == 0.05 else 1.282
        z_power = _Z_POWER_08 if power == 0.8 else 0.842

        mdes_abs = (z_alpha + z_power) * std_ir * math.sqrt(2.0 / max(1, n_windows))

        mean_ir = abs(aggregate_ir.mean_ir) if aggregate_ir.mean_ir != 0 else 0.001
        mdes_rel = mdes_abs / mean_ir

        return MDESResult(
            mdes_absolute=mdes_abs,
            mdes_relative=mdes_rel,
            meta_experiment_length=meta_experiment_length,
            alpha=alpha,
            power=power,
            n_windows_per_experiment=n_windows,
        )
