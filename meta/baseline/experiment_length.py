"""
Meta-experiment length optimizer — optimal K for budget/power tradeoff.
"""

import math
from meta.schemas import ExperimentLengthResult


class MetaExperimentLengthOptimizer:
    """Determine optimal meta-experiment length K."""

    def optimize(self, aggregate_ir, compute_budget_fraction: float = 0.2,
                 total_iterations_per_cycle: int = 500,
                 window_size: int = 20) -> ExperimentLengthResult:
        meta_budget = int(total_iterations_per_cycle * compute_budget_fraction)
        std_ir = aggregate_ir.std_ir if aggregate_ir.std_ir > 0 else 0.001
        mean_ir = abs(aggregate_ir.mean_ir) if aggregate_ir.mean_ir != 0 else 0.001

        best_K = window_size
        best_score = -1.0

        for K in range(window_size, meta_budget + 1, window_size):
            n_experiments = max(1, meta_budget // K)
            n_windows = K // window_size
            # MDES at this K
            mdes = 2.487 * std_ir * math.sqrt(2.0 / max(1, n_windows))
            # P(detect) approximation — higher n_windows = lower MDES = higher P
            p_detect = min(0.95, 1.0 - mdes / (mean_ir + mdes + 1e-10))
            score = n_experiments * p_detect
            if score > best_score:
                best_score = score
                best_K = K

        n_exp = max(1, meta_budget // best_K)
        n_windows = best_K // window_size
        mdes_at_K = 2.487 * std_ir * math.sqrt(2.0 / max(1, n_windows))

        return ExperimentLengthResult(
            optimal_K=best_K,
            n_experiments_per_cycle=n_exp,
            mdes_at_optimal_K=mdes_at_K,
            total_meta_iterations_per_cycle=meta_budget,
        )
