"""
Improvement rate calculator — rolling IR with cross-run statistics.
"""

import math
from meta.schemas import AggregateIR


class ImprovementRateCalculator:
    """Compute rolling improvement rate and cross-run statistics."""

    def compute_rolling(self, deltas: list, window: int = 20) -> list:
        """Compute rolling IR at each step as mean of accepted deltas in window."""
        if not deltas:
            return []
        rates = []
        for i in range(len(deltas)):
            start = max(0, i - window + 1)
            window_deltas = deltas[start:i + 1]
            # Only count negative deltas (improvements)
            accepted = [d for d in window_deltas if d < 0]
            ir = sum(accepted) / max(1, len(window_deltas))
            rates.append(ir)
        return rates

    def compute_aggregate(self, baseline_results: list) -> AggregateIR:
        """Compute cross-run improvement rate statistics."""
        all_irs = []
        per_run_means = []

        for result in baseline_results:
            if result.improvement_rates:
                all_irs.extend(result.improvement_rates)
                per_run_means.append(result.mean_ir)

        if not all_irs:
            return AggregateIR()

        n = len(all_irs)
        mean_ir = sum(all_irs) / n
        variance = sum((x - mean_ir) ** 2 for x in all_irs) / max(1, n - 1)
        std_ir = math.sqrt(variance) if variance > 0 else 0.0

        sorted_irs = sorted(all_irs)
        median_ir = sorted_irs[n // 2]
        q1 = sorted_irs[n // 4] if n >= 4 else sorted_irs[0]
        q3 = sorted_irs[3 * n // 4] if n >= 4 else sorted_irs[-1]
        iqr = q3 - q1

        # Variance decomposition
        within_var = 0.0
        between_var = 0.0
        if per_run_means:
            grand_mean = sum(per_run_means) / len(per_run_means)
            between_var = sum((m - grand_mean) ** 2 for m in per_run_means) / max(1, len(per_run_means) - 1)
            within_var = max(0, variance - between_var)

        z = 1.96  # 95% CI
        ci_lower = mean_ir - z * std_ir / math.sqrt(max(1, n))
        ci_upper = mean_ir + z * std_ir / math.sqrt(max(1, n))

        return AggregateIR(
            mean_ir=mean_ir, std_ir=std_ir,
            ci_95_lower=ci_lower, ci_95_upper=ci_upper,
            median_ir=median_ir, iqr=iqr,
            within_run_variance=within_var,
            between_run_variance=between_var,
            n_windows=n, n_runs=len(baseline_results),
        )
