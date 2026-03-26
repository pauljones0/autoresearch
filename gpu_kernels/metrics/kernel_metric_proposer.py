"""
KernelMetricProposer: proposes kernel-derived diagnostic metrics for
the model scientist metric evolution system.
"""

import math
import time

from model_scientist.schemas import MetricDefinition
from gpu_kernels.schemas import KernelConfigEntry


class KernelMetricProposer:
    """Propose kernel-derived metrics for the diagnostic system.

    Metrics:
      - active_kernel_speedup_total: geometric mean of all active kernel speedups.
      - kernel_coverage_fraction: fraction of step time covered by custom kernels.
      - bandwidth_utilization_improvement: mean improvement in bandwidth utilization.
      - kernel_failure_rate_rolling: rolling 20-experiment failure rate.
    """

    def propose(
        self,
        kernel_config: dict,
        diagnostics: dict,
        existing_metrics: list[MetricDefinition | dict] | None = None,
    ) -> list[MetricDefinition]:
        """Propose kernel-derived metrics.

        Args:
            kernel_config: Current kernel_config.json as dict (group_id -> entry).
            diagnostics: DiagnosticsReport as dict.
            existing_metrics: Already-registered metrics (to avoid duplicates).

        Returns:
            List of MetricDefinition for kernel-derived metrics.
        """
        if existing_metrics is None:
            existing_metrics = []

        existing_names = set()
        for m in existing_metrics:
            if isinstance(m, dict):
                existing_names.add(m.get("name", ""))
            else:
                existing_names.add(m.name)

        proposals: list[MetricDefinition] = []

        # Metric 1: active_kernel_speedup_total
        name = "active_kernel_speedup_total"
        if name not in existing_names:
            speedup_val = self._compute_geometric_mean_speedup(kernel_config)
            proposals.append(MetricDefinition(
                name=name,
                description="Geometric mean of speedup ratios across all active Triton kernels",
                computation_method=(
                    "geometric_mean([entry.speedup for entry in kernel_config.values() "
                    "if entry.backend == 'triton' and entry.enabled])"
                ),
                rationale=(
                    "Tracks overall kernel optimization quality. A declining value "
                    "signals regressions in kernel performance."
                ),
                source="kernel_pipeline",
                status="candidate",
                correlation_with_success=speedup_val,
            ))

        # Metric 2: kernel_coverage_fraction
        name = "kernel_coverage_fraction"
        if name not in existing_names:
            coverage = self._compute_coverage_fraction(kernel_config, diagnostics)
            proposals.append(MetricDefinition(
                name=name,
                description="Fraction of total training step time covered by custom Triton kernels",
                computation_method=(
                    "sum(entry.step_fraction for entry in kernel_config.values() "
                    "if entry.backend == 'triton' and entry.enabled)"
                ),
                rationale=(
                    "Measures how much of the training step benefits from kernel "
                    "optimizations. Higher coverage means more potential for speedup."
                ),
                source="kernel_pipeline",
                status="candidate",
                correlation_with_success=coverage,
            ))

        # Metric 3: bandwidth_utilization_improvement
        name = "bandwidth_utilization_improvement"
        if name not in existing_names:
            bw_improvement = self._compute_bw_improvement(kernel_config)
            proposals.append(MetricDefinition(
                name=name,
                description="Mean improvement in bandwidth utilization from kernel optimizations",
                computation_method=(
                    "mean(entry.bandwidth_utilization - entry.baseline_bandwidth "
                    "for entry in kernel_config.values() if entry.backend == 'triton')"
                ),
                rationale=(
                    "Bandwidth utilization improvement indicates how well kernels "
                    "exploit memory bandwidth compared to baseline PyTorch ops."
                ),
                source="kernel_pipeline",
                status="candidate",
                correlation_with_success=bw_improvement,
            ))

        # Metric 4: kernel_failure_rate_rolling
        name = "kernel_failure_rate_rolling"
        if name not in existing_names:
            failure_rate = self._compute_failure_rate(kernel_config)
            proposals.append(MetricDefinition(
                name=name,
                description="Rolling 20-experiment kernel generation failure rate",
                computation_method=(
                    "sum(1 for e in last_20_experiments if e.verdict == 'rejected') / 20"
                ),
                rationale=(
                    "A high failure rate suggests the search space is exhausted or "
                    "constraints are too loose. Useful for tuning the discovery loop."
                ),
                source="kernel_pipeline",
                status="candidate",
                correlation_with_success=1.0 - failure_rate,
            ))

        return proposals

    def _compute_geometric_mean_speedup(self, kernel_config: dict) -> float:
        """Compute geometric mean of active kernel speedups."""
        speedups = []
        for entry in kernel_config.values():
            if isinstance(entry, dict):
                if entry.get("backend") == "triton" and entry.get("enabled", True):
                    s = entry.get("speedup", 1.0)
                    if s > 0:
                        speedups.append(s)
            elif isinstance(entry, KernelConfigEntry):
                if entry.backend == "triton" and entry.enabled:
                    if entry.speedup > 0:
                        speedups.append(entry.speedup)

        if not speedups:
            return 1.0

        log_sum = sum(math.log(s) for s in speedups)
        return math.exp(log_sum / len(speedups))

    def _compute_coverage_fraction(
        self, kernel_config: dict, diagnostics: dict
    ) -> float:
        """Compute fraction of step time covered by custom kernels."""
        total_fraction = 0.0
        for entry in kernel_config.values():
            if isinstance(entry, dict):
                if entry.get("backend") == "triton" and entry.get("enabled", True):
                    total_fraction += entry.get("step_fraction", 0.0)
            elif isinstance(entry, KernelConfigEntry):
                if entry.backend == "triton" and entry.enabled:
                    total_fraction += getattr(entry, "step_fraction", 0.0)
        return min(total_fraction, 1.0)

    def _compute_bw_improvement(self, kernel_config: dict) -> float:
        """Compute mean bandwidth utilization improvement."""
        improvements = []
        for entry in kernel_config.values():
            if isinstance(entry, dict):
                if entry.get("backend") == "triton":
                    bw = entry.get("bandwidth_utilization", 0.0)
                    baseline_bw = entry.get("baseline_bandwidth", 0.0)
                    if baseline_bw > 0:
                        improvements.append(bw - baseline_bw)
            elif isinstance(entry, KernelConfigEntry):
                if entry.backend == "triton":
                    bw = getattr(entry, "bandwidth_utilization", 0.0)
                    baseline_bw = getattr(entry, "baseline_bandwidth", 0.0)
                    if baseline_bw > 0:
                        improvements.append(bw - baseline_bw)

        if not improvements:
            return 0.0
        return sum(improvements) / len(improvements)

    def _compute_failure_rate(self, kernel_config: dict) -> float:
        """Compute rolling failure rate from config metadata."""
        # Extract recent experiment results from config
        recent_results = []
        for entry in kernel_config.values():
            if isinstance(entry, dict):
                history = entry.get("generation_history", [])
            else:
                history = getattr(entry, "generation_history", [])

            if isinstance(history, list):
                for h in history:
                    if isinstance(h, dict):
                        recent_results.append(h.get("passed", True))

        # Take last 20
        recent_results = recent_results[-20:]
        if not recent_results:
            return 0.0
        failures = sum(1 for r in recent_results if not r)
        return failures / len(recent_results)
