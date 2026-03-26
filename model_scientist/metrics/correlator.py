"""
Phase 4: MetricCorrelator — computes Pearson correlation between
each metric's value and experiment outcomes.
"""

import math

from ..schemas import MetricDefinition, MetricCorrelation, DiagnosticsReport
from .implementer import MetricImplementer


def _pearson_r(xs: list, ys: list) -> float:
    """Compute Pearson correlation coefficient from scratch.

    Returns 0.0 for degenerate cases (constant values, < 2 points).
    """
    n = len(xs)
    if n < 2 or len(ys) != n:
        return 0.0

    x_mean = sum(xs) / n
    y_mean = sum(ys) / n

    num = 0.0
    den_x = 0.0
    den_y = 0.0
    for x, y in zip(xs, ys):
        dx = x - x_mean
        dy = y - y_mean
        num += dx * dy
        den_x += dx * dx
        den_y += dy * dy

    denom = math.sqrt(den_x * den_y)
    if denom == 0.0:
        return 0.0
    return num / denom


def _p_value_approx(r: float, n: int) -> float:
    """Rough two-tailed p-value via t-distribution approximation.

    Uses the t = r * sqrt((n-2)/(1-r^2)) statistic.  For small n the
    approximation is coarse, but we only need a rough significance check.
    """
    if n < 3 or abs(r) >= 1.0:
        return 1.0
    t_stat = abs(r) * math.sqrt((n - 2) / (1.0 - r * r))
    # Approximate using the normal CDF for large-ish t
    # P(|T| > t) ≈ 2 * Φ(-t) via logistic approximation
    p = 2.0 / (1.0 + math.exp(0.7 * t_stat))
    return min(p, 1.0)


class MetricCorrelator:
    """Computes correlations between metric values and experiment outcomes."""

    def __init__(self, eval_interval: int = 10):
        self.eval_interval = eval_interval
        self._implementer = MetricImplementer()

    def compute_correlations(
        self,
        metrics: list,
        journal_entries: list,
        diagnostics_reports: list,
    ) -> list:
        """Compute Pearson r for each metric against experiment outcomes.

        Args:
            metrics: List of MetricDefinition.
            journal_entries: List of dicts (or JournalEntry-like objects)
                with at least 'verdict' and optionally 'actual_delta'.
            diagnostics_reports: Corresponding DiagnosticsReport (or dicts)
                aligned by index with journal_entries.

        Returns:
            List of MetricCorrelation dataclasses.
        """
        if not metrics or not journal_entries or not diagnostics_reports:
            return []

        # Build outcome vector: 1.0 for accepted, 0.0 for rejected, skip crashed
        outcomes: list[float] = []
        valid_reports: list = []
        for entry, report in zip(journal_entries, diagnostics_reports):
            verdict = entry.get("verdict", "") if isinstance(entry, dict) else getattr(entry, "verdict", "")
            if verdict == "accepted":
                outcomes.append(1.0)
            elif verdict == "rejected":
                outcomes.append(0.0)
            else:
                continue
            valid_reports.append(report)

        if len(outcomes) < 2:
            return []

        results: list[MetricCorrelation] = []
        for metric in metrics:
            values = self._compute_values(metric, valid_reports)
            if values is None or len(values) != len(outcomes):
                results.append(MetricCorrelation(
                    metric_name=metric.name if isinstance(metric, MetricDefinition) else metric.get("name", ""),
                    correlation_r=0.0,
                    p_value=1.0,
                    n_experiments=0,
                ))
                continue

            r = _pearson_r(values, outcomes)
            p = _p_value_approx(r, len(values))
            results.append(MetricCorrelation(
                metric_name=metric.name if isinstance(metric, MetricDefinition) else metric.get("name", ""),
                correlation_r=r,
                p_value=p,
                n_experiments=len(values),
            ))

        return results

    def _compute_values(self, metric, reports: list):
        """Safely compute metric values across all reports."""
        values: list[float] = []
        for report in reports:
            try:
                val = self._implementer.compute_metric(metric, report)
                values.append(val)
            except (ValueError, Exception):
                return None
        return values
