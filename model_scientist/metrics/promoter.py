"""
Phase 4: MetricPromoter — runs the selection pressure cycle,
promoting high-correlation candidates and retiring stale metrics.
"""

from ..schemas import MetricCorrelation
from .registry import MetricRegistry


class MetricPromoter:
    """Promotes and retires metrics based on correlation thresholds."""

    def __init__(self, retire_after_cycles: int = 2):
        self.retire_after_cycles = retire_after_cycles

    def run_cycle(
        self,
        registry: MetricRegistry,
        correlations: list,
        threshold: float = 0.3,
    ) -> dict:
        """Execute one promotion/retirement cycle.

        Args:
            registry: The MetricRegistry to mutate.
            correlations: List of MetricCorrelation from the correlator.
            threshold: Minimum |r| to be considered predictive.

        Returns:
            {"promotions": [...], "retirements": [...], "log": [...]}
        """
        corr_map: dict[str, MetricCorrelation] = {}
        for c in correlations:
            name = c.metric_name if isinstance(c, MetricCorrelation) else c.get("metric_name", "")
            corr_map[name] = c

        promotions: list[str] = []
        retirements: list[str] = []
        log: list[str] = []

        # --- Candidates: promote if above threshold ---
        for metric in registry.get_candidates():
            c = corr_map.get(metric.name)
            if c is None:
                continue
            r = c.correlation_r if isinstance(c, MetricCorrelation) else c.get("correlation_r", 0.0)
            n = c.n_experiments if isinstance(c, MetricCorrelation) else c.get("n_experiments", 0)
            registry.update_correlation(metric.name, r)

            if abs(r) >= threshold and n >= 2:
                registry.promote(metric.name)
                promotions.append(metric.name)
                log.append(f"PROMOTE '{metric.name}': |r|={abs(r):.3f} >= {threshold} (n={n})")
            else:
                log.append(f"KEEP candidate '{metric.name}': |r|={abs(r):.3f} < {threshold} (n={n})")

        # --- Active metrics: track and possibly retire ---
        for metric in registry.get_active():
            c = corr_map.get(metric.name)
            if c is None:
                continue
            r = c.correlation_r if isinstance(c, MetricCorrelation) else c.get("correlation_r", 0.0)
            n = c.n_experiments if isinstance(c, MetricCorrelation) else c.get("n_experiments", 0)
            registry.update_correlation(metric.name, r)

            if abs(r) < threshold:
                registry.increment_low_correlation(metric.name)
                current = registry.get(metric.name)
                cycles = current.consecutive_low_correlation_cycles if current else 0
                if cycles >= self.retire_after_cycles:
                    # Don't retire hardcoded metrics
                    if metric.source == "hardcoded":
                        log.append(
                            f"KEEP hardcoded '{metric.name}': |r|={abs(r):.3f} low for {cycles} cycles but protected"
                        )
                        continue
                    registry.retire(metric.name)
                    retirements.append(metric.name)
                    log.append(
                        f"RETIRE '{metric.name}': |r|={abs(r):.3f} < {threshold} for {cycles} consecutive cycles"
                    )
                else:
                    log.append(
                        f"WARNING '{metric.name}': |r|={abs(r):.3f} < {threshold} ({cycles}/{self.retire_after_cycles} cycles)"
                    )
            else:
                registry.reset_low_correlation(metric.name)
                log.append(f"OK '{metric.name}': |r|={abs(r):.3f} >= {threshold}")

        return {
            "promotions": promotions,
            "retirements": retirements,
            "log": log,
        }
