"""
MetaLearningMonitor — tracks the four-level learning system's health:
  1. Model quality: Is val_bpb improving?
  2. Surrogate accuracy: Is the surrogate's accuracy improving?
  3. Ingestion precision: Is the filter targeting productive papers?
  4. Metric evolution: Are evolved metrics improving the surrogate?
"""

import json
import os
import time
from datetime import datetime


class MetaLearningMonitor:
    """Tracks health across all four learning levels."""

    def __init__(self, data_dir: str = None):
        self.data_dir = data_dir or os.path.join(
            os.path.dirname(__file__), "..", "..", "data"
        )
        self._history_path = os.path.join(self.data_dir, "meta_learning_history.jsonl")

    def collect_snapshot(
        self,
        journal_reader=None,
        surrogate_evaluator_metrics: dict = None,
        ingestion_stats: dict = None,
        metric_registry=None,
    ) -> dict:
        """Collect a point-in-time snapshot across all four levels.

        Args:
            journal_reader: JournalReader instance for val_bpb trends.
            surrogate_evaluator_metrics: Dict from SurrogateEvaluator.evaluate().
            ingestion_stats: Dict with papers_ingested, papers_evaluated, papers_accepted.
            metric_registry: MetricRegistry instance.

        Returns:
            Snapshot dict with all four levels.
        """
        snapshot = {"timestamp": time.time(), "levels": {}}

        # Level 1: Model quality
        level1 = {"val_bpb_trend": "unknown", "recent_success_rate": 0.0}
        if journal_reader:
            try:
                stats = journal_reader.summary_stats()
                level1["recent_success_rate"] = stats.get("success_rate", 0.0)
                recent = journal_reader.recent(20)
                if len(recent) >= 2:
                    deltas = [e.get("actual_delta", 0) for e in recent if e.get("actual_delta")]
                    if deltas:
                        avg_delta = sum(deltas) / len(deltas)
                        level1["avg_recent_delta"] = avg_delta
                        level1["val_bpb_trend"] = "improving" if avg_delta < 0 else "stagnant"
            except Exception:
                pass
        snapshot["levels"]["model_quality"] = level1

        # Level 2: Surrogate accuracy
        level2 = {"spearman_rho": 0.0, "mae": 0.0, "trend": "unknown"}
        if surrogate_evaluator_metrics:
            level2["spearman_rho"] = surrogate_evaluator_metrics.get("spearman_rho", 0.0)
            level2["mae"] = surrogate_evaluator_metrics.get("mae", 0.0)
            level2["worst_case_rate"] = surrogate_evaluator_metrics.get("worst_case_rate", 0.0)
        snapshot["levels"]["surrogate_accuracy"] = level2

        # Level 3: Ingestion precision
        level3 = {"precision": 0.0, "papers_ingested": 0, "papers_accepted": 0}
        if ingestion_stats:
            total = ingestion_stats.get("papers_evaluated", 0)
            accepted = ingestion_stats.get("papers_accepted", 0)
            level3["papers_ingested"] = ingestion_stats.get("papers_ingested", 0)
            level3["papers_evaluated"] = total
            level3["papers_accepted"] = accepted
            level3["precision"] = accepted / max(total, 1)
        snapshot["levels"]["ingestion_precision"] = level3

        # Level 4: Metric evolution
        level4 = {"active_metrics": 0, "total_metrics": 0, "avg_correlation": 0.0}
        if metric_registry:
            try:
                active = metric_registry.get_active()
                all_metrics = active + metric_registry.get_candidates()
                level4["active_metrics"] = len(active)
                level4["total_metrics"] = len(all_metrics)
                correlations = [
                    m.correlation_with_success
                    for m in active
                    if hasattr(m, "correlation_with_success") and m.correlation_with_success
                ]
                if correlations:
                    level4["avg_correlation"] = sum(correlations) / len(correlations)
            except Exception:
                pass
        snapshot["levels"]["metric_evolution"] = level4

        return snapshot

    def record_snapshot(self, snapshot: dict):
        """Append snapshot to history."""
        os.makedirs(self.data_dir, exist_ok=True)
        with open(self._history_path, "a") as f:
            f.write(json.dumps(snapshot) + "\n")

    def get_history(self, n: int = 0) -> list:
        """Load snapshot history. If n > 0, return last n entries."""
        entries = []
        if not os.path.exists(self._history_path):
            return entries
        with open(self._history_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        entries.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
        if n > 0:
            entries = entries[-n:]
        return entries

    def compute_trends(self, window: int = 10) -> dict:
        """Compute trends over the last `window` snapshots."""
        history = self.get_history(window)
        if len(history) < 2:
            return {"insufficient_data": True}

        trends = {}

        # Model quality trend
        deltas = [
            s["levels"].get("model_quality", {}).get("avg_recent_delta", 0)
            for s in history
            if "avg_recent_delta" in s.get("levels", {}).get("model_quality", {})
        ]
        if len(deltas) >= 2:
            trends["model_quality_slope"] = _linear_slope(deltas)

        # Surrogate accuracy trend
        rhos = [
            s["levels"].get("surrogate_accuracy", {}).get("spearman_rho", 0)
            for s in history
        ]
        if len(rhos) >= 2:
            trends["surrogate_accuracy_slope"] = _linear_slope(rhos)

        # Ingestion precision trend
        precisions = [
            s["levels"].get("ingestion_precision", {}).get("precision", 0)
            for s in history
        ]
        if len(precisions) >= 2:
            trends["ingestion_precision_slope"] = _linear_slope(precisions)

        # Metric correlation trend
        corrs = [
            s["levels"].get("metric_evolution", {}).get("avg_correlation", 0)
            for s in history
        ]
        if len(corrs) >= 2:
            trends["metric_evolution_slope"] = _linear_slope(corrs)

        return trends

    def generate_report(self) -> str:
        """Generate a human-readable health report."""
        history = self.get_history(1)
        latest = history[-1] if history else {}
        trends = self.compute_trends()

        lines = []
        lines.append("# Meta-Learning Health Report")
        lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")

        if not latest:
            lines.append("No data available yet.")
            return "\n".join(lines)

        levels = latest.get("levels", {})

        # Level 1
        mq = levels.get("model_quality", {})
        lines.append("## Level 1: Model Quality")
        lines.append(f"- Trend: {mq.get('val_bpb_trend', 'unknown')}")
        lines.append(f"- Recent success rate: {mq.get('recent_success_rate', 0):.1%}")
        slope = trends.get("model_quality_slope")
        if slope is not None:
            direction = "improving" if slope < 0 else "stagnating"
            lines.append(f"- Delta trend: {direction} (slope={slope:.6f})")
        lines.append("")

        # Level 2
        sa = levels.get("surrogate_accuracy", {})
        lines.append("## Level 2: Surrogate Accuracy")
        lines.append(f"- Spearman rho: {sa.get('spearman_rho', 0):.3f}")
        lines.append(f"- MAE: {sa.get('mae', 0):.6f}")
        slope = trends.get("surrogate_accuracy_slope")
        if slope is not None:
            direction = "improving" if slope > 0 else "degrading"
            lines.append(f"- Accuracy trend: {direction} (slope={slope:.4f})")
        lines.append("")

        # Level 3
        ip = levels.get("ingestion_precision", {})
        lines.append("## Level 3: Ingestion Precision")
        lines.append(f"- Papers ingested: {ip.get('papers_ingested', 0)}")
        lines.append(f"- Papers evaluated: {ip.get('papers_evaluated', 0)}")
        lines.append(f"- Papers accepted: {ip.get('papers_accepted', 0)}")
        lines.append(f"- Precision: {ip.get('precision', 0):.1%}")
        slope = trends.get("ingestion_precision_slope")
        if slope is not None:
            direction = "improving" if slope > 0 else "degrading"
            lines.append(f"- Precision trend: {direction} (slope={slope:.4f})")
        lines.append("")

        # Level 4
        me = levels.get("metric_evolution", {})
        lines.append("## Level 4: Metric Evolution")
        lines.append(f"- Active metrics: {me.get('active_metrics', 0)}")
        lines.append(f"- Avg correlation: {me.get('avg_correlation', 0):.3f}")
        lines.append("")

        return "\n".join(lines)


def _linear_slope(values: list) -> float:
    """Compute slope of best-fit line through values (indexed 0..n-1)."""
    n = len(values)
    if n < 2:
        return 0.0
    x_mean = (n - 1) / 2.0
    y_mean = sum(values) / n
    num = sum((i - x_mean) * (y - y_mean) for i, y in enumerate(values))
    den = sum((i - x_mean) ** 2 for i in range(n))
    return num / den if den != 0 else 0.0
