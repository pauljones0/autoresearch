"""
Phase 5 — KnowledgeCeilingMonitor: track the fraction of accepted modifications
that are paper-sourced over rolling windows to detect when LLM knowledge is
being exhausted.
"""

import json
import os
import time


class KnowledgeCeilingMonitor:
    """Monitor whether the paper pipeline is becoming more or less critical.

    If the fraction of accepted modifications that come from papers trends
    upward, the LLM's internal knowledge is being exhausted and the paper
    pipeline is increasingly critical.  If it trends toward zero, the LLM
    is sufficient on its own.
    """

    def __init__(self, data_dir: str = "."):
        self.data_dir = data_dir
        self._data_path = os.path.join(data_dir, "ceiling_monitor.jsonl")
        self._history: list[dict] = []

        if os.path.exists(self._data_path):
            self._load()

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def update(
        self,
        accepted_entries: list,
        window_size: int = 50,
    ) -> dict:
        """Update the monitor with a batch of accepted entries.

        Args:
            accepted_entries: list of accepted journal entry dicts.  Each must
                have a 'tags' or 'source' field indicating "paper" origin.
            window_size: rolling window size for fraction computation.

        Returns:
            dict with paper_fraction, trend, strategic_recommendation.
        """
        if not accepted_entries:
            return {
                "paper_fraction": 0.0,
                "trend": 0.0,
                "strategic_recommendation": "Insufficient data.",
            }

        # Count paper-sourced acceptances in the current batch
        paper_count = sum(1 for e in accepted_entries if self._is_paper_sourced(e))
        total = len(accepted_entries)
        fraction = paper_count / total if total else 0.0

        record = {
            "paper_fraction": fraction,
            "paper_count": paper_count,
            "total": total,
            "timestamp": time.time(),
        }
        self._history.append(record)
        self._save_record(record)

        # Compute trend over the rolling window
        recent = self._history[-window_size:]
        fractions = [r["paper_fraction"] for r in recent]
        trend = self.compute_trend(fractions)

        recommendation = self._recommend(fraction, trend)

        return {
            "paper_fraction": fraction,
            "trend": trend,
            "strategic_recommendation": recommendation,
        }

    @staticmethod
    def compute_trend(fractions: list) -> float:
        """Compute the slope of a linear fit over the fraction series.

        Uses closed-form OLS:  slope = cov(x, y) / var(x)
        where x = 0, 1, 2, ... and y = fractions.

        Returns 0.0 if fewer than 2 data points.
        """
        n = len(fractions)
        if n < 2:
            return 0.0

        mean_x = (n - 1) / 2.0
        mean_y = sum(fractions) / n

        cov_xy = sum((i - mean_x) * (y - mean_y) for i, y in enumerate(fractions)) / n
        var_x = sum((i - mean_x) ** 2 for i in range(n)) / n

        if var_x < 1e-15:
            return 0.0

        return cov_xy / var_x

    def get_report(self) -> str:
        """Generate a human-readable strategic assessment."""
        if not self._history:
            return "No data available. The ceiling monitor has not been updated yet."

        recent = self._history[-50:]
        fractions = [r["paper_fraction"] for r in recent]
        current = fractions[-1]
        trend = self.compute_trend(fractions)
        avg = sum(fractions) / len(fractions)

        lines = [
            "=== Knowledge Ceiling Monitor Report ===",
            f"Data points: {len(self._history)}",
            f"Current paper fraction: {current:.1%}",
            f"Rolling average: {avg:.1%}",
            f"Trend (slope): {trend:+.4f}",
            "",
        ]

        lines.append(self._recommend(current, trend))

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _is_paper_sourced(entry: dict) -> bool:
        """Determine if an accepted entry originated from the paper pipeline."""
        source = entry.get("source", "")
        if source == "paper":
            return True
        tags = entry.get("tags", [])
        if isinstance(tags, list):
            return "paper" in tags or "paper_sourced" in tags
        return False

    @staticmethod
    def _recommend(fraction: float, trend: float) -> str:
        """Generate a strategic recommendation from current fraction and trend."""
        if trend > 0.005:
            return (
                "TREND: INCREASING paper reliance. The LLM's internal knowledge "
                "appears to be approaching saturation. Prioritize expanding paper "
                "ingestion sources and improving extraction quality."
            )
        if trend < -0.005:
            return (
                "TREND: DECREASING paper reliance. The LLM is generating sufficient "
                "novel hypotheses internally. Consider reducing paper pipeline "
                "frequency to save compute."
            )

        if fraction > 0.5:
            return (
                "STABLE but HIGH paper fraction. The paper pipeline is a critical "
                "source of innovation. Maintain current ingestion rate."
            )
        if fraction < 0.1:
            return (
                "STABLE and LOW paper fraction. The LLM is self-sufficient. "
                "The paper pipeline has marginal value; consider running it "
                "less frequently."
            )

        return (
            "STABLE and MODERATE paper fraction. Both the LLM and paper pipeline "
            "contribute meaningfully. Current balance is healthy."
        )

    def _save_record(self, record: dict) -> None:
        with open(self._data_path, "a") as f:
            f.write(json.dumps(record) + "\n")

    def _load(self) -> None:
        try:
            with open(self._data_path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            self._history.append(json.loads(line))
                        except json.JSONDecodeError:
                            pass
        except FileNotFoundError:
            pass
