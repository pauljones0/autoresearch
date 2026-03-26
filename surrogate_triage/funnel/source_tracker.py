"""
Phase 3.9 — PaperSourceTracker: track paper source quality (success rates)
by author, venue, category, and technique_category.
"""

import json
import logging
import os
import time
from collections import defaultdict

from surrogate_triage.schemas import PaperSourceQuality

logger = logging.getLogger(__name__)


class PaperSourceTracker:
    """Track paper source quality across multiple dimensions."""

    def __init__(self, quality_path: str = "paper_source_quality.json"):
        self.quality_path = quality_path
        # Nested dict: dimension -> value -> tracking data
        self._data: dict[str, dict[str, dict]] = defaultdict(
            lambda: defaultdict(lambda: {
                "total_evaluated": 0,
                "total_accepted": 0,
                "delta_sum": 0.0,
                "last_updated": 0.0,
            })
        )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self):
        """Persist quality data to disk."""
        os.makedirs(os.path.dirname(self.quality_path) or ".", exist_ok=True)
        # Convert defaultdicts to regular dicts for JSON serialization
        data = {dim: dict(vals) for dim, vals in self._data.items()}
        with open(self.quality_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def load(self):
        """Load quality data from disk."""
        if not os.path.exists(self.quality_path):
            return
        try:
            with open(self.quality_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self._data = defaultdict(
                lambda: defaultdict(lambda: {
                    "total_evaluated": 0,
                    "total_accepted": 0,
                    "delta_sum": 0.0,
                    "last_updated": 0.0,
                })
            )
            for dim, vals in data.items():
                for val, stats in vals.items():
                    self._data[dim][val] = stats
        except (json.JSONDecodeError, IOError) as exc:
            logger.warning("Failed to load source quality data: %s", exc)

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record_evaluation(
        self,
        paper_metadata: dict,
        technique_category: str,
        verdict: str,
        delta: float,
    ):
        """Record an evaluation result across all tracked dimensions.

        Args:
            paper_metadata: Dict with keys like authors, categories, etc.
            technique_category: The modification category of the technique.
            verdict: "accepted", "rejected", or "crashed".
            delta: Actual val_bpb delta.
        """
        accepted = verdict == "accepted"
        now = time.time()

        # Track by technique_category
        self._record_one("technique_category", technique_category, accepted, delta, now)

        # Track by category (arXiv categories)
        for cat in paper_metadata.get("categories", []):
            self._record_one("category", cat, accepted, delta, now)

        # Track by author
        for author in paper_metadata.get("authors", []):
            author_name = author if isinstance(author, str) else str(author)
            self._record_one("author", author_name, accepted, delta, now)

        # Track by venue (if available)
        venue = paper_metadata.get("venue", "")
        if venue:
            self._record_one("venue", venue, accepted, delta, now)

    def _record_one(
        self,
        dimension: str,
        value: str,
        accepted: bool,
        delta: float,
        timestamp: float,
    ):
        """Record a single dimension/value observation."""
        if not value:
            return
        entry = self._data[dimension][value]
        entry["total_evaluated"] += 1
        if accepted:
            entry["total_accepted"] += 1
        entry["delta_sum"] += delta
        entry["last_updated"] = timestamp

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_quality(self, dimension: str, value: str) -> PaperSourceQuality:
        """Get quality stats for a specific dimension/value pair.

        Args:
            dimension: One of "author", "venue", "category", "technique_category".
            value: The specific author name, venue, etc.

        Returns:
            PaperSourceQuality dataclass.
        """
        entry = self._data.get(dimension, {}).get(value, {})
        total = entry.get("total_evaluated", 0)
        accepted = entry.get("total_accepted", 0)
        delta_sum = entry.get("delta_sum", 0.0)

        return PaperSourceQuality(
            dimension=dimension,
            value=value,
            total_evaluated=total,
            total_accepted=accepted,
            success_rate=accepted / total if total > 0 else 0.0,
            avg_delta=delta_sum / total if total > 0 else 0.0,
            last_updated=entry.get("last_updated", 0.0),
        )

    def get_top_sources(self, dimension: str, n: int = 10) -> list:
        """Get top sources by success rate for a dimension.

        Args:
            dimension: Dimension to query.
            n: Number of top sources to return.

        Returns:
            List of PaperSourceQuality sorted by success rate descending.
        """
        dim_data = self._data.get(dimension, {})
        qualities = []
        for value in dim_data:
            q = self.get_quality(dimension, value)
            if q.total_evaluated > 0:
                qualities.append(q)

        qualities.sort(key=lambda q: q.success_rate, reverse=True)
        return qualities[:n]
