"""
Phase 3.8 — ExtractionQualityTracker: track per-paper extraction quality
scores derived from surrogate prediction error.
"""

import logging
import time
from collections import defaultdict

from surrogate_triage.schemas import ExtractionQualityRecord, save_jsonl, load_jsonl

logger = logging.getLogger(__name__)


class ExtractionQualityTracker:
    """Track extraction quality per paper and technique."""

    def __init__(self, quality_path: str = "extraction_quality.jsonl"):
        self.quality_path = quality_path
        self._records: list[ExtractionQualityRecord] = []

    def record(
        self,
        paper_id: str,
        technique_id: str,
        predicted: float,
        actual: float,
    ) -> ExtractionQualityRecord:
        """Record an extraction quality observation.

        quality_score = 1 / (1 + |prediction_error|) — higher is better.

        Args:
            paper_id: arXiv paper ID.
            technique_id: Technique identifier.
            predicted: Surrogate predicted delta.
            actual: Actual observed delta.

        Returns:
            ExtractionQualityRecord with computed quality score.
        """
        prediction_error = predicted - actual
        quality_score = 1.0 / (1.0 + abs(prediction_error))

        record = ExtractionQualityRecord(
            paper_id=paper_id,
            technique_id=technique_id,
            surrogate_predicted_delta=predicted,
            actual_delta=actual,
            prediction_error=prediction_error,
            quality_score=quality_score,
            evaluated_at=time.time(),
        )

        self._records.append(record)

        # Persist
        try:
            save_jsonl([record], self.quality_path)
        except Exception as exc:
            logger.error("Failed to save extraction quality record: %s", exc)

        return record

    def get_quality_scores(self) -> dict:
        """Return average quality score per paper_id.

        Returns:
            Dict mapping paper_id to average quality_score.
        """
        scores_by_paper: dict[str, list[float]] = defaultdict(list)
        for r in self._records:
            scores_by_paper[r.paper_id].append(r.quality_score)

        return {
            pid: sum(scores) / len(scores)
            for pid, scores in scores_by_paper.items()
        }

    def get_low_quality_papers(self, threshold: float = 0.5) -> list:
        """Return paper_ids with average quality below threshold."""
        scores = self.get_quality_scores()
        return [
            pid for pid, avg in scores.items() if avg < threshold
        ]

    def load(self):
        """Load existing records from disk."""
        raw = load_jsonl(self.quality_path)
        self._records = [ExtractionQualityRecord.from_dict(d) for d in raw] if raw else []

    def stats(self) -> dict:
        """Return summary statistics."""
        if not self._records:
            return {"total_records": 0, "avg_quality": 0.0}
        avg = sum(r.quality_score for r in self._records) / len(self._records)
        return {
            "total_records": len(self._records),
            "avg_quality": round(avg, 4),
            "papers_tracked": len(self.get_quality_scores()),
        }
