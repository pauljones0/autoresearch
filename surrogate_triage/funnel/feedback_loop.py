"""
Phase 3.7 — SurrogateFeedbackLoop: feed evaluation results back to the
surrogate retraining pipeline and track prediction errors.
"""

import logging
import time

from surrogate_triage.schemas import save_jsonl

logger = logging.getLogger(__name__)

_LARGE_ERROR_THRESHOLD = 0.05  # Flag errors larger than this


class SurrogateFeedbackLoop:
    """Feed paper evaluation results back to the surrogate for retraining."""

    def __init__(
        self,
        feedback_path: str = "surrogate_feedback.jsonl",
        error_threshold: float = _LARGE_ERROR_THRESHOLD,
    ):
        self.feedback_path = feedback_path
        self.error_threshold = error_threshold
        self._results: list[dict] = []

    def record_result(
        self,
        diff_id: str,
        predicted_delta: float,
        actual_delta: float,
        source: str = "paper",
    ) -> dict:
        """Record an evaluation result for surrogate feedback.

        Args:
            diff_id: Identifier for the diff that was evaluated.
            predicted_delta: Surrogate's predicted val_bpb delta.
            actual_delta: Actual val_bpb delta observed.
            source: "paper" or "internal".

        Returns:
            Dict with prediction error info and whether it was flagged.
        """
        error = predicted_delta - actual_delta
        abs_error = abs(error)
        flagged = abs_error > self.error_threshold

        record = {
            "diff_id": diff_id,
            "predicted_delta": predicted_delta,
            "actual_delta": actual_delta,
            "error": error,
            "abs_error": abs_error,
            "source": source,
            "flagged": flagged,
            "recorded_at": time.time(),
        }

        self._results.append(record)

        if flagged:
            logger.warning(
                "Large surrogate prediction error for %s: "
                "predicted=%.4f, actual=%.4f, error=%.4f",
                diff_id, predicted_delta, actual_delta, error,
            )

        # Persist to feedback file
        try:
            save_jsonl([record], self.feedback_path)
        except Exception as exc:
            logger.error("Failed to save feedback record: %s", exc)

        return record

    def get_prediction_errors(self) -> list:
        """Return list of (predicted, actual, error) tuples for all recorded results."""
        return [
            (r["predicted_delta"], r["actual_delta"], r["error"])
            for r in self._results
        ]

    def get_flagged_errors(self) -> list:
        """Return only flagged (large error) results."""
        return [r for r in self._results if r.get("flagged")]

    def mean_absolute_error(self) -> float:
        """Compute mean absolute error across all recorded results."""
        if not self._results:
            return 0.0
        return sum(r["abs_error"] for r in self._results) / len(self._results)

    def stats(self) -> dict:
        """Return summary statistics."""
        total = len(self._results)
        flagged = sum(1 for r in self._results if r.get("flagged"))
        return {
            "total_results": total,
            "flagged_count": flagged,
            "mean_absolute_error": round(self.mean_absolute_error(), 6),
        }
