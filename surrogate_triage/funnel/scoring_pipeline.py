"""
Phase 3.1 — SurrogateScoringPipeline: enrich features, score with surrogate,
apply constraint penalties, and rank candidates by adjusted score.
"""

import logging

from surrogate_triage.schemas import SurrogatePrediction, SyntheticDiff

logger = logging.getLogger(__name__)


class SurrogateScoringPipeline:
    """Score and rank synthetic diffs using the surrogate model."""

    def score_and_rank(
        self,
        diffs: list,
        surrogate_model,
        enricher,
        constraints: list = None,
    ) -> list:
        """Score diffs with the surrogate and return ranked SurrogatePredictions.

        Args:
            diffs: List of SyntheticDiff objects.
            surrogate_model: Object with a ``predict(feature_vector) -> (delta, confidence)`` method.
            enricher: Object with an ``enrich(diff) -> EnrichedFeatureVector`` method.
            constraints: Optional list of constraint objects; each must expose
                ``penalty(diff) -> float``.

        Returns:
            List of SurrogatePrediction sorted by adjusted_score ascending
            (more negative = better predicted improvement).
        """
        if not diffs:
            return []

        predictions: list[SurrogatePrediction] = []

        for diff in diffs:
            try:
                enriched = enricher.enrich(diff.diff_text, diff_id=diff.diff_id)
                feature_vector = enriched.combined_vector

                prediction_obj = surrogate_model.predict(feature_vector)
                predicted_delta = prediction_obj.predicted_delta
                confidence = prediction_obj.confidence

                # Compute constraint penalty
                constraint_penalty = diff.constraint_penalty
                if constraints:
                    for constraint in constraints:
                        try:
                            constraint_penalty += constraint.penalty(diff)
                        except Exception as exc:
                            logger.warning(
                                "Constraint penalty failed for diff %s: %s",
                                diff.diff_id, exc,
                            )

                adjusted_score = predicted_delta - constraint_penalty

                pred = SurrogatePrediction(
                    diff_id=diff.diff_id,
                    predicted_delta=predicted_delta,
                    confidence=confidence,
                    constraint_penalty=constraint_penalty,
                    adjusted_score=adjusted_score,
                )
                predictions.append(pred)

            except Exception as exc:
                logger.error(
                    "Failed to score diff %s: %s", getattr(diff, "diff_id", "?"), exc
                )

        # Sort ascending by adjusted_score (more negative = better)
        predictions.sort(key=lambda p: p.adjusted_score)

        # Assign ranks
        for rank, pred in enumerate(predictions, start=1):
            pred.rank = rank

        return predictions
