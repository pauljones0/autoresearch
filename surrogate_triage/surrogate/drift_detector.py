"""
Phase 2 — FeatureDriftDetector: detect changes in active metrics
and adapt feature vectors when metrics are promoted or retired.
"""

import logging

logger = logging.getLogger(__name__)


class FeatureDriftDetector:
    """Monitor MetricRegistry for promoted/retired metrics and adapt feature vectors.

    Tracks a baseline set of active metric names. When the active set changes:
    - Retired metrics: zero-fill those dimensions in existing vectors.
    - Promoted metrics: extend input dimension and signal retraining is needed.
    """

    def __init__(self, baseline_metrics: list[str] | None = None):
        self._baseline: list[str] = list(baseline_metrics) if baseline_metrics else []

    @property
    def baseline_metrics(self) -> list[str]:
        return list(self._baseline)

    @baseline_metrics.setter
    def baseline_metrics(self, metrics: list[str]):
        self._baseline = list(metrics)

    def check_drift(
        self,
        current_metrics: list[str],
        baseline_metrics: list[str] | None = None,
    ) -> tuple[bool, dict]:
        """Check if the active metric set has changed relative to baseline.

        Args:
            current_metrics: Currently active metric names.
            baseline_metrics: Override baseline (uses stored baseline if None).

        Returns:
            (drifted, changes) where changes has keys 'retired' and 'promoted'.
        """
        baseline = baseline_metrics if baseline_metrics is not None else self._baseline

        baseline_set = set(baseline)
        current_set = set(current_metrics)

        retired = sorted(baseline_set - current_set)
        promoted = sorted(current_set - baseline_set)

        drifted = bool(retired or promoted)

        if drifted:
            if retired:
                logger.info("Metrics retired: %s", retired)
            if promoted:
                logger.info("Metrics promoted: %s", promoted)

        changes = {
            "retired": retired,
            "promoted": promoted,
            "retired_indices": [baseline.index(m) for m in retired if m in baseline],
            "promoted_names": promoted,
        }

        return drifted, changes

    def adapt_feature_vector(
        self,
        vector: list[float],
        changes: dict,
        code_dim: int = 256,
        failure_dim: int = 23,
    ) -> list[float]:
        """Adapt a feature vector to account for metric drift.

        The feature vector layout is:
            [code_embedding (code_dim)] + [failure_features (failure_dim)] + [metric_features (...)]

        For retired metrics: zero-fill those metric dimensions.
        For promoted metrics: append zeros for the new dimensions.

        Args:
            vector: Original combined feature vector.
            changes: Dict from check_drift with 'retired_indices' and 'promoted_names'.
            code_dim: Dimension of code embedding prefix.
            failure_dim: Dimension of failure feature prefix.

        Returns:
            Adapted feature vector.
        """
        result = list(vector)
        metric_offset = code_dim + failure_dim

        # Zero-fill retired metric dimensions
        for idx in changes.get("retired_indices", []):
            abs_idx = metric_offset + idx
            if abs_idx < len(result):
                result[abs_idx] = 0.0

        # Extend for promoted metrics
        n_promoted = len(changes.get("promoted_names", []))
        if n_promoted > 0:
            result.extend([0.0] * n_promoted)

        return result

    def update_baseline(self, current_metrics: list[str]):
        """Update the stored baseline to the current metric set.

        Call this after retraining the surrogate on the new feature layout.
        """
        old = self._baseline
        self._baseline = list(current_metrics)
        if old != self._baseline:
            logger.info(
                "Baseline updated: %d -> %d metrics",
                len(old), len(self._baseline),
            )
