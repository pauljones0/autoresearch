"""
Phase 2 — FeatureEnricher: combine code embeddings, failure features,
and active metric values into a single enriched feature vector.
"""

import logging

from surrogate_triage.schemas import EnrichedFeatureVector
from surrogate_triage.surrogate.diff_embedder import DiffEmbedder
from model_scientist.schemas import FailureFeatures
from model_scientist.failure_mining.extractor import FailureExtractor

logger = logging.getLogger(__name__)

# Default dimensions
_CODE_DIM = 256
_FAILURE_DIM = 23  # FailureExtractor.extract_features_vector output length


class FeatureEnricher:
    """Combine three feature sources into the surrogate's input vector.

    Sources:
        1. Code embedding from DiffEmbedder (dim=256)
        2. Failure feature vector from FailureExtractor (23 elements)
        3. Current values of active metrics from MetricRegistry
    """

    def __init__(self, code_dim: int = _CODE_DIM):
        self._embedder = DiffEmbedder()
        self._code_dim = code_dim

    def enrich(
        self,
        diff_text: str,
        diagnostics_snapshot: dict | None = None,
        metric_registry=None,
        diff_id: str = "",
    ) -> EnrichedFeatureVector:
        """Build an enriched feature vector for a candidate diff.

        Args:
            diff_text: Unified diff text.
            diagnostics_snapshot: Optional diagnostics dict for failure features.
            metric_registry: Optional MetricRegistry instance for active metric values.
            diff_id: Optional identifier for the diff.

        Returns:
            EnrichedFeatureVector with all component vectors and the concatenation.
        """
        # 1. Code embedding
        code_embedding = self._embedder.embed(diff_text, dim=self._code_dim)

        # 2. Failure features from diagnostics snapshot
        failure_features = self._build_failure_features(diagnostics_snapshot)

        # 3. Metric features from registry
        metric_features = self._build_metric_features(metric_registry)

        # Concatenate
        combined = code_embedding + failure_features + metric_features

        return EnrichedFeatureVector(
            diff_id=diff_id,
            code_embedding=code_embedding,
            failure_features=failure_features,
            metric_features=metric_features,
            combined_vector=combined,
        )

    def get_feature_dim(self, metric_registry=None) -> int:
        """Return the total feature vector dimension.

        Args:
            metric_registry: Optional MetricRegistry to count active metrics.

        Returns:
            Total dimension of the combined vector.
        """
        n_metrics = 0
        if metric_registry is not None:
            try:
                n_metrics = len(metric_registry.get_active())
            except Exception:
                pass
        return self._code_dim + _FAILURE_DIM + n_metrics

    @staticmethod
    def _build_failure_features(diagnostics_snapshot: dict | None) -> list[float]:
        """Extract failure feature vector from a diagnostics snapshot.

        Uses FailureExtractor.extract_features_vector on a synthetic FailureFeatures.
        Returns zero vector if diagnostics are unavailable.
        """
        if not diagnostics_snapshot:
            return [0.0] * _FAILURE_DIM

        try:
            # Build a minimal FailureFeatures to extract the vector
            ff = FailureFeatures(
                diagnostics_snapshot=diagnostics_snapshot,
                # Category and failure mode will be "other"/"no_change" by default
                modification_category="other",
                failure_mode="no_change",
            )
            vec = FailureExtractor.extract_features_vector(ff)
            if len(vec) != _FAILURE_DIM:
                logger.warning(
                    "Unexpected failure feature dim: got %d, expected %d",
                    len(vec), _FAILURE_DIM,
                )
                # Pad or truncate
                vec = (vec + [0.0] * _FAILURE_DIM)[:_FAILURE_DIM]
            return vec
        except Exception as exc:
            logger.warning("Failed to extract failure features: %s", exc)
            return [0.0] * _FAILURE_DIM

    @staticmethod
    def _build_metric_features(metric_registry) -> list[float]:
        """Extract current metric values from the registry.

        Returns a float per active metric (the stored correlation_with_success
        value, which reflects how predictive each metric is).
        """
        if metric_registry is None:
            return []

        try:
            active = metric_registry.get_active()
            # Sort by name for deterministic ordering
            active.sort(key=lambda m: m.name)
            return [float(m.correlation_with_success) for m in active]
        except Exception as exc:
            logger.warning("Failed to extract metric features: %s", exc)
            return []
