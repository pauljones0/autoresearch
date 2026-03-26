"""
Phase 4: MetricRegistry — persists the catalog of all metrics
(hardcoded and critic-proposed) to metric_registry.json.
"""

import json
import os
import time

from ..schemas import MetricDefinition


# ---------------------------------------------------------------------------
# Hardcoded base metrics
# ---------------------------------------------------------------------------

_BASE_METRICS = [
    MetricDefinition(
        name="gradient_norm_mean",
        description="Mean gradient norm across all layers.",
        computation_method=(
            "norms = [g['norm'] for g in diagnostics.get('gradient_stats', [])]\n"
            "result = sum(norms) / len(norms) if norms else 0.0\n"
        ),
        rationale="Overall gradient magnitude is a basic training health signal.",
        source="hardcoded",
        status="active",
    ),
    MetricDefinition(
        name="activation_std_mean",
        description="Mean activation standard deviation across all layers.",
        computation_method=(
            "stds = [a['std'] for a in diagnostics.get('activation_stats', [])]\n"
            "result = sum(stds) / len(stds) if stds else 0.0\n"
        ),
        rationale="Activation magnitudes indicate representation health.",
        source="hardcoded",
        status="active",
    ),
    MetricDefinition(
        name="attention_entropy_mean",
        description="Mean attention entropy across all heads.",
        computation_method=(
            "ents = [a['entropy'] for a in diagnostics.get('attention_stats', [])]\n"
            "result = sum(ents) / len(ents) if ents else 0.0\n"
        ),
        rationale="Attention entropy summarises how focused or distributed attention is.",
        source="hardcoded",
        status="active",
    ),
    MetricDefinition(
        name="loss_decomposition_variance",
        description="Variance of mean loss across token-frequency buckets.",
        computation_method=(
            "losses = [b['mean_loss'] for b in diagnostics.get('loss_decomposition', [])]\n"
            "if len(losses) < 2:\n"
            "    result = 0.0\n"
            "else:\n"
            "    m = sum(losses) / len(losses)\n"
            "    result = sum((x - m) ** 2 for x in losses) / len(losses)\n"
        ),
        rationale="High variance means the model treats frequency buckets very differently.",
        source="hardcoded",
        status="active",
    ),
    MetricDefinition(
        name="dead_neuron_fraction",
        description="Average dead-neuron fraction across activation layers.",
        computation_method=(
            "fracs = [a['dead_neuron_fraction'] for a in diagnostics.get('activation_stats', [])]\n"
            "result = sum(fracs) / len(fracs) if fracs else 0.0\n"
        ),
        rationale="Dead neurons represent wasted capacity.",
        source="hardcoded",
        status="active",
    ),
]


# ---------------------------------------------------------------------------
# MetricRegistry
# ---------------------------------------------------------------------------

class MetricRegistry:
    """Manages the metric catalog, persisted to a JSON file."""

    def __init__(self, path: str = "metric_registry.json"):
        self.path = path
        self._metrics: dict[str, dict] = {}  # name -> serialized MetricDefinition dict

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self):
        with open(self.path, "w") as f:
            json.dump(list(self._metrics.values()), f, indent=2)

    def load(self):
        if not os.path.exists(self.path):
            self._metrics = {}
            self.initialize_defaults()
            return
        with open(self.path) as f:
            entries = json.load(f)
        self._metrics = {e["name"]: e for e in entries}

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def initialize_defaults(self):
        """Seed registry with hardcoded base metrics if they are missing."""
        for m in _BASE_METRICS:
            if m.name not in self._metrics:
                self._metrics[m.name] = self._to_dict(m)

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def add(self, metric: MetricDefinition):
        """Register a new metric (overwrites if name exists)."""
        self._metrics[metric.name] = self._to_dict(metric)

    def get_active(self) -> list:
        """Return MetricDefinitions with status 'active'."""
        return [self._to_metric(v) for v in self._metrics.values() if v.get("status") == "active"]

    def get_candidates(self) -> list:
        """Return MetricDefinitions with status 'candidate'."""
        return [self._to_metric(v) for v in self._metrics.values() if v.get("status") == "candidate"]

    def get_all(self) -> list:
        """Return all MetricDefinitions."""
        return [self._to_metric(v) for v in self._metrics.values()]

    def get(self, name: str):
        """Return a single MetricDefinition or None."""
        entry = self._metrics.get(name)
        return self._to_metric(entry) if entry else None

    def promote(self, name: str):
        """Move a metric from candidate to active."""
        entry = self._metrics.get(name)
        if entry:
            entry["status"] = "active"
            entry["consecutive_low_correlation_cycles"] = 0

    def retire(self, name: str):
        """Move a metric to retired."""
        entry = self._metrics.get(name)
        if entry:
            entry["status"] = "retired"

    def update_correlation(self, name: str, correlation: float):
        """Update the stored correlation value for a metric."""
        entry = self._metrics.get(name)
        if entry:
            entry["correlation_with_success"] = correlation

    def increment_low_correlation(self, name: str):
        """Increment the consecutive low-correlation cycle counter."""
        entry = self._metrics.get(name)
        if entry:
            entry["consecutive_low_correlation_cycles"] = entry.get("consecutive_low_correlation_cycles", 0) + 1

    def reset_low_correlation(self, name: str):
        """Reset the consecutive low-correlation cycle counter."""
        entry = self._metrics.get(name)
        if entry:
            entry["consecutive_low_correlation_cycles"] = 0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_dict(m: MetricDefinition) -> dict:
        return {
            "name": m.name,
            "description": m.description,
            "computation_method": m.computation_method,
            "rationale": m.rationale,
            "source": m.source,
            "created_at": m.created_at,
            "status": m.status,
            "correlation_with_success": m.correlation_with_success,
            "consecutive_low_correlation_cycles": m.consecutive_low_correlation_cycles,
        }

    @staticmethod
    def _to_metric(d: dict) -> MetricDefinition:
        return MetricDefinition(
            name=d.get("name", ""),
            description=d.get("description", ""),
            computation_method=d.get("computation_method", ""),
            rationale=d.get("rationale", ""),
            source=d.get("source", ""),
            created_at=d.get("created_at", 0.0),
            status=d.get("status", "candidate"),
            correlation_with_success=d.get("correlation_with_success", 0.0),
            consecutive_low_correlation_cycles=d.get("consecutive_low_correlation_cycles", 0),
        )
