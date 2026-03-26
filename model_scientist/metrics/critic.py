"""
Phase 4: CriticAgent — proposes new diagnostic metrics based on
anomalies and patterns observed in DiagnosticsReport data.
Uses template-based proposal generation (no LLM).
"""

import time
from ..schemas import DiagnosticsReport, MetricDefinition


# ---------------------------------------------------------------------------
# Metric proposal templates
# ---------------------------------------------------------------------------

_TEMPLATES = [
    # --- Gradient health ---
    {
        "trigger": lambda d: _gradient_norm_cv(d) > 0.5,
        "name": "gradient_norm_cv",
        "description": "Coefficient of variation of per-layer gradient norms — high values indicate uneven gradient flow.",
        "computation_method": (
            "norms = [g['norm'] for g in diagnostics.get('gradient_stats', [])]\n"
            "if len(norms) < 2:\n"
            "    result = 0.0\n"
            "else:\n"
            "    mean = sum(norms) / len(norms)\n"
            "    if mean == 0:\n"
            "        result = 0.0\n"
            "    else:\n"
            "        var = sum((x - mean) ** 2 for x in norms) / len(norms)\n"
            "        result = var ** 0.5 / mean\n"
        ),
        "rationale": "Wide gradient norm variance across layers suggests vanishing/exploding gradient issues.",
    },
    {
        "trigger": lambda d: _max_dead_frac(d) > 0.3,
        "name": "max_layer_dead_gradient_fraction",
        "description": "Maximum dead-gradient fraction across all layers.",
        "computation_method": (
            "fracs = [g['dead_fraction'] for g in diagnostics.get('gradient_stats', [])]\n"
            "result = max(fracs) if fracs else 0.0\n"
        ),
        "rationale": "A layer with many dead gradients is a bottleneck for learning.",
    },
    {
        "trigger": lambda d: _gradient_norm_range(d) > 5.0,
        "name": "gradient_norm_range_ratio",
        "description": "Ratio of max to min gradient norm across layers.",
        "computation_method": (
            "norms = [g['norm'] for g in diagnostics.get('gradient_stats', [])]\n"
            "if len(norms) < 2 or min(norms) == 0:\n"
            "    result = 0.0\n"
            "else:\n"
            "    result = max(norms) / min(norms)\n"
        ),
        "rationale": "Extreme ratio signals gradient imbalance between layers.",
    },
    # --- Attention patterns ---
    {
        "trigger": lambda d: _attention_entropy_cv(d) > 0.4,
        "name": "attention_head_diversity_index",
        "description": "Standard deviation of attention entropy across all heads — measures head specialization diversity.",
        "computation_method": (
            "ents = [a['entropy'] for a in diagnostics.get('attention_stats', [])]\n"
            "if len(ents) < 2:\n"
            "    result = 0.0\n"
            "else:\n"
            "    mean = sum(ents) / len(ents)\n"
            "    var = sum((x - mean) ** 2 for x in ents) / len(ents)\n"
            "    result = var ** 0.5\n"
        ),
        "rationale": "Diverse attention entropy means heads are specializing differently, which is typically healthy.",
    },
    {
        "trigger": lambda d: _mean_collapse_score(d) > 0.5,
        "name": "attention_collapse_fraction",
        "description": "Fraction of attention heads with collapse_score > 0.7.",
        "computation_method": (
            "scores = [a['collapse_score'] for a in diagnostics.get('attention_stats', [])]\n"
            "if not scores:\n"
            "    result = 0.0\n"
            "else:\n"
            "    result = sum(1 for s in scores if s > 0.7) / len(scores)\n"
        ),
        "rationale": "High collapse fraction means many heads attend uniformly — wasted capacity.",
    },
    {
        "trigger": lambda d: _min_attention_entropy(d) < 0.5,
        "name": "min_attention_entropy",
        "description": "Minimum attention entropy across all heads — detects extremely peaked attention.",
        "computation_method": (
            "ents = [a['entropy'] for a in diagnostics.get('attention_stats', [])]\n"
            "result = min(ents) if ents else 0.0\n"
        ),
        "rationale": "Very low entropy in any head can indicate degenerate attention patterns.",
    },
    # --- Loss landscape ---
    {
        "trigger": lambda d: _loss_bucket_range(d) > 1.0,
        "name": "loss_bucket_disparity",
        "description": "Range of mean loss across token-frequency buckets.",
        "computation_method": (
            "losses = [b['mean_loss'] for b in diagnostics.get('loss_decomposition', [])]\n"
            "if len(losses) < 2:\n"
            "    result = 0.0\n"
            "else:\n"
            "    result = max(losses) - min(losses)\n"
        ),
        "rationale": "Large disparity means the model is much worse on some token frequency ranges.",
    },
    {
        "trigger": lambda d: _loss_std_mean(d) > 0.5,
        "name": "loss_std_mean_ratio",
        "description": "Mean of per-bucket loss standard deviations — measures loss volatility.",
        "computation_method": (
            "stds = [b['std_loss'] for b in diagnostics.get('loss_decomposition', [])]\n"
            "result = sum(stds) / len(stds) if stds else 0.0\n"
        ),
        "rationale": "High loss std within buckets indicates inconsistent predictions.",
    },
    # --- Training dynamics ---
    {
        "trigger": lambda d: _activation_std_cv(d) > 0.5,
        "name": "activation_std_cv",
        "description": "Coefficient of variation of per-layer activation std — measures representation stability.",
        "computation_method": (
            "stds = [a['std'] for a in diagnostics.get('activation_stats', [])]\n"
            "if len(stds) < 2:\n"
            "    result = 0.0\n"
            "else:\n"
            "    mean = sum(stds) / len(stds)\n"
            "    if mean == 0:\n"
            "        result = 0.0\n"
            "    else:\n"
            "        var = sum((x - mean) ** 2 for x in stds) / len(stds)\n"
            "        result = var ** 0.5 / mean\n"
        ),
        "rationale": "Unstable activation magnitudes across layers may hinder training.",
    },
    {
        "trigger": lambda d: _max_activation_abs(d) > 50.0,
        "name": "max_activation_magnitude",
        "description": "Maximum absolute activation value across all layers.",
        "computation_method": (
            "vals = [a['max_abs'] for a in diagnostics.get('activation_stats', [])]\n"
            "result = max(vals) if vals else 0.0\n"
        ),
        "rationale": "Extremely large activations can cause numerical instability.",
    },
    # --- Representation quality ---
    {
        "trigger": lambda d: _dead_neuron_total(d) > 0.1,
        "name": "total_dead_neuron_fraction",
        "description": "Average dead neuron fraction across all layers.",
        "computation_method": (
            "fracs = [a['dead_neuron_fraction'] for a in diagnostics.get('activation_stats', [])]\n"
            "result = sum(fracs) / len(fracs) if fracs else 0.0\n"
        ),
        "rationale": "High dead neuron count across layers wastes model capacity.",
    },
    {
        "trigger": lambda d: _layer_similarity_high(d),
        "name": "mean_adjacent_layer_similarity",
        "description": "Mean CKA similarity between adjacent layers — high values suggest redundant layers.",
        "computation_method": (
            "entries = diagnostics.get('layer_similarity_matrix', [])\n"
            "adj = [e['cka_score'] for e in entries if abs(e['layer_i'] - e['layer_j']) == 1]\n"
            "result = sum(adj) / len(adj) if adj else 0.0\n"
        ),
        "rationale": "Adjacent layers with very similar representations may be redundant.",
    },
]


# ---------------------------------------------------------------------------
# Trigger helper functions (operate on dict-form diagnostics)
# ---------------------------------------------------------------------------

def _as_dict(d):
    """Convert DiagnosticsReport to dict if needed."""
    if isinstance(d, DiagnosticsReport):
        return d.to_dict()
    return d


def _gradient_norm_cv(d):
    d = _as_dict(d)
    norms = [g["norm"] for g in d.get("gradient_stats", [])]
    if len(norms) < 2:
        return 0.0
    mean = sum(norms) / len(norms)
    if mean == 0:
        return 0.0
    var = sum((x - mean) ** 2 for x in norms) / len(norms)
    return var ** 0.5 / mean


def _max_dead_frac(d):
    d = _as_dict(d)
    fracs = [g["dead_fraction"] for g in d.get("gradient_stats", [])]
    return max(fracs) if fracs else 0.0


def _gradient_norm_range(d):
    d = _as_dict(d)
    norms = [g["norm"] for g in d.get("gradient_stats", [])]
    if len(norms) < 2 or min(norms) == 0:
        return 0.0
    return max(norms) / min(norms)


def _attention_entropy_cv(d):
    d = _as_dict(d)
    ents = [a["entropy"] for a in d.get("attention_stats", [])]
    if len(ents) < 2:
        return 0.0
    mean = sum(ents) / len(ents)
    if mean == 0:
        return 0.0
    var = sum((x - mean) ** 2 for x in ents) / len(ents)
    return var ** 0.5 / mean


def _mean_collapse_score(d):
    d = _as_dict(d)
    scores = [a["collapse_score"] for a in d.get("attention_stats", [])]
    return sum(scores) / len(scores) if scores else 0.0


def _min_attention_entropy(d):
    d = _as_dict(d)
    ents = [a["entropy"] for a in d.get("attention_stats", [])]
    return min(ents) if ents else float("inf")


def _loss_bucket_range(d):
    d = _as_dict(d)
    losses = [b["mean_loss"] for b in d.get("loss_decomposition", [])]
    if len(losses) < 2:
        return 0.0
    return max(losses) - min(losses)


def _loss_std_mean(d):
    d = _as_dict(d)
    stds = [b["std_loss"] for b in d.get("loss_decomposition", [])]
    return sum(stds) / len(stds) if stds else 0.0


def _activation_std_cv(d):
    d = _as_dict(d)
    stds = [a["std"] for a in d.get("activation_stats", [])]
    if len(stds) < 2:
        return 0.0
    mean = sum(stds) / len(stds)
    if mean == 0:
        return 0.0
    var = sum((x - mean) ** 2 for x in stds) / len(stds)
    return var ** 0.5 / mean


def _max_activation_abs(d):
    d = _as_dict(d)
    vals = [a["max_abs"] for a in d.get("activation_stats", [])]
    return max(vals) if vals else 0.0


def _dead_neuron_total(d):
    d = _as_dict(d)
    fracs = [a["dead_neuron_fraction"] for a in d.get("activation_stats", [])]
    return sum(fracs) / len(fracs) if fracs else 0.0


def _layer_similarity_high(d):
    d = _as_dict(d)
    entries = d.get("layer_similarity_matrix", [])
    adj = [e["cka_score"] for e in entries if abs(e["layer_i"] - e["layer_j"]) == 1]
    if not adj:
        return False
    return sum(adj) / len(adj) > 0.8


# ---------------------------------------------------------------------------
# CriticAgent
# ---------------------------------------------------------------------------

class CriticAgent:
    """Proposes new diagnostic metrics via template-based analysis."""

    def __init__(self, max_proposals: int = 3):
        self.max_proposals = max_proposals

    def propose_metrics(
        self,
        diagnostics: DiagnosticsReport,
        existing_metrics: list,
    ) -> list:
        """Propose 1-3 new metrics based on patterns in the diagnostics report.

        Args:
            diagnostics: Current DiagnosticsReport.
            existing_metrics: List of MetricDefinition (or dicts) already known.

        Returns:
            List of MetricDefinition proposals.
        """
        existing_names = {
            self._normalize_name(m.name if isinstance(m, MetricDefinition) else m.get("name", ""))
            for m in existing_metrics
        }

        proposals: list[MetricDefinition] = []

        for tmpl in _TEMPLATES:
            if len(proposals) >= self.max_proposals:
                break

            norm_name = self._normalize_name(tmpl["name"])

            # Dedup: skip if a similar metric already exists
            if self._is_duplicate(norm_name, existing_names):
                continue

            # Check whether the trigger condition fires
            try:
                if not tmpl["trigger"](diagnostics):
                    continue
            except (KeyError, TypeError, IndexError, ZeroDivisionError):
                continue

            proposals.append(MetricDefinition(
                name=tmpl["name"],
                description=tmpl["description"],
                computation_method=tmpl["computation_method"],
                rationale=tmpl["rationale"],
                source="critic",
                created_at=time.time(),
                status="candidate",
            ))
            existing_names.add(norm_name)

        return proposals

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_name(name: str) -> str:
        return name.lower().replace("-", "_").replace(" ", "_")

    @staticmethod
    def _is_duplicate(candidate_name: str, existing_names: set) -> bool:
        """Check for exact match or high token overlap."""
        if candidate_name in existing_names:
            return True
        cand_tokens = set(candidate_name.split("_"))
        for existing in existing_names:
            exist_tokens = set(existing.split("_"))
            if not cand_tokens or not exist_tokens:
                continue
            overlap = len(cand_tokens & exist_tokens)
            max_len = max(len(cand_tokens), len(exist_tokens))
            if max_len > 0 and overlap / max_len >= 0.8:
                return True
        return False
