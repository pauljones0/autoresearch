"""
Phase 2.1 — FailureExtractor: parse hypothesis journal for rejected/crashed
experiments and extract structured FailureFeatures.
"""

import math
from model_scientist.schemas import (
    FailureFeatures,
    JournalEntry,
    load_jsonl,
)

# Keyword sets used to classify modification diffs.
_CATEGORY_KEYWORDS: dict[str, list[str]] = {
    "architecture": [
        "n_head", "n_embd", "n_layer", "depth", "width", "embed",
        "attention", "mlp", "feedforward", "residual", "skip", "block",
        "transformer", "head", "layer_norm", "groupnorm", "rmsnorm",
    ],
    "optimizer": [
        "optimizer", "adam", "sgd", "momentum", "weight_decay",
        "learning_rate", "lr", "betas", "eps", "gradient_clip",
        "grad_clip", "warmup",
    ],
    "hyperparameter": [
        "batch_size", "sequence_length", "seq_len", "max_steps",
        "tokens", "accumulation", "micro_batch",
    ],
    "activation": [
        "relu", "gelu", "silu", "swish", "tanh", "sigmoid",
        "activation", "softmax", "softplus", "mish",
    ],
    "initialization": [
        "init", "xavier", "kaiming", "normal_", "uniform_",
        "trunc_normal", "orthogonal", "zeros_", "ones_",
    ],
    "regularization": [
        "dropout", "drop_path", "label_smooth", "weight_norm",
        "spectral_norm", "l2_reg", "l1_reg",
    ],
    "scheduling": [
        "scheduler", "cosine", "linear_decay", "step_lr",
        "warmup_steps", "cooldown", "cycle", "one_cycle",
    ],
}


def _classify_category(diff_text: str) -> str:
    """Return the best-matching modification category for *diff_text*."""
    if not diff_text:
        return "other"

    diff_lower = diff_text.lower()
    scores: dict[str, int] = {}
    for category, keywords in _CATEGORY_KEYWORDS.items():
        scores[category] = sum(1 for kw in keywords if kw in diff_lower)

    best = max(scores, key=scores.get)  # type: ignore[arg-type]
    if scores[best] == 0:
        return "other"
    return best


def _classify_failure_mode(entry: JournalEntry) -> str:
    """Classify the failure mode of a rejected/crashed journal entry."""
    verdict = (entry.verdict or "").lower()

    if verdict == "crashed":
        return "crash"

    # Check diagnostics for NaN / Inf signals.
    diag = entry.diagnostics_summary or {}
    diag_str = str(diag).lower()
    if "nan" in diag_str or "inf" in diag_str:
        return "instability"

    delta = entry.actual_delta
    if delta is None:
        return "crash"

    # actual_delta is change in val_bpb; positive means it got worse.
    if abs(delta) < 1e-4:
        return "no_change"
    if delta > 0:
        return "regression"

    # Negative delta but still rejected — treat as no meaningful change.
    return "no_change"


def _extract_diagnostics_snapshot(diag: dict) -> dict:
    """Pull out a compact snapshot of key diagnostics metrics."""
    snapshot: dict = {}
    for key in (
        "val_bpb",
        "train_loss",
        "grad_norm_mean",
        "grad_norm_max",
        "grad_dead_fraction",
        "activation_mean",
        "activation_std",
        "attention_entropy_mean",
        "attention_collapse_mean",
    ):
        if key in diag:
            snapshot[key] = diag[key]

    # Flatten nested gradient stats if present.
    if "gradient_stats" in diag and isinstance(diag["gradient_stats"], list):
        norms = [
            g.get("norm", 0.0) if isinstance(g, dict) else 0.0
            for g in diag["gradient_stats"]
        ]
        if norms:
            snapshot.setdefault("grad_norm_mean", sum(norms) / len(norms))
            snapshot.setdefault("grad_norm_max", max(norms))

    if "activation_stats" in diag and isinstance(diag["activation_stats"], list):
        means = [
            a.get("mean", 0.0) if isinstance(a, dict) else 0.0
            for a in diag["activation_stats"]
        ]
        if means:
            snapshot.setdefault("activation_mean", sum(means) / len(means))

    return snapshot


# Canonical feature-vector keys and their defaults.
_CATEGORY_LIST = [
    "architecture", "optimizer", "hyperparameter", "activation",
    "initialization", "regularization", "scheduling", "other",
]
_FAILURE_MODE_LIST = ["regression", "instability", "no_change", "crash"]
_DIAG_KEYS = [
    "val_bpb", "train_loss", "grad_norm_mean", "grad_norm_max",
    "grad_dead_fraction", "activation_mean", "activation_std",
    "attention_entropy_mean", "attention_collapse_mean",
]


class FailureExtractor:
    """Extract structured FailureFeatures from hypothesis journal entries."""

    def extract(self, journal_path: str) -> list[FailureFeatures]:
        """Parse *journal_path* (JSONL) and return features for failures."""
        raw = load_jsonl(journal_path)
        if not raw:
            return []

        results: list[FailureFeatures] = []
        for data in raw:
            entry = JournalEntry.from_dict(data)
            if entry.verdict not in ("rejected", "crashed"):
                continue

            features = FailureFeatures(
                journal_id=entry.id,
                modification_category=_classify_category(entry.modification_diff),
                diagnostics_snapshot=_extract_diagnostics_snapshot(
                    entry.diagnostics_summary or {}
                ),
                predicted_delta=entry.predicted_delta or 0.0,
                actual_delta=entry.actual_delta or 0.0,
                failure_mode=_classify_failure_mode(entry),
            )
            results.append(features)

        return results

    @staticmethod
    def extract_features_vector(failure: FailureFeatures) -> list[float]:
        """Convert a FailureFeatures into a flat numerical vector.

        Layout:
          [one-hot category (8)] + [diagnostics (9)] + [one-hot failure_mode (4)]
          + [predicted_delta, actual_delta]
        Total length: 23
        """
        vec: list[float] = []

        # One-hot modification_category.
        for cat in _CATEGORY_LIST:
            vec.append(1.0 if failure.modification_category == cat else 0.0)

        # Numerical diagnostics.
        snap = failure.diagnostics_snapshot or {}
        for key in _DIAG_KEYS:
            val = snap.get(key)
            if val is None or (isinstance(val, float) and (math.isnan(val) or math.isinf(val))):
                vec.append(0.0)
            else:
                vec.append(float(val))

        # One-hot failure_mode.
        for fm in _FAILURE_MODE_LIST:
            vec.append(1.0 if failure.failure_mode == fm else 0.0)

        # Predicted / actual deltas.
        vec.append(float(failure.predicted_delta or 0.0))
        vec.append(float(failure.actual_delta or 0.0))

        return vec
