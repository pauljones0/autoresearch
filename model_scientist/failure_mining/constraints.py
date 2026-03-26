"""
Phase 2.1 — ConstraintGenerator: convert FailurePattern clusters into
natural-language NegativeConstraint objects.
"""

from model_scientist.schemas import FailurePattern, NegativeConstraint
from model_scientist.failure_mining.extractor import _DIAG_KEYS


def _describe_centroid(centroid: dict) -> str:
    """Build a human-readable condition string from a centroid feature dict."""
    raw = centroid.get("raw")
    if not raw or not isinstance(raw, list):
        return "unknown conditions"

    # The centroid vector layout mirrors FailureExtractor.extract_features_vector:
    # [one-hot category (8)] + [diagnostics (9)] + [one-hot failure_mode (4)]
    # + [predicted_delta, actual_delta]
    diag_start = 8
    diag_end = diag_start + len(_DIAG_KEYS)

    if len(raw) < diag_end:
        return "unknown conditions"

    diag_values = raw[diag_start:diag_end]
    parts: list[str] = []
    for key, val in zip(_DIAG_KEYS, diag_values):
        if val == 0.0:
            continue
        label = key.replace("_", " ")
        parts.append(f"{label} ~ {val:.4f}")

    if not parts:
        return "various diagnostic conditions"

    return ", ".join(parts[:4])  # keep it concise


class ConstraintGenerator:
    """Generate NegativeConstraint objects from FailurePattern clusters."""

    def generate(
        self,
        patterns: list[FailurePattern],
        journal_entries: list,
    ) -> list[NegativeConstraint]:
        """Convert patterns into NegativeConstraint objects."""
        if not patterns:
            return []

        constraints: list[NegativeConstraint] = []
        for pat in patterns:
            condition = _describe_centroid(pat.centroid_features)
            text = (
                f"When {condition}, "
                f"{pat.modification_type} modifications have failed "
                f"{pat.instance_count} time(s) "
                f"(avg delta {pat.avg_actual_delta:+.5f}). "
                f"Avoid similar changes under these conditions."
            )

            constraints.append(NegativeConstraint(
                constraint_id=pat.pattern_id,
                pattern_id=pat.pattern_id,
                text=text,
                precision=0.0,
                recall=0.0,
                is_valid=False,
            ))

        return constraints

    @staticmethod
    def format_for_prompt(constraints: list[NegativeConstraint]) -> str:
        """Format validated constraints for injection into a research-agent prompt."""
        if not constraints:
            return ""

        valid = [c for c in constraints if c.is_valid]
        if not valid:
            # Fall back to all constraints if none validated yet.
            valid = constraints

        lines = ["## Negative Constraints (failure patterns to avoid)", ""]
        for c in valid:
            marker = "[validated]" if c.is_valid else "[unvalidated]"
            lines.append(
                f"- {marker} (precision={c.precision:.2f}, recall={c.recall:.2f}) "
                f"{c.text}"
            )
        lines.append("")
        return "\n".join(lines)
