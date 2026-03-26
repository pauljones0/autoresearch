"""
KernelConstraintSpecialist: generates kernel-specific constraint templates
from failure patterns and kernel journal entries.
"""

from model_scientist.schemas import FailurePattern, JournalEntry


class KernelConstraintSpecialist:
    """Generate kernel-specific negative constraints from failure data.

    Produces constraints following the pattern:
      "When targeting {op_type} with block_size {range} on {hardware},
       {failure_type} failures occurred {count}/{total} times."
    """

    def generate_kernel_constraints(
        self,
        failure_patterns: list[FailurePattern | dict],
        kernel_entries: list[JournalEntry | dict],
    ) -> list[dict]:
        """Generate kernel-specific constraint objects.

        Args:
            failure_patterns: Clustered failure patterns from failure mining.
            kernel_entries: Kernel-related journal entries (rejected/crashed).

        Returns:
            List of constraint dicts with keys: constraint_id, text,
            op_type, block_size_range, failure_type, failure_count,
            total_count, precision, penalty.
        """
        # Normalize inputs
        patterns = [_normalize_pattern(p) for p in failure_patterns]
        entries = [_normalize_entry(e) for e in kernel_entries]

        if not entries:
            return []

        # Group entries by characteristics
        by_op = _group_by_key(entries, "target_op")
        by_block_size = _group_by_key(entries, "block_size_bucket")
        by_failure = _group_by_key(entries, "failure_type")

        constraints: list[dict] = []
        constraint_id = 0

        # Generate constraints for op_type + failure_type combinations
        for op_type, op_entries in by_op.items():
            if not op_type:
                continue
            total = len(op_entries)
            failure_counts: dict[str, int] = {}
            for e in op_entries:
                ft = e.get("failure_type", "unknown")
                failure_counts[ft] = failure_counts.get(ft, 0) + 1

            for failure_type, count in failure_counts.items():
                if count < 2:
                    continue
                precision = count / total if total > 0 else 0.0
                if precision < 0.3:
                    continue

                # Detect block_size range from entries
                block_sizes = [
                    e.get("block_size", 0) for e in op_entries
                    if e.get("failure_type") == failure_type and e.get("block_size", 0) > 0
                ]
                bs_range = _format_range(block_sizes)
                hardware = _most_common(
                    [e.get("hardware", "") for e in op_entries if e.get("hardware")]
                )

                text = _format_constraint_text(
                    op_type, bs_range, hardware, failure_type, count, total
                )

                constraints.append({
                    "constraint_id": constraint_id,
                    "text": text,
                    "op_type": op_type,
                    "block_size_range": bs_range,
                    "failure_type": failure_type,
                    "failure_count": count,
                    "total_count": total,
                    "precision": precision,
                    "penalty": min(0.8, precision),
                })
                constraint_id += 1

        # Generate constraints from failure patterns
        for pat in patterns:
            mod_type = pat.get("modification_type", "")
            if "kernel" not in mod_type.lower() and mod_type != "":
                continue
            count = pat.get("instance_count", 0)
            if count < 2:
                continue

            desc = pat.get("description", "")
            centroid = pat.get("centroid_features", {})

            text = f"Pattern: {desc}. Occurred {count} times."
            constraints.append({
                "constraint_id": constraint_id,
                "text": text,
                "op_type": centroid.get("op_type", ""),
                "block_size_range": "",
                "failure_type": centroid.get("failure_mode", ""),
                "failure_count": count,
                "total_count": count,
                "precision": 1.0,
                "penalty": min(0.8, count / max(count + 2, 1)),
            })
            constraint_id += 1

        return constraints


def _normalize_pattern(p) -> dict:
    """Convert FailurePattern or dict to dict."""
    if isinstance(p, dict):
        return p
    return {
        "pattern_id": p.pattern_id,
        "description": p.description,
        "modification_type": p.modification_type,
        "instance_count": p.instance_count,
        "centroid_features": p.centroid_features,
        "avg_actual_delta": p.avg_actual_delta,
    }


def _normalize_entry(e) -> dict:
    """Normalize a journal entry to a dict with kernel-relevant keys."""
    if isinstance(e, JournalEntry):
        d = e.to_dict()
    elif isinstance(e, dict):
        d = e
    else:
        d = {}

    # Extract kernel-specific fields from the diff/tags
    diff = d.get("modification_diff", "")
    diag = d.get("diagnostics_summary", {})

    import re
    block_match = re.search(r'BLOCK_SIZE\s*=\s*(\d+)', diff)
    block_size = int(block_match.group(1)) if block_match else 0

    op_match = re.search(r'(?:fused|kernel|group_id).*?["\'](\w+)', diff, re.I)
    target_op = op_match.group(1) if op_match else ""

    # Bucket block sizes
    if block_size <= 128:
        bs_bucket = "small"
    elif block_size <= 512:
        bs_bucket = "medium"
    else:
        bs_bucket = "large"

    # Classify failure type
    failure_type = "unknown"
    diag_str = str(diag).lower()
    if "correctness" in diag_str or "mismatch" in diag_str:
        failure_type = "correctness"
    elif "nan" in diag_str or "inf" in diag_str:
        failure_type = "stability"
    elif "diverge" in diag_str:
        failure_type = "divergence"
    elif d.get("actual_delta", 0) > 0:
        failure_type = "performance_regression"

    return {
        "target_op": target_op,
        "block_size": block_size,
        "block_size_bucket": bs_bucket,
        "failure_type": failure_type,
        "hardware": diag.get("hardware", "") if isinstance(diag, dict) else "",
        "verdict": d.get("verdict", ""),
    }


def _group_by_key(entries: list[dict], key: str) -> dict[str, list[dict]]:
    """Group entries by a key value."""
    groups: dict[str, list[dict]] = {}
    for e in entries:
        val = e.get(key, "")
        if val not in groups:
            groups[val] = []
        groups[val].append(e)
    return groups


def _format_range(values: list[int]) -> str:
    """Format a list of integers as a range string."""
    if not values:
        return ""
    mn, mx = min(values), max(values)
    if mn == mx:
        return str(mn)
    return f"{mn}-{mx}"


def _most_common(values: list[str]) -> str:
    """Return the most common non-empty string in a list."""
    if not values:
        return ""
    counts: dict[str, int] = {}
    for v in values:
        if v:
            counts[v] = counts.get(v, 0) + 1
    if not counts:
        return ""
    return max(counts, key=counts.get)


def _format_constraint_text(
    op_type: str,
    bs_range: str,
    hardware: str,
    failure_type: str,
    count: int,
    total: int,
) -> str:
    """Format a constraint as human-readable text."""
    parts = [f"When targeting {op_type}"]
    if bs_range:
        parts.append(f"with block_size {bs_range}")
    if hardware:
        parts.append(f"on {hardware}")
    parts.append(f"{failure_type} failures occurred {count}/{total} times.")
    return " ".join(parts) + " Avoid this configuration."
