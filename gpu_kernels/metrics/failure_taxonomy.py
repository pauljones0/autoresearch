"""
KernelFailureTaxonomist: extends the failure feature extraction for
kernel-specific modifications (modification_category == "kernel").
Extends the 23-element feature vector to 28 elements.
"""

import math
import re

from ...model_scientist.schemas import JournalEntry, FailureFeatures


# Kernel-specific failure types
_KERNEL_FAILURE_TYPES = [
    "correctness",
    "stability",
    "divergence",
    "performance_regression",
    "marginal",
]

# Kernel-specific feature keys (5 additional features, indices 23-27)
# [target_operation_hash, block_size_normalized, fusion_degree,
#  kernel_failure_type_onehot (5 values compressed to index)]
# Total additional: 5 elements -> 28 total


class KernelFailureTaxonomist:
    """Handle failure feature extraction for kernel-type modifications.

    Extends the standard 23-element feature vector from FailureExtractor
    with 5 kernel-specific features:
      [23] target_operation (hashed to float)
      [24] block_size (normalized: log2(block_size) / 12)
      [25] fusion_degree (number of fused ops, normalized)
      [26] kernel_failure_type (encoded as index / len)
      [27] speedup_ratio (0.0 if unknown)

    The extended vector is backward-compatible: non-kernel entries get
    zero-padding for elements 23-27.
    """

    def extract_kernel_features(self, journal_entry: dict | JournalEntry) -> list[float]:
        """Extract a 28-element feature vector for a kernel journal entry.

        Args:
            journal_entry: JournalEntry or dict with kernel modification data.

        Returns:
            28-element float vector. Elements 0-22 follow the standard layout
            from FailureExtractor. Elements 23-27 are kernel-specific.
        """
        if isinstance(journal_entry, dict):
            entry = JournalEntry.from_dict(journal_entry)
        else:
            entry = journal_entry

        # Build the base 23-element vector
        base_vec = self._extract_base_vector(entry)

        # Extract kernel-specific features
        kernel_features = self._extract_kernel_specific(entry)

        return base_vec + kernel_features

    def _extract_base_vector(self, entry: JournalEntry) -> list[float]:
        """Extract the standard 23-element feature vector."""
        from model_scientist.failure_mining.extractor import (
            FailureExtractor,
            _classify_category,
            _classify_failure_mode,
            _extract_diagnostics_snapshot,
        )

        # Build a FailureFeatures from the journal entry
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
        return FailureExtractor.extract_features_vector(features)

    def _extract_kernel_specific(self, entry: JournalEntry) -> list[float]:
        """Extract the 5 kernel-specific features (indices 23-27)."""
        diff = entry.modification_diff or ""
        diag = entry.diagnostics_summary or {}

        # Feature 23: target_operation hash
        target_op = self._detect_target_operation(diff)
        op_hash = _stable_hash(target_op) if target_op else 0.0

        # Feature 24: block_size normalized
        block_size = self._detect_block_size(diff)
        bs_norm = math.log2(max(block_size, 1)) / 12.0 if block_size > 0 else 0.0

        # Feature 25: fusion_degree
        fusion_degree = self._detect_fusion_degree(diff)
        fd_norm = min(fusion_degree / 10.0, 1.0)

        # Feature 26: kernel failure type
        failure_type = self._classify_kernel_failure(entry)
        ft_index = _KERNEL_FAILURE_TYPES.index(failure_type) if failure_type in _KERNEL_FAILURE_TYPES else 0
        ft_norm = ft_index / max(len(_KERNEL_FAILURE_TYPES) - 1, 1)

        # Feature 27: speedup ratio
        speedup = float(diag.get("kernel_speedup", 0.0))

        return [op_hash, bs_norm, fd_norm, ft_norm, speedup]

    def _detect_target_operation(self, diff: str) -> str:
        """Detect the target operation from a kernel modification diff."""
        # Look for common op patterns in the diff
        patterns = [
            r'(?:fused|kernel|triton)_(\w+)',
            r'target.*?["\'](\w+)["\']',
            r'group_id.*?["\'](\w+)["\']',
        ]
        for pat in patterns:
            match = re.search(pat, diff, re.I)
            if match:
                return match.group(1)
        return ""

    def _detect_block_size(self, diff: str) -> int:
        """Detect BLOCK_SIZE from a kernel diff."""
        match = re.search(r'BLOCK_SIZE\s*=\s*(\d+)', diff)
        if match:
            return int(match.group(1))
        return 0

    def _detect_fusion_degree(self, diff: str) -> int:
        """Detect the number of fused operations from a kernel diff."""
        # Count op references in "Fused ops:" comments
        match = re.search(r'[Ff]used ops?:\s*(.+)', diff)
        if match:
            ops_str = match.group(1)
            return len([x.strip() for x in ops_str.split(",") if x.strip()])
        # Fallback: count tl.load / tl.store pairs as proxy
        loads = len(re.findall(r'tl\.load', diff))
        stores = len(re.findall(r'tl\.store', diff))
        return max(loads, stores)

    def _classify_kernel_failure(self, entry: JournalEntry) -> str:
        """Classify the kernel-specific failure type."""
        diag = entry.diagnostics_summary or {}
        diff = entry.modification_diff or ""
        verdict = (entry.verdict or "").lower()
        diag_str = str(diag).lower()

        # Check for correctness failure
        if "correctness" in diag_str or "mismatch" in diag_str:
            return "correctness"

        # Check for stability
        if "nan" in diag_str or "inf" in diag_str or "overflow" in diag_str:
            return "stability"

        # Check for divergence
        if "diverge" in diag_str or "divergence" in diag_str:
            return "divergence"

        # Performance regression: kernel was slower
        actual_delta = entry.actual_delta or 0.0
        speedup = diag.get("kernel_speedup", 1.0)
        if isinstance(speedup, (int, float)) and speedup < 1.0:
            return "performance_regression"
        if actual_delta > 0:
            return "performance_regression"

        # Marginal: passed but improvement too small
        if 0 < actual_delta < 0.001:
            return "marginal"

        return "marginal"


def _stable_hash(s: str) -> float:
    """Hash a string to a float in [0, 1] deterministically."""
    if not s:
        return 0.0
    h = 0
    for c in s:
        h = (h * 31 + ord(c)) & 0xFFFFFFFF
    return (h % 10000) / 10000.0
