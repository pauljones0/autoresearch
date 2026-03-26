"""
Elementwise target selector: picks top-2 elementwise fusion targets from fuseable groups.
"""

import re
from typing import List

import sys, os
from ..schemas import KernelTarget, FuseableGroup


class ElementwiseTargetSelector:
    """Select the top-2 elementwise fusion targets from fuseable groups,
    ranked by combined_gpu_time_us. Validates targets against current train.py source."""

    def _validate_target_in_source(self, group: FuseableGroup, train_source: str) -> bool:
        """Verify that the operations in this group still exist in train.py."""
        # Map common op names to patterns we expect in train.py
        op_patterns = {
            "relu": r"F\.relu|\.relu\(",
            "square": r"\.square\(\)",
            "rms_norm": r"F\.rms_norm|rms_norm",
            "tanh": r"torch\.tanh|\.tanh\(",
            "sigmoid": r"torch\.sigmoid|\.sigmoid\(",
            "softmax": r"F\.softmax|\.softmax\(",
            "mul": r"\*",
            "add": r"\+",
            "cross_entropy": r"F\.cross_entropy",
            "embedding": r"nn\.Embedding",
            "linear": r"nn\.Linear",
            "softcap": r"softcap",
        }
        for op_name in group.op_names:
            op_key = op_name.lower().replace("aten::", "").replace("_", "")
            matched = False
            for pattern_key, pattern in op_patterns.items():
                if pattern_key.replace("_", "") in op_key or op_key in pattern_key.replace("_", ""):
                    if re.search(pattern, train_source):
                        matched = True
                        break
            # If we can't find a specific pattern, check if any token from the op
            # name appears in the source (loose validation)
            if not matched:
                tokens = op_name.lower().replace("aten::", "").split("_")
                for token in tokens:
                    if len(token) > 2 and token in train_source.lower():
                        matched = True
                        break
            if not matched:
                return False
        return True

    def _extract_shapes(self, group: FuseableGroup) -> dict:
        """Extract tensor shapes from group metadata."""
        return group.tensor_shapes if group.tensor_shapes else {}

    def select(self, fuseable_groups: List[FuseableGroup], train_source: str) -> List[KernelTarget]:
        """Select top-2 elementwise fusion targets from fuseable groups.

        Args:
            fuseable_groups: List of FuseableGroup from Phase 1 profiling.
            train_source: Current source code of train.py for validation.

        Returns:
            List of up to 2 KernelTarget objects, sorted by combined_gpu_time_us descending.
        """
        # Filter to elementwise groups only
        elementwise_groups = [
            g for g in fuseable_groups
            if g.fusion_type == "elementwise"
        ]

        # Sort by combined GPU time (highest first = most impactful)
        elementwise_groups.sort(key=lambda g: g.combined_gpu_time_us, reverse=True)

        targets = []
        for group in elementwise_groups:
            if len(targets) >= 2:
                break

            # Validate that ops still exist in train.py
            if not self._validate_target_in_source(group, train_source):
                continue

            target = KernelTarget(
                group_id=group.group_id,
                op_sequence=list(group.op_names),
                shapes=self._extract_shapes(group),
                reference_path="",
                estimated_speedup=group.estimated_speedup_ratio,
                fusion_type="elementwise",
            )
            targets.append(target)

        return targets
