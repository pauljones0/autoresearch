"""
FuseableOperationDetector: analyzes profiler traces to identify groups
of operations that can be fused into single Triton kernels.
"""

import re
import uuid
from typing import Optional

from ..schemas import OperationProfile, FuseableGroup


# Fusion pattern definitions: (pattern_name, fusion_type, sequence of op regexes)
_FUSION_PATTERNS = [
    # RMSNorm: norm reduction + scale
    (
        "rmsnorm",
        "normalization",
        [re.compile(r"rms_norm|layer_norm", re.I)],
    ),
    # MLP activation: relu + square (ReGLU-style)
    (
        "mlp_activation",
        "elementwise",
        [re.compile(r"relu", re.I), re.compile(r"square|pow", re.I)],
    ),
    # Softcap: div + tanh + mul
    (
        "softcap",
        "elementwise",
        [re.compile(r"div", re.I), re.compile(r"tanh", re.I), re.compile(r"mul", re.I)],
    ),
    # Value embedding gate: sigmoid + mul + add
    (
        "ve_gate",
        "elementwise",
        [re.compile(r"sigmoid", re.I), re.compile(r"mul", re.I), re.compile(r"add", re.I)],
    ),
    # Rotary embedding: slice + mul + cat
    (
        "rotary_emb",
        "elementwise",
        [re.compile(r"slice|narrow|chunk", re.I), re.compile(r"mul", re.I), re.compile(r"cat", re.I)],
    ),
    # Residual connection: mul (lambda) + mul (lambda) + add
    (
        "residual_scale",
        "elementwise",
        [re.compile(r"mul", re.I), re.compile(r"mul", re.I), re.compile(r"add", re.I)],
    ),
    # Cross-entropy components
    (
        "cross_entropy",
        "reduction",
        [re.compile(r"cross_entropy|nll_loss|log_softmax", re.I)],
    ),
]


class FuseableOperationDetector:
    """Detects groups of operations that can be fused into Triton kernels.

    Uses heuristics based on:
    - Consecutive elementwise ops on same-shaped tensors
    - Known fuseable patterns (linear->activation, norm components, etc.)
    - Optimizer step components
    """

    def detect(self, profiles: list[OperationProfile]) -> list[FuseableGroup]:
        """Analyze profiled ops and return fuseable groups.

        Args:
            profiles: List of OperationProfile from the profiler.

        Returns:
            List of FuseableGroup describing fuseable operation clusters.
        """
        groups = []

        # Pass 1: Match known fusion patterns
        used_indices = set()
        groups.extend(self._match_known_patterns(profiles, used_indices))

        # Pass 2: Find consecutive elementwise ops on same shapes
        groups.extend(self._find_consecutive_elementwise(profiles, used_indices))

        return groups

    def _match_known_patterns(
        self,
        profiles: list[OperationProfile],
        used_indices: set[int],
    ) -> list[FuseableGroup]:
        """Match known fusion patterns against the op sequence."""
        groups = []

        for pattern_name, fusion_type, op_regexes in _FUSION_PATTERNS:
            # Sliding window search
            window_size = len(op_regexes)
            for i in range(len(profiles) - window_size + 1):
                if any(j in used_indices for j in range(i, i + window_size)):
                    continue

                match = True
                for j, regex in enumerate(op_regexes):
                    if not regex.search(profiles[i + j].op_name):
                        match = False
                        break

                if match:
                    matched_profiles = profiles[i : i + window_size]
                    combined_time = sum(p.gpu_time_us for p in matched_profiles)

                    # Estimate fused time: eliminate kernel launch overhead
                    # and memory round-trips between ops
                    estimated_fused = combined_time * _estimate_fusion_factor(
                        fusion_type, window_size
                    )

                    group = FuseableGroup(
                        group_id=f"{pattern_name}_{uuid.uuid4().hex[:8]}",
                        op_names=[p.op_name for p in matched_profiles],
                        combined_gpu_time_us=combined_time,
                        estimated_fused_time_us=estimated_fused,
                        estimated_speedup_ratio=(
                            combined_time / estimated_fused if estimated_fused > 0 else 1.0
                        ),
                        fusion_type=fusion_type,
                        tensor_shapes=_collect_shapes(matched_profiles),
                    )
                    groups.append(group)
                    used_indices.update(range(i, i + window_size))

        return groups

    def _find_consecutive_elementwise(
        self,
        profiles: list[OperationProfile],
        used_indices: set[int],
    ) -> list[FuseableGroup]:
        """Find runs of consecutive elementwise ops on same-shaped tensors."""
        groups = []
        elementwise_re = re.compile(
            r"add|mul|sub|div|relu|gelu|silu|tanh|sigmoid|exp|log|sqrt|"
            r"square|pow|neg|abs|clamp|where|copy|fill|zero",
            re.I,
        )

        run_start = None
        run_profiles = []

        for i, op in enumerate(profiles):
            if i in used_indices:
                if len(run_profiles) >= 2:
                    groups.append(self._make_elementwise_group(run_profiles))
                run_start = None
                run_profiles = []
                continue

            is_elementwise = elementwise_re.search(op.op_name)
            if not is_elementwise:
                if len(run_profiles) >= 2:
                    groups.append(self._make_elementwise_group(run_profiles))
                run_start = None
                run_profiles = []
                continue

            if run_profiles:
                # Check shape compatibility
                prev_shapes = run_profiles[-1].input_shapes
                curr_shapes = op.input_shapes
                if prev_shapes and curr_shapes and prev_shapes != curr_shapes:
                    if len(run_profiles) >= 2:
                        groups.append(self._make_elementwise_group(run_profiles))
                    run_start = i
                    run_profiles = [op]
                    continue

            if run_start is None:
                run_start = i
            run_profiles.append(op)

        if len(run_profiles) >= 2:
            groups.append(self._make_elementwise_group(run_profiles))

        return groups

    def _make_elementwise_group(
        self, matched_profiles: list[OperationProfile]
    ) -> FuseableGroup:
        """Create a FuseableGroup from a run of elementwise ops."""
        combined_time = sum(p.gpu_time_us for p in matched_profiles)
        n = len(matched_profiles)
        estimated_fused = combined_time * _estimate_fusion_factor("elementwise", n)

        return FuseableGroup(
            group_id=f"elementwise_{uuid.uuid4().hex[:8]}",
            op_names=[p.op_name for p in matched_profiles],
            combined_gpu_time_us=combined_time,
            estimated_fused_time_us=estimated_fused,
            estimated_speedup_ratio=(
                combined_time / estimated_fused if estimated_fused > 0 else 1.0
            ),
            fusion_type="elementwise",
            tensor_shapes=_collect_shapes(matched_profiles),
        )


def _estimate_fusion_factor(fusion_type: str, n_ops: int) -> float:
    """Estimate the time ratio (fused / unfused) for a fusion type.

    Accounts for eliminated kernel launches (~5us each) and reduced
    memory round-trips.
    """
    # Base: each fusion eliminates n-1 kernel launches and memory passes
    if fusion_type == "elementwise":
        # Elementwise fusion is very effective — near 1/n for memory-bound ops
        return max(0.15, 1.0 / n_ops + 0.1)
    elif fusion_type == "normalization":
        return 0.5  # Norm fusion typically ~2x
    elif fusion_type == "reduction":
        return 0.7  # Reductions harder to fuse
    elif fusion_type == "attention":
        return 0.4  # Attention fusion very effective
    elif fusion_type == "optimizer":
        return 0.3  # Optimizer fusion very effective (many elementwise)
    return 0.6


def _collect_shapes(profiles: list[OperationProfile]) -> dict:
    """Collect unique tensor shapes from a group of profiles."""
    shapes = {}
    for i, p in enumerate(profiles):
        if p.input_shapes:
            shapes[f"op_{i}_input"] = p.input_shapes
        if p.output_shapes:
            shapes[f"op_{i}_output"] = p.output_shapes
    return shapes
