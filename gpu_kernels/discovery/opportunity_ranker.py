"""
KernelOpportunityRanker: ranks un-optimized operations by estimated impact
to prioritize kernel generation targets.
"""

from ..schemas import (
    FuseableGroup,
    KernelConfigEntry,
    KernelOpportunity,
    OperationProfile,
    load_json,
)


class KernelOpportunityRanker:
    """Rank fuseable groups by optimization potential.

    Estimates impact as combined_gpu_time_us * (1 - bandwidth_utilization),
    filters already-optimized groups, and applies constraint penalties.
    """

    def rank(
        self,
        fuseable_groups: list[FuseableGroup],
        kernel_profiling: list[OperationProfile],
        kernel_config: dict,
        constraints: list[dict] | None = None,
    ) -> list[KernelOpportunity]:
        """Rank un-optimized ops by estimated impact.

        Args:
            fuseable_groups: Detected fuseable operation groups.
            kernel_profiling: Per-operation profiling data.
            kernel_config: Current kernel_config.json as dict (group_id -> entry).
            constraints: Optional negative constraints to penalize certain targets.

        Returns:
            List of KernelOpportunity sorted by adjusted_score descending.
        """
        if constraints is None:
            constraints = []

        # Build lookup: op_name -> bandwidth_utilization from profiling
        bw_by_op: dict[str, float] = {}
        for prof in kernel_profiling:
            if isinstance(prof, dict):
                bw_by_op[prof.get("op_name", "")] = prof.get("bandwidth_utilization", 0.0)
            else:
                bw_by_op[prof.op_name] = prof.bandwidth_utilization

        # Normalize kernel_config to a set of active group_ids
        active_group_ids = set()
        for group_id, entry in kernel_config.items():
            if isinstance(entry, dict):
                if entry.get("backend") == "triton" and entry.get("enabled", True):
                    active_group_ids.add(group_id)
            elif isinstance(entry, KernelConfigEntry):
                if entry.backend == "triton" and entry.enabled:
                    active_group_ids.add(group_id)

        opportunities: list[KernelOpportunity] = []

        for group in fuseable_groups:
            g = group if not isinstance(group, dict) else _group_from_dict(group)

            # Skip already-optimized groups
            if g.group_id in active_group_ids:
                continue

            # Compute average bandwidth utilization across ops in the group
            bw_values = [bw_by_op.get(op, 0.0) for op in g.op_names]
            avg_bw = sum(bw_values) / len(bw_values) if bw_values else 0.0

            # Estimated impact: time that could be saved
            estimated_impact = g.combined_gpu_time_us * (1.0 - avg_bw)

            # Count how many times this group was previously attempted
            already_attempted = _count_attempts(g.group_id, kernel_config)

            # Apply constraint penalties
            penalty = _compute_constraint_penalty(g, constraints)
            adjusted_score = estimated_impact * (1.0 - penalty)

            opp = KernelOpportunity(
                group_id=g.group_id,
                op_names=list(g.op_names),
                estimated_impact_us=estimated_impact,
                current_bandwidth_utilization=avg_bw,
                fusion_type=g.fusion_type,
                constraint_penalty=penalty,
                adjusted_score=adjusted_score,
                already_attempted_count=already_attempted,
            )
            opportunities.append(opp)

        # Sort by adjusted_score descending
        opportunities.sort(key=lambda o: o.adjusted_score, reverse=True)
        return opportunities


def _group_from_dict(d: dict) -> FuseableGroup:
    """Convert a dict to FuseableGroup."""
    g = FuseableGroup()
    for k, v in d.items():
        if hasattr(g, k):
            setattr(g, k, v)
    return g


def _count_attempts(group_id: str, kernel_config: dict) -> int:
    """Count previous generation attempts for a group from config metadata."""
    entry = kernel_config.get(group_id)
    if not entry:
        return 0
    if isinstance(entry, dict):
        return entry.get("attempt_count", 0)
    return getattr(entry, "attempt_count", 0)


def _compute_constraint_penalty(
    group: FuseableGroup, constraints: list[dict]
) -> float:
    """Compute a penalty in [0, 1] based on matching negative constraints."""
    if not constraints:
        return 0.0

    total_penalty = 0.0
    for constraint in constraints:
        # Match constraints by fusion_type or op_name overlap
        c_fusion = constraint.get("fusion_type", "")
        c_ops = set(constraint.get("op_names", []))
        c_penalty = constraint.get("penalty", 0.2)

        if c_fusion and c_fusion == group.fusion_type:
            total_penalty += c_penalty
        elif c_ops and c_ops & set(group.op_names):
            total_penalty += c_penalty * 0.5

    return min(total_penalty, 1.0)
