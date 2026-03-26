"""Track return on meta-optimization investment."""

from typing import List

from meta.schemas import MetaBanditState, MetaExperimentResult, ROIData


class MetaROITracker:
    """Computes ROI metrics for meta-optimization."""

    def compute_roi(
        self,
        meta_state: MetaBanditState,
        experiment_history: List[MetaExperimentResult],
        baseline_ir: float,
    ) -> ROIData:
        """Compute return on investment for meta-optimization.

        Metrics:
        - total_meta_iterations: sum of iterations across all experiments
        - total_production_iterations: budget_cycle_length - total_meta_iterations
        - improvement_from_meta: cumulative IR improvement over baseline
        - cost_of_meta: fraction of total iterations spent on meta
        - roi: improvement / cost (higher is better)
        """
        total_meta_iters = sum(e.n_iterations for e in experiment_history)
        cycle_length = meta_state.budget_cycle_length
        total_production = max(0, cycle_length - total_meta_iters)

        # Cumulative improvement: sum of (experiment_ir - baseline_ir) for each experiment
        bl = baseline_ir.mean_ir if hasattr(baseline_ir, "mean_ir") else float(baseline_ir or 0)
        improvement = 0.0
        attribution: dict = {}
        for exp in experiment_history:
            delta = exp.improvement_rate - bl
            improvement += max(0.0, delta)

            # Attribute improvement per dimension
            for diff in exp.config_diff:
                if isinstance(diff, dict):
                    pid = diff.get("param_id", "unknown")
                else:
                    pid = getattr(diff, "param_id", "unknown")
                attribution[pid] = attribution.get(pid, 0.0) + max(0.0, delta)

        # Cost: meta iterations as fraction of cycle
        cost = total_meta_iters / cycle_length if cycle_length > 0 else 0.0

        # ROI: improvement per unit cost
        roi = improvement / cost if cost > 0 else 0.0

        # Cumulative val_bpb improvement (proxy: sum of positive deltas)
        cumulative_bpb = sum(
            sum(d for d in exp.raw_deltas if d > 0)
            for exp in experiment_history
        )

        return ROIData(
            total_meta_iterations=total_meta_iters,
            total_production_iterations=total_production,
            improvement_from_meta=improvement,
            cost_of_meta=cost,
            roi=roi,
            cumulative_val_bpb_improvement=cumulative_bpb,
            attribution=attribution,
        )
