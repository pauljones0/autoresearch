"""Meta-budget optimizer: adjusts exploration budget based on recent performance."""

from typing import List

from meta.schemas import (
    BudgetRecommendation,
    MetaBanditState,
    ROIData,
    MetaExperimentResult,
)


class MetaBudgetOptimizer:
    """Recommends meta-budget fraction based on recent experiment outcomes."""

    def recommend_budget(
        self,
        meta_state: MetaBanditState,
        roi_data: ROIData,
    ) -> BudgetRecommendation:
        """Recommend a new budget fraction.

        Rules:
        - If last 5 experiments all improved: increase to min(0.35, current * 1.25)
        - If last 5 experiments all failed: decrease to max(0.05, current * 0.6)
        - Otherwise: keep current fraction
        """
        current = meta_state.budget_fraction

        # Determine recent trend from ROI data
        recent_improved = self._check_recent_trend(roi_data)

        if recent_improved is True:
            recommended = min(0.35, current * 1.25)
            reason = "Last 5 meta-experiments all improved; increasing budget."
            confidence = "high"
        elif recent_improved is False:
            recommended = max(0.05, current * 0.6)
            reason = "Last 5 meta-experiments all failed; decreasing budget."
            confidence = "high"
        else:
            recommended = current
            reason = "Mixed results in recent experiments; maintaining budget."
            confidence = "medium"

        return BudgetRecommendation(
            current_fraction=current,
            recommended_fraction=recommended,
            reason=reason,
            confidence=confidence,
        )

    def recommend_from_history(
        self,
        meta_state: MetaBanditState,
        experiment_history: List[MetaExperimentResult],
        baseline_ir: float = 0.0,
    ) -> BudgetRecommendation:
        """Recommend budget based on raw experiment history.

        Convenience method that evaluates the last 5 experiments
        against the baseline IR.
        """
        current = meta_state.budget_fraction
        last_5 = experiment_history[-5:] if len(experiment_history) >= 5 else experiment_history

        if len(last_5) < 5:
            return BudgetRecommendation(
                current_fraction=current,
                recommended_fraction=current,
                reason=f"Only {len(last_5)} experiments so far; need 5 for budget adjustment.",
                confidence="low",
            )

        all_improved = all(e.improvement_rate > baseline_ir for e in last_5)
        all_failed = all(e.improvement_rate <= baseline_ir for e in last_5)

        if all_improved:
            recommended = min(0.35, current * 1.25)
            reason = "Last 5 experiments all beat baseline; increasing budget."
            confidence = "high"
        elif all_failed:
            recommended = max(0.05, current * 0.6)
            reason = "Last 5 experiments all at or below baseline; decreasing budget."
            confidence = "high"
        else:
            recommended = current
            reason = "Mixed results; maintaining current budget."
            confidence = "medium"

        return BudgetRecommendation(
            current_fraction=current,
            recommended_fraction=recommended,
            reason=reason,
            confidence=confidence,
        )

    @staticmethod
    def _check_recent_trend(roi_data: ROIData):
        """Determine trend from ROI data.

        Returns True if improving, False if failing, None if mixed.
        Uses roi > 1.0 as the improvement threshold.
        """
        if roi_data.roi > 1.0 and roi_data.improvement_from_meta > 0:
            return True
        if roi_data.roi < 0.1 and roi_data.total_meta_iterations > 0:
            return False
        return None
