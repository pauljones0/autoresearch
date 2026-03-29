"""
Bridge to KnowledgeCeilingMonitor for boosting paper/kernel arms.
"""

from bandit.schemas import BanditState, ArmState


class CeilingMonitorBanditBridge:
    """Applies knowledge ceiling signals to bandit arm state."""

    def apply_ceiling_signal(
        self,
        state: BanditState,
        ceiling_report: dict,
    ) -> BanditState:
        """Boost paper/kernel arms based on ceiling monitor trend signals.

        Uses paper_fraction_trend and kernel_fraction_trend from
        KnowledgeCeilingMonitor to adjust arm priors.

        Args:
            state: Current bandit state.
            ceiling_report: Dict with keys like 'paper_fraction_trend',
                           'kernel_fraction_trend' (positive = increasing contribution).

        Returns:
            Updated BanditState with boosted arms.
        """
        paper_trend = ceiling_report.get("paper_fraction_trend", 0.0)
        kernel_trend = ceiling_report.get("kernel_fraction_trend", 0.0)

        for arm_id, arm in state.arms.items():
            if not isinstance(arm, ArmState):
                continue

            if arm.source_type == "paper" and paper_trend > 0:
                # Positive trend: paper contributions are increasing,
                # boost diagnostics_boost (not alpha) to preserve evidence conservation
                boost = min(paper_trend * 2.0, 1.0)
                arm.diagnostics_boost += boost

            elif arm.source_type == "kernel" and kernel_trend > 0:
                # Positive trend: kernel contributions are increasing
                boost = min(kernel_trend * 2.0, 1.0)
                arm.diagnostics_boost += boost

        return state
