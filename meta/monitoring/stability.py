"""
Long-term stability monitoring for the meta-optimization loop.

Tracks IR stability, false positive rates, and maintenance experiment quality.
"""

import math

from meta.schemas import (
    MetaBanditState,
    StabilityReport,
    DivergenceAlert,
)


class LongTermStabilityMonitor:
    """Monitor long-term stability of the meta-optimized configuration.

    Checks that improvement rate stays within acceptable bounds,
    divergence alerts are not excessive (false positive rate), and
    maintenance experiments continue producing useful discoveries.
    """

    IR_TOLERANCE = 0.10  # +-10% tolerance band around baseline IR

    def __init__(self, baseline_ir: float = 0.0):
        """
        Args:
            baseline_ir: The reference improvement rate to compare against.
        """
        self._baseline_ir = baseline_ir

    def check(self, meta_state: MetaBanditState,
              recent_ir_windows: list,
              divergence_history: list) -> StabilityReport:
        """Run stability checks and return a report.

        Args:
            meta_state: Current meta-bandit state.
            recent_ir_windows: List of recent IR measurements (floats).
            divergence_history: List of DivergenceAlert dicts/objects,
                each with 'triggered' bool.

        Returns:
            StabilityReport with stability assessment.
        """
        # --- IR stability ---
        ir_stable, mean_ir, std_ir = self._check_ir_stability(
            recent_ir_windows
        )

        # --- False positive rate ---
        total_alerts, false_positives = self._analyze_divergence_history(
            divergence_history, recent_ir_windows
        )

        # --- Maintenance experiment quality ---
        maint_experiments, maint_discoveries = self._check_maintenance_quality(
            meta_state
        )

        return StabilityReport(
            ir_stable=ir_stable,
            mean_ir=round(mean_ir, 6),
            std_ir=round(std_ir, 6),
            divergence_triggers=total_alerts,
            false_positives=false_positives,
            maintenance_experiments=maint_experiments,
            maintenance_discoveries=maint_discoveries,
        )

    # ------------------------------------------------------------------
    # Internal checks
    # ------------------------------------------------------------------

    def _check_ir_stability(self, ir_windows: list) -> tuple:
        """Check if IR is stable within +-10% of baseline.

        Returns:
            (is_stable, mean_ir, std_ir)
        """
        if not ir_windows:
            return True, 0.0, 0.0

        mean_ir = sum(ir_windows) / len(ir_windows)
        variance = sum((x - mean_ir) ** 2 for x in ir_windows) / len(ir_windows)
        std_ir = math.sqrt(variance)

        if self._baseline_ir <= 0:
            # No baseline to compare against; consider stable if low variance
            return std_ir < 0.01, mean_ir, std_ir

        lower = self._baseline_ir * (1.0 - self.IR_TOLERANCE)
        upper = self._baseline_ir * (1.0 + self.IR_TOLERANCE)

        # Check if all windows fall within the tolerance band
        all_within = all(lower <= ir <= upper for ir in ir_windows)
        return all_within, mean_ir, std_ir

    def _analyze_divergence_history(self, divergence_history: list,
                                     ir_windows: list) -> tuple:
        """Count divergence triggers and estimate false positives.

        A triggered alert is a false positive if the IR recovered (i.e.,
        subsequent windows returned to acceptable range).

        Returns:
            (total_triggers, false_positive_count)
        """
        total_triggers = 0
        false_positives = 0

        for entry in divergence_history:
            triggered = False
            if isinstance(entry, dict):
                triggered = entry.get("triggered", False)
            elif isinstance(entry, DivergenceAlert):
                triggered = entry.triggered
            elif hasattr(entry, "triggered"):
                triggered = entry.triggered

            if triggered:
                total_triggers += 1

        # Estimate false positives: if IR is currently stable and we had
        # many triggers, some were likely false positives.
        if ir_windows and total_triggers > 0:
            if self._baseline_ir > 0:
                mean_ir = sum(ir_windows) / len(ir_windows)
                lower = self._baseline_ir * (1.0 - self.IR_TOLERANCE)
                upper = self._baseline_ir * (1.0 + self.IR_TOLERANCE)
                if lower <= mean_ir <= upper:
                    # IR recovered; past triggers are likely false positives
                    # Heuristic: half of triggers when IR is currently stable
                    false_positives = max(0, total_triggers - 1)

        return total_triggers, false_positives

    def _check_maintenance_quality(self, meta_state: MetaBanditState) -> tuple:
        """Check maintenance experiment quality.

        In maintenance mode, experiments should still occasionally find
        improvements (discoveries).

        Returns:
            (maintenance_experiments, maintenance_discoveries)
        """
        if meta_state.meta_regime != "maintenance":
            return 0, 0

        # Count total experiments as maintenance experiments
        maint_experiments = meta_state.total_meta_experiments

        # Count promotions across dimensions as discoveries
        discoveries = 0
        for dim_id, dim in meta_state.dimensions.items():
            if hasattr(dim, "last_promoted") and dim.last_promoted > 0:
                discoveries += 1

        return maint_experiments, discoveries
