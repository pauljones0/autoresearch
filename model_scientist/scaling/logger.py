"""
Logs scale-gate decisions to the hypothesis journal.
"""

from dataclasses import asdict
from model_scientist.schemas import ScalingResult, ScalingPrediction


class ScaleGateLogger:
    """Logs all scale-gate decisions to the hypothesis journal."""

    def log_decision(
        self,
        journal_writer,
        modification_id: str,
        results: list[ScalingResult],
        prediction: ScalingPrediction,
        passed: bool,
        reason: str,
    ):
        """Log a scale-gate decision to the journal.

        Args:
            journal_writer: Object with an update_entry(id, **kwargs) method
                or a log(entry_dict) method for writing to the journal.
            modification_id: ID of the modification being evaluated.
            results: List of ScalingResult from multi-scale testing.
            prediction: ScalingPrediction from curve fitting.
            passed: Whether the modification passed the scale gate.
            reason: Human-readable reason for the decision.
        """
        scaling_data = self._format_scaling_data(results, prediction, passed, reason)

        # Try update_entry first (for existing journal entries)
        if hasattr(journal_writer, "update_entry"):
            try:
                journal_writer.update_entry(
                    modification_id,
                    scaling_data=scaling_data,
                    scale_gate_passed=passed,
                )
                return
            except Exception:
                pass

        # Fallback: try log method
        if hasattr(journal_writer, "log"):
            journal_writer.log({
                "id": modification_id,
                "scaling_data": scaling_data,
                "scale_gate_passed": passed,
            })

    def _format_scaling_data(
        self,
        results: list[ScalingResult],
        prediction: ScalingPrediction,
        passed: bool,
        reason: str,
    ) -> dict:
        """Format scaling data as a dict suitable for JournalEntry.scaling_data."""
        results_data = []
        for r in results:
            entry = {
                "scale_factor": r.scale_factor,
                "val_bpb": r.val_bpb,
                "delta_vs_baseline": r.delta_vs_baseline,
                "converged": r.converged,
                "training_seconds": r.training_seconds,
                "config": r.config if isinstance(r.config, dict) else asdict(r.config) if hasattr(r.config, '__dataclass_fields__') else {},
            }
            results_data.append(entry)

        return {
            "results": results_data,
            "prediction": {
                "predicted_delta_1x": prediction.predicted_delta_1x,
                "confidence_interval": list(prediction.confidence_interval),
                "power_law_exponent": prediction.power_law_exponent,
                "r_squared": prediction.r_squared,
            },
            "gate_passed": passed,
            "gate_reason": reason,
        }
