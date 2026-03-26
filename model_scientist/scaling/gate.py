"""
Scale gate: decides whether a modification should proceed to full evaluation
based on multi-scale testing results and scaling curve predictions.
"""

from model_scientist.schemas import ScalingResult, ScalingPrediction
from model_scientist.scaling.curve_fitter import ScalingCurveFitter


class ScaleGate:
    """Decides whether a modification passes the scale gate."""

    def __init__(self):
        self._fitter = ScalingCurveFitter()

    def evaluate(
        self, prediction: ScalingPrediction, threshold: float = 0.5
    ) -> tuple[bool, str]:
        """Evaluate whether a modification should proceed based on scaling prediction.

        Args:
            prediction: Scaling curve prediction.
            threshold: predicted improvement at 1x must be >= threshold * small_scale_improvement.

        Returns:
            (passed, reason) tuple.
        """
        predicted_delta = prediction.predicted_delta_1x
        ci_low, ci_high = prediction.confidence_interval

        # No data
        if not prediction.scaling_results:
            return False, "no scaling results available"

        converged = [r for r in prediction.scaling_results if r.converged]
        if not converged:
            return False, "no converged results at any scale"

        # Check if prediction shows improvement (negative delta = better)
        if predicted_delta >= 0:
            return False, f"predicted delta at 1x is non-negative ({predicted_delta:.6f})"

        # Check confidence interval — reject if it includes zero
        if ci_low <= 0 <= ci_high or ci_high <= 0 <= ci_low:
            # CI spans zero — but check which end is closer
            if min(abs(ci_low), abs(ci_high)) < abs(predicted_delta) * 0.1:
                return False, (
                    f"confidence interval includes zero "
                    f"({ci_low:.6f}, {ci_high:.6f}), prediction uncertain"
                )

        # Check threshold: predicted improvement should be meaningful
        # compared to small-scale improvement
        small_scale_results = [
            r for r in converged if r.scale_factor < 1.0 and r.delta_vs_baseline < 0
        ]
        if small_scale_results:
            best_small = min(r.delta_vs_baseline for r in small_scale_results)
            # predicted_delta should be at least threshold * best_small_scale improvement
            if abs(predicted_delta) < threshold * abs(best_small):
                return False, (
                    f"predicted improvement at 1x ({predicted_delta:.6f}) "
                    f"is less than {threshold}x the best small-scale improvement ({best_small:.6f})"
                )

        return True, (
            f"predicted delta at 1x: {predicted_delta:.6f}, "
            f"CI: ({ci_low:.6f}, {ci_high:.6f}), "
            f"power law exponent: {prediction.power_law_exponent:.3f}"
        )

    def evaluate_from_results(
        self, results: list[ScalingResult], threshold: float = 0.5
    ) -> tuple[bool, str, ScalingPrediction]:
        """Fit scaling curve and evaluate in one step.

        Returns:
            (passed, reason, prediction) tuple.
        """
        prediction = self._fitter.fit(results)
        passed, reason = self.evaluate(prediction, threshold)
        return passed, reason, prediction
