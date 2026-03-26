"""
Fits power-law scaling curves to multi-scale results.
Model: delta(s) = a * s^b where s is scale factor, delta is improvement in val_bpb.
Uses only standard library math (no scipy/numpy).
"""

import math
from model_scientist.schemas import ScalingResult, ScalingPrediction


class ScalingCurveFitter:
    """Fits power-law scaling curves and makes predictions."""

    def fit(self, results: list[ScalingResult]) -> ScalingPrediction:
        """Fit a power-law to scaling results and predict at 1x scale."""
        # Filter to converged results with nonzero delta
        valid = [r for r in results if r.converged and r.delta_vs_baseline != 0.0]

        if not valid:
            return ScalingPrediction(
                predicted_delta_1x=0.0,
                confidence_interval=(0.0, 0.0),
                power_law_exponent=0.0,
                r_squared=0.0,
                scaling_results=results,
            )

        if len(valid) == 1:
            return self._single_point_prediction(valid[0], results)

        # Determine sign: are deltas mostly negative (improvement) or positive?
        neg_count = sum(1 for r in valid if r.delta_vs_baseline < 0)
        is_improvement = neg_count > len(valid) / 2

        # Work with absolute values for log-log fitting
        points = []
        for r in valid:
            s = r.scale_factor
            d = abs(r.delta_vs_baseline)
            if s > 0 and d > 0:
                points.append((math.log(s), math.log(d)))

        if len(points) < 2:
            return self._single_point_prediction(valid[0], results)

        # Least squares on log-log: log(|delta|) = log(a) + b * log(s)
        log_a, b, r_sq, residual_se = self._least_squares(points)
        a = math.exp(log_a)

        # Predict at scale=1.0: delta(1) = a * 1^b = a
        predicted_abs = a
        predicted_delta = -predicted_abs if is_improvement else predicted_abs

        # Confidence interval using residual standard error
        ci_low, ci_high = self._confidence_interval(
            log_a, b, residual_se, target_log_s=0.0, n=len(points), is_improvement=is_improvement
        )

        return ScalingPrediction(
            predicted_delta_1x=predicted_delta,
            confidence_interval=(ci_low, ci_high),
            power_law_exponent=b,
            r_squared=r_sq,
            scaling_results=results,
        )

    def predict_at_scale(
        self, prediction: ScalingPrediction, target_scale: float
    ) -> tuple[float, float, float]:
        """Predict delta at a given scale using the fitted power law.

        Returns (predicted_delta, confidence_low, confidence_high).
        """
        if prediction.r_squared == 0.0 and prediction.predicted_delta_1x == 0.0:
            return 0.0, 0.0, 0.0

        # Recover sign from predicted_delta_1x
        is_improvement = prediction.predicted_delta_1x < 0

        # At scale=1.0, delta = predicted_delta_1x = sign * a
        a = abs(prediction.predicted_delta_1x)
        b = prediction.power_law_exponent

        if target_scale <= 0:
            return 0.0, 0.0, 0.0

        predicted_abs = a * (target_scale ** b)
        predicted = -predicted_abs if is_improvement else predicted_abs

        # Scale the confidence interval proportionally
        ci_low, ci_high = prediction.confidence_interval
        if prediction.predicted_delta_1x != 0:
            ratio = predicted / prediction.predicted_delta_1x
            ci_low_scaled = ci_low * ratio
            ci_high_scaled = ci_high * ratio
        else:
            ci_low_scaled, ci_high_scaled = 0.0, 0.0

        return predicted, ci_low_scaled, ci_high_scaled

    def _single_point_prediction(
        self, result: ScalingResult, all_results: list[ScalingResult]
    ) -> ScalingPrediction:
        """With only one data point, use it directly with wide confidence."""
        delta = result.delta_vs_baseline
        # Wide confidence interval: from 2x the delta to 0
        if delta < 0:
            ci = (delta * 2, 0.0)
        elif delta > 0:
            ci = (0.0, delta * 2)
        else:
            ci = (0.0, 0.0)

        return ScalingPrediction(
            predicted_delta_1x=delta,
            confidence_interval=ci,
            power_law_exponent=0.0,
            r_squared=0.0,
            scaling_results=all_results,
        )

    def _least_squares(
        self, points: list[tuple[float, float]]
    ) -> tuple[float, float, float, float]:
        """Fit y = intercept + slope * x via ordinary least squares.

        Returns (intercept, slope, r_squared, residual_standard_error).
        """
        n = len(points)
        sum_x = sum(p[0] for p in points)
        sum_y = sum(p[1] for p in points)
        sum_xx = sum(p[0] ** 2 for p in points)
        sum_xy = sum(p[0] * p[1] for p in points)

        denom = n * sum_xx - sum_x ** 2
        if abs(denom) < 1e-15:
            # All x values the same — can't fit a line
            mean_y = sum_y / n
            return mean_y, 0.0, 0.0, 0.0

        slope = (n * sum_xy - sum_x * sum_y) / denom
        intercept = (sum_y - slope * sum_x) / n

        # R-squared
        mean_y = sum_y / n
        ss_tot = sum((p[1] - mean_y) ** 2 for p in points)
        ss_res = sum((p[1] - (intercept + slope * p[0])) ** 2 for p in points)
        r_sq = 1.0 - ss_res / ss_tot if ss_tot > 1e-15 else 0.0

        # Residual standard error
        if n > 2:
            residual_se = math.sqrt(ss_res / (n - 2))
        else:
            residual_se = 0.0

        return intercept, slope, r_sq, residual_se

    def _confidence_interval(
        self,
        log_a: float,
        b: float,
        residual_se: float,
        target_log_s: float,
        n: int,
        is_improvement: bool,
    ) -> tuple[float, float]:
        """Compute approximate confidence interval for prediction at target_log_s."""
        if residual_se == 0.0 or n < 3:
            # Not enough data for meaningful CI — use wide range
            predicted = math.exp(log_a + b * target_log_s)
            margin = predicted * 0.5
        else:
            # Use ~2 standard errors for ~95% CI
            predicted_log = log_a + b * target_log_s
            margin_log = 2.0 * residual_se
            predicted = math.exp(predicted_log)
            upper = math.exp(predicted_log + margin_log)
            lower = math.exp(predicted_log - margin_log)
            margin = (upper - lower) / 2

        if is_improvement:
            return -(predicted + margin), -(predicted - margin)
        else:
            return predicted - margin, predicted + margin
