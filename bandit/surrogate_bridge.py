"""
Surrogate modulation factor calculation for annealing temperature.
"""


class SurrogateModulationCalculator:
    """Computes surrogate modulation factor based on prediction accuracy."""

    def compute_modulation(
        self,
        surrogate_predicted_delta: float = None,
        actual_delta: float = 0.0,
    ) -> float:
        """Compute surrogate modulation factor.

        Rules:
          - No prediction (None): 1.0 (neutral)
          - Predicted regression AND actual regression: 0.5 (cool down, surrogate was right)
          - Predicted improvement AND actual regression: 1.5 (heat up, surrogate was wrong)
          - Actual improvement (delta <= 0): 1.0 (neutral, good outcome regardless)
        """
        if surrogate_predicted_delta is None:
            return 1.0

        actual_is_regression = actual_delta > 0
        predicted_regression = surrogate_predicted_delta > 0
        predicted_improvement = surrogate_predicted_delta <= 0

        if not actual_is_regression:
            # Actual improvement — neutral regardless of prediction
            return 1.0

        # Actual is a regression
        if predicted_regression:
            # Surrogate correctly predicted regression — cool down
            return 0.5
        elif predicted_improvement:
            # Surrogate incorrectly predicted improvement — heat up
            return 1.5

        return 1.0
