"""
Phase 5 — PaperSpecificRetraining: after sufficient paper-sourced evaluations,
train a small linear adapter that corrects systematic bias in surrogate
predictions for paper-sourced diffs.

Uses closed-form linear regression (no scipy/numpy).
"""


class PaperSpecificRetraining:
    """Linear adapter that adjusts surrogate predictions for paper-sourced diffs.

    Synthetic diffs generated from paper descriptions tend to have systematic
    bias relative to internally-generated diffs.  After accumulating enough
    paper evaluations, we fit  actual = weight * predicted + bias  and use
    that to correct future predictions.
    """

    def __init__(self):
        self._adapter_weights: dict = {}

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    @staticmethod
    def should_train(n_paper_evaluations: int, min_required: int = 50) -> bool:
        """Whether we have enough paper evaluations to train the adapter."""
        return n_paper_evaluations >= min_required

    def train_adapter(
        self,
        paper_predictions: list,
        paper_actuals: list,
    ) -> dict:
        """Fit a linear adapter: actual = weight * predicted + bias.

        Uses the closed-form OLS solution:
            weight = cov(x, y) / var(x)
            bias   = mean(y) - weight * mean(x)

        Returns:
            dict with keys 'weight' and 'bias'.
        """
        n = len(paper_predictions)
        if n < 2:
            self._adapter_weights = {"weight": 1.0, "bias": 0.0}
            return dict(self._adapter_weights)

        mean_x = sum(paper_predictions) / n
        mean_y = sum(paper_actuals) / n

        cov_xy = sum(
            (x - mean_x) * (y - mean_y)
            for x, y in zip(paper_predictions, paper_actuals)
        ) / n
        var_x = sum((x - mean_x) ** 2 for x in paper_predictions) / n

        if var_x < 1e-15:
            # All predictions identical — just use bias correction
            weight = 1.0
            bias = mean_y - mean_x
        else:
            weight = cov_xy / var_x
            bias = mean_y - weight * mean_x

        self._adapter_weights = {"weight": weight, "bias": bias}
        return dict(self._adapter_weights)

    @staticmethod
    def adjust_prediction(prediction: float, adapter_weights: dict) -> float:
        """Apply the linear adapter to a raw surrogate prediction.

        Args:
            prediction: Raw surrogate prediction.
            adapter_weights: dict with 'weight' and 'bias'.

        Returns:
            Adjusted prediction.
        """
        w = adapter_weights.get("weight", 1.0)
        b = adapter_weights.get("bias", 0.0)
        return w * prediction + b
