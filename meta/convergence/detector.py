"""Convergence detection for the meta-optimization loop."""

from meta.schemas import MetaBanditState, ConvergenceStatus


class MetaConvergenceDetector:
    """Detects when the meta-loop has converged and no further tuning is needed."""

    def __init__(self, convergence_window: int = 5):
        self.convergence_window = convergence_window

    def check(self, meta_state: MetaBanditState, promotion_history: list) -> ConvergenceStatus:
        """Check whether the meta-loop has converged.

        Convergence requires all three conditions:
        1. No promotion in last `convergence_window` meta-experiments.
        2. All posterior variances < 0.01.
        3. F-test p-value > 0.1 (variants are not significantly different).
        """
        if meta_state.meta_regime == "maintenance":
            return ConvergenceStatus(
                converged=True,
                meta_experiments_since_last_promotion=self._experiments_since_last_promotion(
                    promotion_history, meta_state.total_meta_experiments
                ),
                max_posterior_variance=self._max_posterior_variance(meta_state),
                f_test_p_value=1.0,
                recommendation="already_in_maintenance",
            )

        exps_since = self._experiments_since_last_promotion(
            promotion_history, meta_state.total_meta_experiments
        )
        no_recent_promotion = exps_since >= self.convergence_window

        max_var = self._max_posterior_variance(meta_state)
        low_variance = max_var < 0.01

        f_p = self._f_test_p_value(meta_state)
        f_test_pass = f_p > 0.1

        converged = no_recent_promotion and low_variance and f_test_pass

        if converged:
            recommendation = "enter_maintenance"
        else:
            recommendation = "continue"

        return ConvergenceStatus(
            converged=converged,
            meta_experiments_since_last_promotion=exps_since,
            max_posterior_variance=max_var,
            f_test_p_value=f_p,
            recommendation=recommendation,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _experiments_since_last_promotion(
        self, promotion_history: list, total_experiments: int
    ) -> int:
        if not promotion_history:
            return total_experiments
        last_promo_iter = max(
            entry.get("meta_iteration", 0) for entry in promotion_history
        )
        return total_experiments - last_promo_iter

    def _max_posterior_variance(self, meta_state: MetaBanditState) -> float:
        max_var = 0.0
        for dim in meta_state.dimensions.values():
            if not hasattr(dim, "variant_posteriors"):
                continue
            for post in dim.variant_posteriors.values():
                if isinstance(post, dict):
                    a = post.get("alpha", 1.0)
                    b = post.get("beta", 1.0)
                    var = (a * b) / ((a + b) ** 2 * (a + b + 1))
                    if var > max_var:
                        max_var = var
        return max_var

    def _f_test_p_value(self, meta_state: MetaBanditState) -> float:
        """Approximate F-test across variant posteriors.

        Uses the ratio of between-dimension variance to within-dimension
        variance of posterior means.  When all posteriors have converged
        to similar values the p-value will be high.
        """
        all_means = []
        group_means = []
        for dim in meta_state.dimensions.values():
            if not hasattr(dim, "variant_posteriors"):
                continue
            dim_means = []
            for post in dim.variant_posteriors.values():
                if isinstance(post, dict):
                    a = post.get("alpha", 1.0)
                    b = post.get("beta", 1.0)
                    mean = a / (a + b)
                    dim_means.append(mean)
                    all_means.append(mean)
            if dim_means:
                group_means.append(sum(dim_means) / len(dim_means))

        if len(all_means) < 2 or len(group_means) < 2:
            return 1.0  # not enough data to reject

        grand_mean = sum(all_means) / len(all_means)
        between_var = sum((m - grand_mean) ** 2 for m in group_means) / (
            len(group_means) - 1
        )
        within_var = sum((m - grand_mean) ** 2 for m in all_means) / (
            len(all_means) - 1
        )

        if within_var < 1e-12:
            return 1.0  # essentially zero variance → converged

        f_stat = between_var / within_var
        # Approximate p-value: when f_stat is small posteriors are similar.
        # Use a simple sigmoid approximation (stdlib-only).
        import math

        p_approx = 1.0 / (1.0 + math.exp(5.0 * (f_stat - 1.0)))
        return max(0.0, min(1.0, p_approx))
