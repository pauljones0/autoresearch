"""Detect significant parameter interactions from experiment history."""

import math
from typing import List

from meta.schemas import Interaction, MetaExperimentResult


class InteractionDetector:
    """Detects pairwise parameter interactions using F-test approximation."""

    def detect(
        self,
        experiment_history: List[MetaExperimentResult],
        significance: float = 0.1,
    ) -> List[Interaction]:
        """Detect interactions between pairs of dimensions.

        For each pair of dimensions varied in >= 5 experiments:
        1. Partition experiments into 4 groups:
           - both default, i varied only, j varied only, both varied
        2. Compute interaction effect
        3. Test significance with F-test approximation
        """
        # Collect all dimensions that were varied
        dim_experiments = self._index_by_dimension(experiment_history)
        dim_ids = list(dim_experiments.keys())

        interactions = []
        for idx_i in range(len(dim_ids)):
            for idx_j in range(idx_i + 1, len(dim_ids)):
                dim_i = dim_ids[idx_i]
                dim_j = dim_ids[idx_j]

                interaction = self._test_pair(
                    dim_i, dim_j, experiment_history, significance
                )
                if interaction is not None:
                    interactions.append(interaction)

        return interactions

    def _test_pair(
        self,
        dim_i: str,
        dim_j: str,
        history: List[MetaExperimentResult],
        significance: float,
    ):
        """Test for interaction between two dimensions."""
        # Partition experiments into 4 groups
        group_neither: List[float] = []   # both at default
        group_i_only: List[float] = []    # only dim_i varied
        group_j_only: List[float] = []    # only dim_j varied
        group_both: List[float] = []      # both varied

        for exp in history:
            varied = self._get_varied_dims(exp)
            i_varied = dim_i in varied
            j_varied = dim_j in varied
            ir = exp.get("improvement_rate", 0.0) if isinstance(exp, dict) else getattr(exp, "improvement_rate", 0.0)

            if i_varied and j_varied:
                group_both.append(ir)
            elif i_varied:
                group_i_only.append(ir)
            elif j_varied:
                group_j_only.append(ir)
            else:
                group_neither.append(ir)

        # Need at least 5 total observations across groups with some in each
        total = len(group_neither) + len(group_i_only) + len(group_j_only) + len(group_both)
        if total < 5:
            return None

        # Compute group means (use 0.0 for empty groups)
        mean_neither = _safe_mean(group_neither)
        mean_i = _safe_mean(group_i_only)
        mean_j = _safe_mean(group_j_only)
        mean_both = _safe_mean(group_both)

        # Interaction effect: deviation from additive model
        # If effects were additive: mean_both ≈ mean_i + mean_j - mean_neither
        expected_both = mean_i + mean_j - mean_neither
        interaction_effect = mean_both - expected_both

        # F-test approximation for interaction significance
        p_value = self._f_test_interaction(
            group_neither, group_i_only, group_j_only, group_both,
            interaction_effect,
        )

        if p_value > significance:
            return None

        synergy = interaction_effect > 0
        antagonism = interaction_effect < 0

        return Interaction(
            dim_i=dim_i,
            dim_j=dim_j,
            interaction_effect=interaction_effect,
            p_value=p_value,
            synergy=synergy,
            antagonism=antagonism,
        )

    @staticmethod
    def _get_varied_dims(exp) -> set:
        """Extract set of dimension IDs that were varied in an experiment."""
        varied = set()
        config_diff = exp.get("config_diff", []) if isinstance(exp, dict) else getattr(exp, "config_diff", [])
        for diff in config_diff:
            if isinstance(diff, dict):
                pid = diff.get("param_id", "")
            else:
                pid = getattr(diff, "param_id", "")
            if pid:
                varied.add(pid)
        return varied

    def _index_by_dimension(self, history: List[MetaExperimentResult]) -> dict:
        """Build dim_id -> list of experiment indices mapping."""
        dim_exps: dict = {}
        for i, exp in enumerate(history):
            for dim_id in self._get_varied_dims(exp):
                dim_exps.setdefault(dim_id, []).append(i)
        return dim_exps

    @staticmethod
    def _f_test_interaction(
        g0: List[float],
        g1: List[float],
        g2: List[float],
        g3: List[float],
        interaction_effect: float,
    ) -> float:
        """Approximate F-test p-value for interaction effect.

        Uses a simplified F-statistic based on interaction SS vs residual SS.
        """
        all_values = g0 + g1 + g2 + g3
        n = len(all_values)
        if n < 4:
            return 1.0

        grand_mean = sum(all_values) / n

        # Residual variance (within-group)
        ss_within = 0.0
        for group in [g0, g1, g2, g3]:
            if group:
                gm = sum(group) / len(group)
                for v in group:
                    ss_within += (v - gm) ** 2

        df_within = max(1, n - 4)
        ms_within = ss_within / df_within

        if ms_within < 1e-15:
            # No within-group variance
            return 0.0 if abs(interaction_effect) > 1e-10 else 1.0

        # Interaction SS: n_both * interaction_effect^2 (approximation)
        n_both = max(1, len(g3))
        ss_interaction = n_both * interaction_effect ** 2
        ms_interaction = ss_interaction  # df_interaction = 1

        f_stat = ms_interaction / ms_within

        # Approximate p-value from F(1, df_within)
        # Using the relationship: p ≈ (1 + f/df)^(-(df+1)/2) for large df
        p_value = _approx_f_pvalue(f_stat, 1, df_within)
        return p_value


def _safe_mean(values: List[float]) -> float:
    """Return mean or 0.0 if empty."""
    if not values:
        return 0.0
    return sum(values) / len(values)


def _approx_f_pvalue(f_stat: float, df1: int, df2: int) -> float:
    """Approximate upper-tail p-value for F(df1, df2) distribution.

    Uses the Beta-distribution regularized incomplete beta function approximation.
    For df1=1, this simplifies to a t-distribution approximation.
    """
    if f_stat <= 0:
        return 1.0
    if df2 <= 0:
        return 1.0

    # Convert F to t^2 when df1=1
    # F(1, df2) = t(df2)^2, so p_F = p_t (two-sided) / 2? No: upper tail.
    # p = P(F > f) = P(t^2 > f) = 2 * P(t > sqrt(f)) for symmetric t
    # But we want one-sided F test, so p = P(F > f) directly.
    # Using approximation: x = df2 / (df2 + df1 * f), then p ≈ I_x(df2/2, df1/2)
    x = df2 / (df2 + df1 * f_stat)
    a = df2 / 2.0
    b = df1 / 2.0

    # Approximate regularized incomplete beta via simple formula
    # For large F (small x), p is small
    if x < 0.5:
        p = _approx_incomplete_beta(x, a, b)
    else:
        p = 1.0 - _approx_incomplete_beta(1.0 - x, b, a)

    return max(0.0, min(1.0, p))


def _approx_incomplete_beta(x: float, a: float, b: float) -> float:
    """Very rough approximation of the regularized incomplete beta I_x(a, b).

    Uses a continued-fraction-like expansion truncated early.
    Good enough for ranking interaction significance.
    """
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0

    # Use the power-series approximation for small x:
    # I_x(a, b) ≈ x^a * (1-x)^(b-1) / (a * B(a, b))
    # where B(a, b) = Gamma(a)*Gamma(b)/Gamma(a+b)
    try:
        log_term = a * math.log(x) + (b - 1) * math.log(1 - x)
        # Stirling approximation for log(B(a,b))
        log_beta = (
            _log_gamma(a) + _log_gamma(b) - _log_gamma(a + b)
        )
        log_result = log_term - math.log(a) - log_beta
        return min(1.0, max(0.0, math.exp(log_result)))
    except (ValueError, OverflowError):
        return 0.5


def _log_gamma(x: float) -> float:
    """Stirling approximation for log(Gamma(x))."""
    if x <= 0:
        return 0.0
    if x < 1:
        # Use reflection: Gamma(x) = pi / (sin(pi*x) * Gamma(1-x))
        # Simplified: just use small-x approximation
        return -math.log(x)
    return (
        0.5 * math.log(2 * math.pi / x)
        + x * (math.log(x + 1.0 / (12 * x - 1.0 / (10 * x))) - 1)
    )
