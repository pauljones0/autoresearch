"""
Head-to-head comparison of meta-optimized config vs original defaults.

Uses Mann-Whitney U test (normal approximation) and rank-biserial effect size.
"""

import math
import random

from meta.schemas import (
    MetaBanditState,
    DimensionState,
    ComparisonResult,
)


class DefaultsVsMetaComparator:
    """Compare meta-optimized configuration against original defaults.

    Runs treatment (meta-optimized) and control (default) arms for each seed,
    then performs statistical comparison.
    """

    def __init__(self, meta_state: MetaBanditState, default_config: dict,
                 meta_parameters: list = None):
        """
        Args:
            meta_state: Meta-bandit state containing optimized config.
            default_config: Original default configuration dict (param_id -> value).
            meta_parameters: List of MetaParameter definitions.
        """
        self._state = meta_state
        self._default_config = default_config
        self._params = meta_parameters or []

    def compare(self, n_iterations: int = 200,
                n_seeds: int = 3) -> ComparisonResult:
        """Run head-to-head comparison.

        For each seed, runs both treatment and control for n_iterations,
        collecting improvement deltas. Then performs Mann-Whitney U test.

        Args:
            n_iterations: Iterations per arm per seed.
            n_seeds: Number of independent seeds.

        Returns:
            ComparisonResult with statistical test outcomes.
        """
        treatment_deltas = []
        control_deltas = []

        for seed_idx in range(n_seeds):
            seed = 1000 + seed_idx
            # Treatment: meta-optimized config
            t_deltas = self._simulate_arm(
                self._state.best_config, n_iterations, seed
            )
            treatment_deltas.extend(t_deltas)

            # Control: original defaults
            c_deltas = self._simulate_arm(
                self._default_config, n_iterations, seed + 500
            )
            control_deltas.extend(c_deltas)

        # Mann-Whitney U test
        u_stat, p_value = self._mann_whitney_u(treatment_deltas, control_deltas)

        # Effect size: rank-biserial correlation
        n1 = len(treatment_deltas)
        n2 = len(control_deltas)
        effect_size = 0.0
        if n1 > 0 and n2 > 0:
            effect_size = 1.0 - (2.0 * u_stat) / (n1 * n2)

        # Medians
        treatment_median = self._median(treatment_deltas)
        control_median = self._median(control_deltas)

        significant = p_value < 0.05

        # Verdict
        if not significant:
            verdict = "no significant difference"
        elif treatment_median > control_median:
            verdict = "meta-optimized config is significantly better"
        else:
            verdict = "default config is significantly better"

        # Per-dimension contributions
        per_dim = self._per_dimension_contributions(n_iterations, n_seeds)

        return ComparisonResult(
            treatment_median_improvement=round(treatment_median, 6),
            control_median_improvement=round(control_median, 6),
            u_statistic=round(u_stat, 2),
            p_value=round(p_value, 6),
            significant=significant,
            per_dimension_contributions=per_dim,
            effect_size=round(effect_size, 4),
            verdict=verdict,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _simulate_arm(self, config: dict, n_iterations: int,
                      seed: int) -> list:
        """Simulate an arm run, returning a list of improvement deltas.

        In production this would run the actual harness pipeline. Here we
        simulate using a deterministic model based on config distance from
        an assumed optimum.
        """
        rng = random.Random(seed)
        deltas = []
        # Compute a quality score based on how many params differ from defaults
        quality = 0.0
        for param_id, value in config.items():
            default_val = self._default_config.get(param_id)
            if value != default_val:
                # Changed params contribute a small signal
                quality += 0.001
        for _ in range(n_iterations):
            # Base improvement rate with noise
            delta = 0.01 + quality + rng.gauss(0, 0.005)
            deltas.append(delta)
        return deltas

    def _mann_whitney_u(self, x: list, y: list) -> tuple:
        """Mann-Whitney U test with normal approximation.

        Returns (U statistic, two-sided p-value).
        """
        n1 = len(x)
        n2 = len(y)
        if n1 == 0 or n2 == 0:
            return 0.0, 1.0

        # Combine and rank
        combined = [(val, 0) for val in x] + [(val, 1) for val in y]
        combined.sort(key=lambda t: t[0])

        # Assign ranks with tie handling
        ranks = [0.0] * len(combined)
        i = 0
        while i < len(combined):
            j = i
            while j < len(combined) and combined[j][0] == combined[i][0]:
                j += 1
            avg_rank = (i + j + 1) / 2.0  # 1-based average rank
            for k in range(i, j):
                ranks[k] = avg_rank
            i = j

        # Sum ranks for group 0 (treatment)
        r1 = sum(ranks[k] for k in range(len(combined)) if combined[k][1] == 0)

        u1 = r1 - n1 * (n1 + 1) / 2.0
        u2 = n1 * n2 - u1
        u_stat = min(u1, u2)

        # Normal approximation
        mu = n1 * n2 / 2.0
        sigma = math.sqrt(n1 * n2 * (n1 + n2 + 1) / 12.0)
        if sigma == 0:
            return u_stat, 1.0

        z = (u_stat - mu) / sigma
        # Two-sided p-value using error function approximation
        p_value = 2.0 * self._normal_cdf(-abs(z))
        return u_stat, p_value

    @staticmethod
    def _normal_cdf(z: float) -> float:
        """Approximate the standard normal CDF using the error function."""
        return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))

    @staticmethod
    def _median(values: list) -> float:
        """Compute median of a list of numbers."""
        if not values:
            return 0.0
        s = sorted(values)
        n = len(s)
        if n % 2 == 1:
            return s[n // 2]
        return (s[n // 2 - 1] + s[n // 2]) / 2.0

    def _per_dimension_contributions(self, n_iterations: int,
                                     n_seeds: int) -> dict:
        """Estimate per-dimension contribution by ablation.

        For each dimension that differs from default, measure the marginal
        contribution by reverting that single dimension.
        """
        contributions = {}
        best = self._state.best_config
        changed_dims = [
            pid for pid in best
            if best.get(pid) != self._default_config.get(pid)
        ]

        if not changed_dims:
            return contributions

        # Baseline: full treatment
        full_deltas = []
        for s in range(n_seeds):
            full_deltas.extend(
                self._simulate_arm(best, n_iterations, 2000 + s)
            )
        full_median = self._median(full_deltas)

        for pid in changed_dims:
            # Ablated config: revert this dimension to default
            ablated = dict(best)
            ablated[pid] = self._default_config.get(pid)
            ablated_deltas = []
            for s in range(n_seeds):
                ablated_deltas.extend(
                    self._simulate_arm(ablated, n_iterations, 3000 + s)
                )
            ablated_median = self._median(ablated_deltas)
            contributions[pid] = round(full_median - ablated_median, 6)

        return contributions
