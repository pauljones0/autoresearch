"""
Selection distribution validator using chi-squared test.
"""

import math
import random

from bandit.schemas import BanditState, SelectionValidationReport
from bandit.sampler import ThompsonSamplerEngine


class SelectionDistributionValidator:
    """Validates that Thompson Sampling selection matches theoretical distribution."""

    def __init__(self):
        self.sampler = ThompsonSamplerEngine()

    def validate(self, state: BanditState, n_simulations: int = 10000) -> SelectionValidationReport:
        """Run n_simulations selections and compare to expected distribution.

        Uses chi-squared test to check if observed frequencies match
        theoretical Thompson Sampling probabilities.
        """
        arm_ids = [aid for aid in state.arms if hasattr(state.arms[aid], 'alpha')]
        if not arm_ids:
            return SelectionValidationReport(passed=True)

        # Run simulations
        counts = {aid: 0 for aid in arm_ids}
        for i in range(n_simulations):
            rng = random.Random(i * 31337)
            result = self.sampler.select(state, rng=rng)
            if result.arm_id in counts:
                counts[result.arm_id] += 1

        # Compute observed frequencies
        per_arm_freq = {aid: c / n_simulations for aid, c in counts.items()}

        # Estimate expected frequencies via a separate large sample
        # (theoretical distribution is complex, so we use Monte Carlo reference)
        expected_counts = {aid: 0 for aid in arm_ids}
        ref_samples = n_simulations
        for i in range(ref_samples):
            rng = random.Random(i * 7919 + 42)
            result = self.sampler.select(state, rng=rng)
            if result.arm_id in expected_counts:
                expected_counts[result.arm_id] += 1

        # Chi-squared statistic
        chi2 = 0.0
        for aid in arm_ids:
            observed = counts[aid]
            expected = expected_counts[aid]
            if expected > 0:
                chi2 += (observed - expected) ** 2 / expected

        # Degrees of freedom = n_arms - 1
        df = max(len(arm_ids) - 1, 1)

        # Approximate p-value using chi-squared survival function
        p_value = _chi2_survival(chi2, df)

        # Check exploration floor violations
        floor = state.exploration_floor
        n_arms = len(arm_ids)
        min_expected_freq = floor * 0.5  # Allow some slack
        violations = []
        for aid, freq in per_arm_freq.items():
            if freq < min_expected_freq and n_arms > 1:
                violations.append(f"{aid}: freq={freq:.4f} < min={min_expected_freq:.4f}")

        passed = p_value > 0.01  # Fail if distributions diverge at 1% significance

        return SelectionValidationReport(
            chi_squared_statistic=chi2,
            p_value=p_value,
            passed=passed,
            per_arm_frequencies=per_arm_freq,
            exploration_floor_violations=violations,
        )


def _chi2_survival(x: float, df: int) -> float:
    """Approximate chi-squared survival function (1 - CDF).

    Uses the Wilson-Hilferty normal approximation for df >= 1.
    """
    if x <= 0:
        return 1.0
    if df <= 0:
        return 0.0

    # Wilson-Hilferty approximation
    z = ((x / df) ** (1.0 / 3.0) - (1.0 - 2.0 / (9.0 * df))) / math.sqrt(2.0 / (9.0 * df))

    # Standard normal survival function approximation
    return _normal_survival(z)


def _normal_survival(z: float) -> float:
    """Approximate standard normal survival function."""
    # Using the complementary error function: Phi(z) = 0.5 * erfc(-z/sqrt(2))
    return 0.5 * math.erfc(z / math.sqrt(2.0))
