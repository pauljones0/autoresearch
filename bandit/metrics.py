"""
Bandit-derived metrics: proposal and computation.
"""

import math

from bandit.schemas import ArmState, BanditState, load_jsonl


# ---------------------------------------------------------------------------
# Digamma approximation (standard library only)
# ---------------------------------------------------------------------------

def _digamma(x: float) -> float:
    """Digamma function approximation using Stirling's series.

    For x > 6:  psi(x) ~ ln(x) - 1/(2x) - 1/(12x^2)
    For x <= 6: use recursion  psi(x) = psi(x+1) - 1/x
    """
    result = 0.0
    # Shift x upward until x > 6
    while x <= 6.0:
        result -= 1.0 / x
        x += 1.0
    # Stirling approximation for large x
    result += math.log(x) - 1.0 / (2.0 * x) - 1.0 / (12.0 * x * x)
    return result


def _log_beta(a: float, b: float) -> float:
    """Log of the Beta function: ln B(a,b) = ln Gamma(a) + ln Gamma(b) - ln Gamma(a+b)."""
    return math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)


# ---------------------------------------------------------------------------
# Metric Proposer
# ---------------------------------------------------------------------------

class BanditMetricProposer:
    """Proposes bandit-derived metrics for dashboard monitoring."""

    def propose(self, state: BanditState, existing_metrics: list) -> list:
        """Return a list of 5 metric definitions.

        Each definition is a dict with keys: name, description, category.
        """
        existing_names = set()
        for m in existing_metrics:
            if isinstance(m, dict):
                existing_names.add(m.get("name", ""))
            elif isinstance(m, str):
                existing_names.add(m)

        definitions = [
            {
                "name": "arm_selection_entropy",
                "description": (
                    "Shannon entropy of the Thompson Sampling selection "
                    "distribution. Higher values indicate more exploratory "
                    "behavior; lower values indicate convergence toward "
                    "exploitation of a few arms."
                ),
                "category": "exploration",
            },
            {
                "name": "annealing_stepping_stone_rate",
                "description": (
                    "Fraction of annealing-accepted (non-greedy) iterations "
                    "that led to subsequent improvements within a window. "
                    "Measures whether stochastic acceptance is productive."
                ),
                "category": "annealing",
            },
            {
                "name": "posterior_kl_divergence_from_prior",
                "description": (
                    "Mean KL divergence of arm posteriors Beta(alpha, beta) "
                    "from the uninformative prior Beta(1,1). Tracks how far "
                    "the bandit has moved from ignorance."
                ),
                "category": "learning",
            },
            {
                "name": "temperature_dispersion_ratio",
                "description": (
                    "Ratio max(T)/min(T) across all arms. A high ratio "
                    "indicates heterogeneous exploration pressure."
                ),
                "category": "annealing",
            },
            {
                "name": "regime_change_frequency",
                "description": (
                    "Number of regime changes (success rate drops) detected "
                    "in the recent log window. Elevated values signal "
                    "environmental instability."
                ),
                "category": "stability",
            },
        ]
        return definitions


# ---------------------------------------------------------------------------
# Metric Computer
# ---------------------------------------------------------------------------

class BanditMetricComputer:
    """Computes current values of all bandit-derived metrics."""

    def compute_all(
        self,
        state: BanditState,
        log_path: str,
        window: int = 50,
    ) -> dict:
        """Compute all 5 metrics and return as name -> float dict."""
        log_entries = load_jsonl(log_path)
        return {
            "arm_selection_entropy": self._arm_selection_entropy(state),
            "annealing_stepping_stone_rate": self._stepping_stone_rate(log_entries, window),
            "posterior_kl_divergence_from_prior": self._posterior_kl(state),
            "temperature_dispersion_ratio": self._temp_dispersion(state),
            "regime_change_frequency": self._regime_change_freq(log_entries, window),
        }

    # ------------------------------------------------------------------
    # Individual metric implementations
    # ------------------------------------------------------------------

    def _arm_selection_entropy(self, state: BanditState) -> float:
        """Shannon entropy of the selection probability distribution.

        Uses mean of each arm's Beta as the unnormalized weight,
        then normalises to get a distribution.
        """
        means = []
        for arm_id, arm in state.arms.items():
            if not isinstance(arm, ArmState):
                continue
            mean = arm.alpha / (arm.alpha + arm.beta)
            means.append(max(mean, 1e-12))

        if not means:
            return 0.0

        total = sum(means)
        probs = [m / total for m in means]
        entropy = 0.0
        for p in probs:
            if p > 0:
                entropy -= p * math.log(p)
        return entropy

    def _stepping_stone_rate(self, log_entries: list, window: int) -> float:
        """Fraction of annealing-accepted iterations that led to a subsequent
        improvement within the window."""
        recent = log_entries[-window:] if len(log_entries) > window else log_entries
        annealing_accepted = []
        for i, entry in enumerate(recent):
            if entry.get("type") == "acceptance" and entry.get("accepted_by") == "annealing":
                annealing_accepted.append(i)

        if not annealing_accepted:
            return 0.0

        stepping_stones = 0
        for idx in annealing_accepted:
            # Look at subsequent entries for an improvement
            for j in range(idx + 1, len(recent)):
                subsequent = recent[j]
                if subsequent.get("type") == "acceptance":
                    if subsequent.get("accepted_by") == "improvement":
                        stepping_stones += 1
                        break
                    elif subsequent.get("accepted_by") == "rejected":
                        break

        return stepping_stones / len(annealing_accepted)

    def _posterior_kl(self, state: BanditState) -> float:
        """Mean KL divergence from Beta(1,1) prior across all arms.

        KL(Beta(a,b) || Beta(1,1)) =
            ln(B(1,1)/B(a,b)) + (a-1)*psi(a) + (b-1)*psi(b) - (a+b-2)*psi(a+b)

        where B(1,1) = 1 so ln(B(1,1)) = 0.
        """
        kl_values = []
        for arm_id, arm in state.arms.items():
            if not isinstance(arm, ArmState):
                continue
            a = arm.alpha
            b = arm.beta
            if a == 1.0 and b == 1.0:
                kl_values.append(0.0)
                continue
            # ln(B(1,1)/B(a,b)) = -ln(B(a,b)) since B(1,1) = 1
            log_beta_ab = _log_beta(a, b)
            psi_a = _digamma(a)
            psi_b = _digamma(b)
            psi_ab = _digamma(a + b)

            kl = -log_beta_ab + (a - 1) * psi_a + (b - 1) * psi_b - (a + b - 2) * psi_ab
            kl_values.append(max(kl, 0.0))  # KL >= 0 by definition

        if not kl_values:
            return 0.0
        return sum(kl_values) / len(kl_values)

    def _temp_dispersion(self, state: BanditState) -> float:
        """Ratio max(T) / min(T) across all arms."""
        temps = []
        for arm_id, arm in state.arms.items():
            if not isinstance(arm, ArmState):
                continue
            temps.append(arm.temperature)

        if not temps or min(temps) <= 0:
            return 0.0
        return max(temps) / min(temps)

    def _regime_change_freq(self, log_entries: list, window: int) -> float:
        """Count regime_change events in the recent window."""
        recent = log_entries[-window:] if len(log_entries) > window else log_entries
        count = 0
        for entry in recent:
            if entry.get("type") == "regime_change":
                count += 1
        return float(count)
