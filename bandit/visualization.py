"""
CLI visualization of posterior state for the bandit.
"""

import math

from bandit.schemas import BanditState, ArmState


class PosteriorVisualizer:
    """Renders a fixed-width CLI table of bandit arm posteriors."""

    def render_cli(self, state: BanditState) -> str:
        """Render an 80-column table of all arms with stats.

        Shows: arm_id, Beta params, mean, 95% CI, temperature,
        attempts, successes, text histogram bar.
        """
        lines = []
        lines.append("=" * 80)
        lines.append(f"  Bandit Posterior  |  Iteration: {state.global_iteration}"
                     f"  |  Regime: {state.regime}")
        lines.append("=" * 80)

        # Header
        hdr = f"{'Arm':<16} {'a':>5} {'b':>5} {'Mean':>6} {'95% CI':>13} {'T':>6} {'Att':>4} {'Suc':>4} Hist"
        lines.append(hdr)
        lines.append("-" * 80)

        for arm_id, arm in sorted(state.arms.items()):
            if not isinstance(arm, ArmState):
                continue

            a = arm.alpha
            b = arm.beta
            mean = a / (a + b) if (a + b) > 0 else 0.0

            # 95% CI approximation using normal approximation to Beta
            ci_lo, ci_hi = _beta_ci(a, b)

            # Temperature
            temp = arm.temperature

            # Histogram bar (20 chars max, proportional to mean)
            bar_len = int(round(mean * 20))
            bar = "#" * bar_len + "." * (20 - bar_len)

            # Truncate arm_id for display
            name = arm_id[:15]

            line = (f"{name:<16} {a:5.1f} {b:5.1f} {mean:6.3f} "
                    f"[{ci_lo:.3f},{ci_hi:.3f}] {temp:6.4f} {arm.total_attempts:4d} "
                    f"{arm.total_successes:4d} {bar}")
            lines.append(line)

        lines.append("-" * 80)

        # Global stats
        total_attempts = sum(
            a.total_attempts for a in state.arms.values() if isinstance(a, ArmState)
        )
        total_successes = sum(
            a.total_successes for a in state.arms.values() if isinstance(a, ArmState)
        )
        overall_rate = total_successes / max(total_attempts, 1)
        n_delayed = len(state.delayed_corrections)

        lines.append(f"  Total: {total_attempts} attempts, {total_successes} successes "
                     f"({overall_rate:.1%})  |  Delayed corrections: {n_delayed}")
        lines.append(f"  T_base: {state.T_base:.4f}  |  Exploration floor: "
                     f"{state.exploration_floor:.3f}  |  "
                     f"Paper pref: {state.paper_preference_ratio:.2f}")
        lines.append("=" * 80)

        return "\n".join(lines)


def _beta_ci(a: float, b: float, z: float = 1.96) -> tuple:
    """Approximate 95% CI for Beta(a, b) using normal approximation.

    Returns (lower, upper) clamped to [0, 1].
    """
    if a + b <= 0:
        return (0.0, 1.0)

    mean = a / (a + b)
    var = (a * b) / ((a + b) ** 2 * (a + b + 1))
    std = math.sqrt(var) if var > 0 else 0.0

    lo = max(0.0, mean - z * std)
    hi = min(1.0, mean + z * std)
    return (lo, hi)
