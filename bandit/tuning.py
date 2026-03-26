"""
Auto-tuner: analyzes bandit performance and recommends parameter adjustments.
"""

import math

from bandit.schemas import (
    BanditState, ArmState, TuningRecommendation, load_jsonl,
)


class AutoTuner:
    """Analyzes bandit state and logs to recommend parameter tuning."""

    def recommend(
        self,
        state: BanditState,
        log_path: str,
        interval: int = 100,
    ) -> list:
        """Generate tuning recommendations based on recent performance.

        Analyzes:
        - Convergence rate
        - Annealing acceptance rates
        - Arm dominance
        - Kernel exhaustion
        - Paper utilization

        Args:
            state: Current bandit state.
            log_path: Path to bandit log (JSONL).
            interval: Number of recent iterations to analyze.

        Returns:
            List of TuningRecommendation objects.
        """
        recommendations = []
        entries = load_jsonl(log_path)

        # Use last `interval` entries
        recent = entries[-interval:] if len(entries) > interval else entries

        if not recent:
            return recommendations

        # 1. Convergence analysis
        recommendations.extend(self._check_convergence(state, recent))

        # 2. Annealing rate analysis
        recommendations.extend(self._check_annealing_rates(state, recent))

        # 3. Arm dominance
        recommendations.extend(self._check_arm_dominance(state, recent))

        # 4. Kernel exhaustion
        recommendations.extend(self._check_kernel_exhaustion(state))

        # 5. Paper utilization
        recommendations.extend(self._check_paper_utilization(state, recent))

        return recommendations

    def _check_convergence(
        self, state: BanditState, recent: list
    ) -> list:
        """Check if bandit has converged too quickly or too slowly."""
        recs = []

        # Compute acceptance rate
        accepted = sum(1 for e in recent if e.get("accepted", False))
        acceptance_rate = accepted / len(recent) if recent else 0.0

        # Too high acceptance = explore more (raise T_base or exploration_floor)
        if acceptance_rate > 0.9 and len(recent) >= 50:
            recs.append(TuningRecommendation(
                parameter="exploration_floor",
                current_value=state.exploration_floor,
                recommended_value=min(state.exploration_floor * 1.5, 0.3),
                reason=f"Acceptance rate {acceptance_rate:.1%} is very high; "
                       f"increase exploration to avoid premature convergence.",
                confidence="medium",
                auto_applicable=True,
            ))

        # Too low acceptance = temperatures may be too low
        if acceptance_rate < 0.1 and len(recent) >= 50:
            recs.append(TuningRecommendation(
                parameter="T_base",
                current_value=state.T_base,
                recommended_value=state.T_base * 2.0,
                reason=f"Acceptance rate {acceptance_rate:.1%} is very low; "
                       f"increase T_base to allow more exploration.",
                confidence="medium",
                auto_applicable=True,
            ))

        return recs

    def _check_annealing_rates(
        self, state: BanditState, recent: list
    ) -> list:
        """Check annealing acceptance rates for temperature tuning."""
        recs = []

        annealing_accepts = sum(
            1 for e in recent if e.get("accepted_by") == "annealing")
        total_regressions = sum(
            1 for e in recent
            if e.get("delta") is not None and e["delta"] > 0)

        if total_regressions > 0:
            annealing_rate = annealing_accepts / total_regressions
        else:
            return recs

        # Annealing rate too high -> cool down
        if annealing_rate > 0.5 and total_regressions >= 20:
            recs.append(TuningRecommendation(
                parameter="T_base",
                current_value=state.T_base,
                recommended_value=state.T_base * 0.7,
                reason=f"Annealing acceptance rate {annealing_rate:.1%} of "
                       f"regressions is high; reduce T_base to be more selective.",
                confidence="high",
                auto_applicable=True,
            ))

        # Annealing rate too low -> temperatures already cold
        if annealing_rate < 0.05 and total_regressions >= 20:
            recs.append(TuningRecommendation(
                parameter="reheat_factor",
                current_value=state.reheat_factor,
                recommended_value=min(state.reheat_factor * 1.5, 10.0),
                reason=f"Annealing acceptance rate {annealing_rate:.1%} is very low; "
                       f"increase reheat_factor for stronger reheats.",
                confidence="low",
                auto_applicable=False,
            ))

        return recs

    def _check_arm_dominance(
        self, state: BanditState, recent: list
    ) -> list:
        """Check if one arm is dominating selections."""
        recs = []

        arm_counts = {}
        for e in recent:
            arm_id = e.get("arm_selected", "")
            if arm_id:
                arm_counts[arm_id] = arm_counts.get(arm_id, 0) + 1

        if not arm_counts:
            return recs

        total = sum(arm_counts.values())
        max_arm = max(arm_counts, key=arm_counts.get)
        max_frac = arm_counts[max_arm] / total

        n_arms = len(state.arms)
        if n_arms <= 1:
            return recs

        # If one arm gets >60% of selections in a multi-arm setup
        if max_frac > 0.6 and n_arms >= 4:
            recs.append(TuningRecommendation(
                parameter="exploration_floor",
                current_value=state.exploration_floor,
                recommended_value=min(state.exploration_floor * 2.0, 0.2),
                reason=f"Arm {max_arm} dominates at {max_frac:.1%} of selections; "
                       f"raise exploration floor to diversify.",
                confidence="medium",
                auto_applicable=True,
            ))

        return recs

    def _check_kernel_exhaustion(self, state: BanditState) -> list:
        """Check if kernel arms are exhausted (many attempts, no successes)."""
        recs = []

        for arm_id, arm in state.arms.items():
            if not isinstance(arm, ArmState):
                continue
            if arm.source_type != "kernel":
                continue

            if arm.total_attempts >= 30 and arm.total_successes == 0:
                recs.append(TuningRecommendation(
                    parameter="K_reheat_threshold",
                    current_value=float(state.K_reheat_threshold),
                    recommended_value=float(max(state.K_reheat_threshold - 1, 2)),
                    reason=f"Kernel arm {arm_id} has {arm.total_attempts} attempts "
                           f"with 0 successes; lower reheat threshold to recover faster.",
                    confidence="low",
                    auto_applicable=False,
                ))

        return recs

    def _check_paper_utilization(
        self, state: BanditState, recent: list
    ) -> list:
        """Check paper-path utilization."""
        recs = []

        paper_dispatches = sum(
            1 for e in recent if e.get("dispatch_path") == "paper")
        total = len(recent)

        if total < 30:
            return recs

        paper_rate = paper_dispatches / total

        # If paper_preference_ratio is high but actual paper dispatches are low
        if state.paper_preference_ratio > 0.3 and paper_rate < 0.05:
            recs.append(TuningRecommendation(
                parameter="paper_preference_ratio",
                current_value=state.paper_preference_ratio,
                recommended_value=max(state.paper_preference_ratio * 0.5, 0.1),
                reason=f"Paper preference is {state.paper_preference_ratio:.2f} "
                       f"but only {paper_rate:.1%} dispatches use papers; "
                       f"reduce ratio to match actual availability.",
                confidence="medium",
                auto_applicable=True,
            ))

        return recs
