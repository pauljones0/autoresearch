"""
Posterior health checker — detects pathological bandit states.
"""

import time

from bandit.schemas import BanditState, ArmState, HealthAlert


class PosteriorHealthChecker:
    """Checks bandit state for pathological conditions."""

    def check(self, state: BanditState) -> list:
        """Run all health checks and return list of HealthAlerts.

        Checks for:
        - Starvation: arms not selected for too long
        - Collapse: one arm dominates > 80% of selections
        - Stuck arms: high consecutive failures
        - Dominance: one arm's mean >> all others
        - Evidence conservation: alpha + beta should grow with attempts
        - Delayed correction backlog
        """
        alerts = []
        alerts.extend(self._check_starvation(state))
        alerts.extend(self._check_collapse(state))
        alerts.extend(self._check_stuck(state))
        alerts.extend(self._check_dominance(state))
        alerts.extend(self._check_evidence_conservation(state))
        alerts.extend(self._check_delayed_backlog(state))
        return alerts

    def _check_starvation(self, state: BanditState) -> list:
        """Check for arms that haven't been selected in too long."""
        alerts = []
        now = time.time()
        for arm_id, arm in state.arms.items():
            if not isinstance(arm, ArmState):
                continue
            if arm.last_selected == 0.0:
                continue
            gap = now - arm.last_selected
            # Warn if not selected in > 1 hour and iteration > 10
            if gap > 3600 and state.global_iteration > 10:
                alerts.append(HealthAlert(
                    severity="warning",
                    arm_id=arm_id,
                    message=f"Arm not selected for {gap/3600:.1f} hours",
                    recommendation="Consider increasing exploration floor or diagnostics boost",
                ))
        return alerts

    def _check_collapse(self, state: BanditState) -> list:
        """Check if one arm dominates > 80% of total attempts."""
        alerts = []
        total = sum(
            a.total_attempts for a in state.arms.values() if isinstance(a, ArmState)
        )
        if total < 20:
            return alerts

        for arm_id, arm in state.arms.items():
            if not isinstance(arm, ArmState):
                continue
            ratio = arm.total_attempts / max(total, 1)
            if ratio > 0.8:
                alerts.append(HealthAlert(
                    severity="warning",
                    arm_id=arm_id,
                    message=f"Arm has {ratio:.0%} of all attempts — possible collapse",
                    recommendation="Increase exploration floor or review arm rewards",
                ))
        return alerts

    def _check_stuck(self, state: BanditState) -> list:
        """Check for arms with high consecutive failures."""
        alerts = []
        for arm_id, arm in state.arms.items():
            if not isinstance(arm, ArmState):
                continue
            if arm.consecutive_failures >= 10:
                severity = "critical" if arm.consecutive_failures >= 20 else "warning"
                alerts.append(HealthAlert(
                    severity=severity,
                    arm_id=arm_id,
                    message=f"Arm has {arm.consecutive_failures} consecutive failures",
                    recommendation="Consider reheat or disabling this arm temporarily",
                ))
        return alerts

    def _check_dominance(self, state: BanditState) -> list:
        """Check if one arm's posterior mean >> all others."""
        alerts = []
        means = {}
        for arm_id, arm in state.arms.items():
            if not isinstance(arm, ArmState):
                continue
            means[arm_id] = arm.alpha / (arm.alpha + arm.beta)

        if len(means) < 2:
            return alerts

        sorted_means = sorted(means.values(), reverse=True)
        if sorted_means[0] > 0.9 and sorted_means[1] < 0.3:
            dominant = max(means, key=means.get)
            alerts.append(HealthAlert(
                severity="info",
                arm_id=dominant,
                message=f"Arm strongly dominates (mean={sorted_means[0]:.3f} vs next={sorted_means[1]:.3f})",
                recommendation="May be correct — verify with diagnostics",
            ))
        return alerts

    def _check_evidence_conservation(self, state: BanditState) -> list:
        """Check that alpha + beta - 2 == total_attempts for each arm."""
        alerts = []
        for arm_id, arm in state.arms.items():
            if not isinstance(arm, ArmState):
                continue
            evidence = (arm.alpha - 1.0) + (arm.beta - 1.0)
            expected = arm.total_attempts
            if abs(evidence - expected) > 0.5 and expected > 0:
                alerts.append(HealthAlert(
                    severity="warning",
                    arm_id=arm_id,
                    message=(f"Evidence mismatch: alpha+beta-2={evidence:.1f} "
                             f"vs attempts={expected}"),
                    recommendation="Check for missed updates or double-counting",
                ))
        return alerts

    def _check_delayed_backlog(self, state: BanditState) -> list:
        """Check for growing delayed correction backlog."""
        alerts = []
        n = len(state.delayed_corrections)
        if n > 10:
            alerts.append(HealthAlert(
                severity="critical",
                message=f"Large delayed correction backlog: {n} pending",
                recommendation="Investigate why corrections are not being applied",
            ))
        elif n > 5:
            alerts.append(HealthAlert(
                severity="warning",
                message=f"Delayed correction backlog: {n} pending",
                recommendation="Check if target arms exist in state",
            ))
        return alerts
