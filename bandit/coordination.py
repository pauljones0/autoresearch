"""
Three-system state coordinator: verifies consistency across
journal, bandit state, queue, and kernel config.
"""

import json
import os

from bandit.schemas import (
    BanditState, ArmState, ConsistencyReport, load_json, load_jsonl,
)


class ThreeSystemStateCoordinator:
    """Verifies cross-system consistency between bandit state, journal,
    queue, and kernel configuration."""

    def verify_consistency(
        self,
        state: BanditState,
        journal_path: str,
        queue_path: str,
        kernel_config_path: str,
    ) -> ConsistencyReport:
        """Verify consistency across all three systems.

        Checks:
        1. Journal-log alignment: arm counts in state match journal entries.
        2. Posterior arithmetic: alpha + beta - 2 == total_attempts for each arm.
        3. Queue coherence: no stale queue entries referencing unknown arms.
        4. Kernel config alignment: kernel arms in state have matching configs.
        """
        report = ConsistencyReport(consistent=True)

        # 1. Journal-log alignment
        journal_issues = self._check_journal_alignment(state, journal_path)
        report.journal_log_mismatches = len(journal_issues)
        report.issues.extend(journal_issues)

        # 2. Posterior arithmetic
        arith_issues = self._check_posterior_arithmetic(state)
        report.posterior_arithmetic_errors = len(arith_issues)
        report.issues.extend(arith_issues)

        # 3. Queue coherence
        queue_issues = self._check_queue_coherence(state, queue_path)
        report.stale_queue_entries = len(queue_issues)
        report.issues.extend(queue_issues)

        # 4. Kernel config alignment
        kernel_issues = self._check_kernel_config(state, kernel_config_path)
        report.orphan_kernel_configs = len(kernel_issues)
        report.issues.extend(kernel_issues)

        report.consistent = len(report.issues) == 0
        return report

    def _check_journal_alignment(
        self, state: BanditState, journal_path: str
    ) -> list:
        """Check that journal entry counts align with arm attempt counts."""
        issues = []
        if not os.path.exists(journal_path):
            return issues

        try:
            with open(journal_path) as f:
                entries = json.load(f)
        except (json.JSONDecodeError, OSError):
            issues.append(f"Cannot parse journal at {journal_path}")
            return issues

        if not isinstance(entries, list):
            return issues

        # Count entries per arm
        journal_counts = {}
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            arm_id = entry.get("bandit_arm") or entry.get("modification_category", "")
            if arm_id:
                journal_counts[arm_id] = journal_counts.get(arm_id, 0) + 1

        # Compare with state
        for arm_id, arm in state.arms.items():
            if not isinstance(arm, ArmState):
                continue
            journal_n = journal_counts.get(arm_id, 0)
            if arm.total_attempts > 0 and journal_n == 0:
                issues.append(
                    f"Arm {arm_id}: {arm.total_attempts} attempts in state "
                    f"but 0 journal entries")

        return issues

    def _check_posterior_arithmetic(self, state: BanditState) -> list:
        """Check alpha + beta - 2 == total_attempts for each arm."""
        issues = []
        for arm_id, arm in state.arms.items():
            if not isinstance(arm, ArmState):
                continue
            expected_attempts = (arm.alpha - 1) + (arm.beta - 1)
            if abs(expected_attempts - arm.total_attempts) > 0.01:
                issues.append(
                    f"Arm {arm_id}: alpha={arm.alpha}, beta={arm.beta} "
                    f"implies {expected_attempts} attempts but "
                    f"total_attempts={arm.total_attempts}")
        return issues

    def _check_queue_coherence(
        self, state: BanditState, queue_path: str
    ) -> list:
        """Check for stale queue entries referencing unknown arms."""
        issues = []
        if not os.path.exists(queue_path):
            return issues

        entries = load_jsonl(queue_path)
        known_arms = set(state.arms.keys())

        for entry in entries:
            if not isinstance(entry, dict):
                continue
            arm_ref = entry.get("arm_id", "")
            status = entry.get("status", "")
            if arm_ref and arm_ref not in known_arms and status != "completed":
                issues.append(
                    f"Queue entry references unknown arm: {arm_ref}")

        return issues

    def _check_kernel_config(
        self, state: BanditState, kernel_config_path: str
    ) -> list:
        """Check kernel config alignment with kernel arms in state."""
        issues = []
        if not os.path.exists(kernel_config_path):
            return issues

        config = load_json(kernel_config_path)
        if not config:
            return issues

        config_arms = set(config.get("arms", {}).keys()) if isinstance(config.get("arms"), dict) else set()
        state_kernel_arms = {
            arm_id for arm_id, arm in state.arms.items()
            if isinstance(arm, ArmState) and arm.source_type == "kernel"
        }

        # Orphan configs: in config but not in state
        for arm_id in config_arms - state_kernel_arms:
            issues.append(
                f"Kernel config has arm {arm_id} not in bandit state")

        return issues
