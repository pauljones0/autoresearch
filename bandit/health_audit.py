"""
Comprehensive health audit for the Adaptive Bandit pipeline.
"""

import json
import math

from bandit.schemas import (
    ArmState,
    BanditState,
    BanditAuditReport,
    load_json,
    load_jsonl,
)
from bandit.temperature import TemperatureDeriver


class BanditHealthAuditor:
    """Runs comprehensive health checks across bandit state, logs, and
    cross-system artifacts.  Returns a single BanditAuditReport."""

    def __init__(self):
        self._temp_deriver = TemperatureDeriver()

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def audit(
        self,
        state: BanditState,
        journal_path: str,
        log_path: str,
        queue_path: str,
        kernel_config_path: str,
    ) -> BanditAuditReport:
        """Run all audit checks and aggregate into a single report."""
        report = BanditAuditReport()
        checks = [
            ("state_integrity", self._check_state_integrity),
            ("evidence_conservation", self._check_evidence_conservation),
            ("journal_log_alignment", self._check_journal_log_alignment),
            ("temperature_consistency", self._check_temperature_consistency),
            ("queue_coherence", self._check_queue_coherence),
            ("kernel_alignment", self._check_kernel_alignment),
        ]
        passed = 0
        for name, fn in checks:
            report.checks_run += 1
            try:
                issues = fn(state, journal_path, log_path, queue_path, kernel_config_path)
            except Exception as exc:
                issues = [f"{name}: unexpected error: {exc}"]
            if not issues:
                passed += 1
            else:
                report.issues.extend(issues)
        report.checks_passed = passed
        report.all_clear = passed == report.checks_run
        return report

    # ------------------------------------------------------------------
    # Individual checks
    # ------------------------------------------------------------------

    def _check_state_integrity(self, state, journal_path, log_path, queue_path, kernel_config_path):
        """Validate state fields: arm_ids valid, alpha/beta >= 1, temps >= 0,
        global_iteration matches log entry count."""
        issues = state.validate()  # built-in checks

        # global_iteration vs log
        log_entries = load_jsonl(log_path)
        posterior_updates = [e for e in log_entries if e.get("type") == "posterior_update"]
        if state.global_iteration != len(posterior_updates):
            issues.append(
                f"global_iteration={state.global_iteration} does not match "
                f"posterior_update count={len(posterior_updates)} in log"
            )
        return issues

    def _check_evidence_conservation(self, state, journal_path, log_path, queue_path, kernel_config_path):
        """Sum of (alpha_i - 1 + beta_i - 1) across all arms must equal the
        number of posterior_update entries in the log."""
        issues = []
        log_entries = load_jsonl(log_path)
        posterior_updates = [e for e in log_entries if e.get("type") == "posterior_update"]
        n_updates = len(posterior_updates)

        total_evidence = 0.0
        for arm_id, arm in state.arms.items():
            if not isinstance(arm, ArmState):
                continue
            total_evidence += (arm.alpha - 1) + (arm.beta - 1)

        if abs(total_evidence - n_updates) > 0.01:
            issues.append(
                f"Evidence conservation failed: sum(alpha_i-1 + beta_i-1)="
                f"{total_evidence:.4f} != posterior_update count={n_updates}"
            )
        return issues

    def _check_journal_log_alignment(self, state, journal_path, log_path, queue_path, kernel_config_path):
        """Check that every posterior_update in the log references a valid
        journal entry ID."""
        issues = []
        log_entries = load_jsonl(log_path)
        journal_data = load_json(journal_path)

        # Build set of journal entry IDs
        journal_ids = set()
        if isinstance(journal_data, list):
            for entry in journal_data:
                if isinstance(entry, dict):
                    eid = entry.get("id") or entry.get("entry_id", "")
                    if eid:
                        journal_ids.add(eid)
        elif isinstance(journal_data, dict):
            for eid in journal_data:
                journal_ids.add(eid)

        for i, entry in enumerate(log_entries):
            if entry.get("type") != "posterior_update":
                continue
            ref = entry.get("journal_entry_id", "")
            if ref and journal_ids and ref not in journal_ids:
                issues.append(
                    f"Log entry {i}: journal_entry_id={ref!r} not found in journal"
                )
        return issues

    def _check_temperature_consistency(self, state, journal_path, log_path, queue_path, kernel_config_path):
        """Each arm's temperature must match the formula-derived value within
        tolerance, accounting for reheats."""
        issues = []
        tolerance = 0.0001
        for arm_id, arm in state.arms.items():
            if not isinstance(arm, ArmState):
                continue
            expected = self._temp_deriver.compute(arm, state.T_base, state.min_temperature)
            # Account for reheat: reheated temp = expected * reheat_factor^reheat_count
            # but decays back.  If arm has been reheated, the stored temperature
            # may be higher than the formula value.  We accept it if it is at
            # most reheat_factor * expected (one reheat not yet decayed).
            if arm.reheat_count > 0:
                upper_bound = expected * (state.reheat_factor ** arm.reheat_count)
                if arm.temperature > upper_bound + tolerance:
                    issues.append(
                        f"Arm {arm_id}: temperature={arm.temperature:.6f} exceeds "
                        f"reheat upper bound={upper_bound:.6f}"
                    )
                elif arm.temperature < expected - tolerance:
                    issues.append(
                        f"Arm {arm_id}: temperature={arm.temperature:.6f} below "
                        f"expected={expected:.6f} (reheat_count={arm.reheat_count})"
                    )
            else:
                if abs(arm.temperature - expected) > tolerance:
                    issues.append(
                        f"Arm {arm_id}: temperature={arm.temperature:.6f} != "
                        f"expected={expected:.6f} (diff={abs(arm.temperature - expected):.6f})"
                    )
        return issues

    def _check_queue_coherence(self, state, journal_path, log_path, queue_path, kernel_config_path):
        """Verify queue entries reference valid arm IDs and have valid status."""
        issues = []
        queue_data = load_json(queue_path)
        if not queue_data:
            return issues  # No queue file is acceptable

        entries = queue_data if isinstance(queue_data, list) else queue_data.get("entries", [])
        valid_arm_ids = set(state.arms.keys())
        valid_statuses = {"pending", "in_progress", "completed", "failed", "skipped"}

        for i, entry in enumerate(entries):
            if not isinstance(entry, dict):
                continue
            arm_id = entry.get("arm_id") or entry.get("category", "")
            status = entry.get("status", "")
            if arm_id and valid_arm_ids and arm_id not in valid_arm_ids:
                issues.append(
                    f"Queue entry {i}: arm_id={arm_id!r} not in bandit state"
                )
            if status and status not in valid_statuses:
                issues.append(
                    f"Queue entry {i}: invalid status={status!r}"
                )
        return issues

    def _check_kernel_alignment(self, state, journal_path, log_path, queue_path, kernel_config_path):
        """Verify kernel config references valid kernel arm IDs."""
        issues = []
        kernel_config = load_json(kernel_config_path)
        if not kernel_config:
            return issues  # No config is acceptable

        kernel_arm_ids = {
            aid for aid, arm in state.arms.items()
            if isinstance(arm, ArmState) and arm.source_type == "kernel"
        }
        config_arms = kernel_config.get("arms", kernel_config.get("kernels", []))
        if isinstance(config_arms, list):
            for entry in config_arms:
                if isinstance(entry, dict):
                    kid = entry.get("arm_id", "")
                    if kid and kernel_arm_ids and kid not in kernel_arm_ids:
                        issues.append(
                            f"Kernel config references arm_id={kid!r} "
                            f"not in kernel arms: {sorted(kernel_arm_ids)}"
                        )
        return issues
