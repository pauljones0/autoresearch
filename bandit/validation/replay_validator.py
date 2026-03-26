"""
Log replay validator -- reconstructs bandit state from log entries and
compares against the persisted strategy_state.json.
"""

import math

from bandit.schemas import (
    ArmState,
    BanditState,
    ReplayValidationReport,
    load_json,
    load_jsonl,
)
from bandit.temperature import TemperatureDeriver


class LogReplayValidator:
    """Replay bandit_log.jsonl from the warm_start entry, apply each
    posterior_update sequentially, and compare to saved state."""

    def __init__(self):
        self._temp_deriver = TemperatureDeriver()

    def validate(self, state_path: str, log_path: str) -> ReplayValidationReport:
        """Replay the log and compare reconstructed state to saved state.

        Returns a ReplayValidationReport with per-arm discrepancies.
        """
        report = ReplayValidationReport()

        saved_data = load_json(state_path)
        if not saved_data:
            report.per_arm_discrepancies["_global"] = "Cannot load state file"
            return report
        saved_state = BanditState.from_dict(saved_data)

        log_entries = load_jsonl(log_path)
        if not log_entries:
            report.per_arm_discrepancies["_global"] = "Cannot load log file"
            return report

        # Find warm_start entry to seed initial state
        reconstructed_arms = {}
        warm_start_found = False

        for entry in log_entries:
            event = entry.get("type", "")
            if event == "warm_start":
                warm_start_found = True
                initial_arms = entry.get("arms", {})
                for arm_id, arm_data in initial_arms.items():
                    if isinstance(arm_data, dict):
                        reconstructed_arms[arm_id] = ArmState.from_dict(arm_data)
                    else:
                        reconstructed_arms[arm_id] = ArmState()
                continue

            if not warm_start_found:
                # Also handle init events
                if event == "init":
                    initial_arms = entry.get("arms", {})
                    for arm_id, arm_data in initial_arms.items():
                        if isinstance(arm_data, dict):
                            reconstructed_arms[arm_id] = ArmState.from_dict(arm_data)
                        else:
                            reconstructed_arms[arm_id] = ArmState()
                    warm_start_found = True
                continue

            if event == "posterior_update":
                report.log_entries_replayed += 1
                arm_id = entry.get("arm_id", "")
                if arm_id not in reconstructed_arms:
                    reconstructed_arms[arm_id] = ArmState()

                arm = reconstructed_arms[arm_id]
                success = entry.get("success", False)
                if success:
                    arm.alpha += 1
                    arm.total_successes += 1
                else:
                    arm.beta += 1

                arm.total_attempts += 1

                # Update consecutive failures
                if success:
                    arm.consecutive_failures = 0
                else:
                    arm.consecutive_failures += 1

            elif event == "reheat":
                arm_id = entry.get("arm_id", "")
                if arm_id in reconstructed_arms:
                    arm = reconstructed_arms[arm_id]
                    arm.reheat_count += 1
                    new_temp = entry.get("temperature_after", arm.temperature)
                    arm.temperature = new_temp

        # Now update temperatures for reconstructed arms using the formula
        for arm_id, arm in reconstructed_arms.items():
            if arm.reheat_count == 0:
                arm.temperature = self._temp_deriver.compute(
                    arm, saved_state.T_base, saved_state.min_temperature
                )

        # Compare reconstructed vs saved
        alpha_tol = 0.001
        beta_tol = 0.001
        temp_tol = 0.01

        max_alpha_err = 0.0
        max_beta_err = 0.0
        max_temp_err = 0.0
        all_match = True

        all_arm_ids = set(saved_state.arms.keys()) | set(reconstructed_arms.keys())

        for arm_id in sorted(all_arm_ids):
            saved_arm = saved_state.arms.get(arm_id)
            recon_arm = reconstructed_arms.get(arm_id)

            if saved_arm is None:
                report.per_arm_discrepancies[arm_id] = "missing from saved state"
                all_match = False
                continue
            if recon_arm is None:
                report.per_arm_discrepancies[arm_id] = "missing from replayed state"
                all_match = False
                continue

            if not isinstance(saved_arm, ArmState):
                saved_arm = ArmState.from_dict(saved_arm) if isinstance(saved_arm, dict) else ArmState()

            discrepancies = []

            alpha_err = abs(saved_arm.alpha - recon_arm.alpha)
            beta_err = abs(saved_arm.beta - recon_arm.beta)
            temp_err = abs(saved_arm.temperature - recon_arm.temperature)

            max_alpha_err = max(max_alpha_err, alpha_err)
            max_beta_err = max(max_beta_err, beta_err)
            max_temp_err = max(max_temp_err, temp_err)

            if alpha_err > alpha_tol:
                discrepancies.append(
                    f"alpha: saved={saved_arm.alpha:.4f} replayed={recon_arm.alpha:.4f}"
                )
            if beta_err > beta_tol:
                discrepancies.append(
                    f"beta: saved={saved_arm.beta:.4f} replayed={recon_arm.beta:.4f}"
                )
            if temp_err > temp_tol:
                discrepancies.append(
                    f"temperature: saved={saved_arm.temperature:.6f} "
                    f"replayed={recon_arm.temperature:.6f}"
                )

            if discrepancies:
                report.per_arm_discrepancies[arm_id] = "; ".join(discrepancies)
                all_match = False

        report.max_alpha_error = max_alpha_err
        report.max_beta_error = max_beta_err
        report.max_temperature_error = max_temp_err
        report.replay_matches = all_match
        return report
