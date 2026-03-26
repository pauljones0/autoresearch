"""
Automatic kernel recovery agent.

After a kernel is auto-disabled, diagnoses the issue, re-runs verification,
and either re-enables (transient issue) or keeps disabled and triggers
mutation for replacement (genuine issue).
"""

import json
import os
import time


class KernelAutoRecoveryAgent:
    """Attempt automatic recovery of auto-disabled kernels."""

    def __init__(self, data_dir: str = None):
        self.data_dir = data_dir or os.path.join(
            os.path.dirname(__file__), "..", "data"
        )

    def attempt_recovery(
        self,
        kernel_id: str,
        disable_reason: str,
        verification_pipeline=None,
        max_retry_checks: int = 3,
        increased_check_interval: int = 50,
    ) -> dict:
        """
        Attempt to recover an auto-disabled kernel.

        Re-runs verification with current weights. If verification passes,
        diagnoses as transient and re-enables with increased monitoring.
        If it fails, diagnoses as genuine and keeps disabled, optionally
        triggering a mutation for replacement.

        Args:
            kernel_id: The disabled kernel ID.
            disable_reason: Why the kernel was disabled.
            verification_pipeline: Optional verification callable.
                Should accept kernel_id and return dict with 'passed' key.
            max_retry_checks: Number of re-verification attempts.
            increased_check_interval: Monitoring interval after recovery.

        Returns:
            dict with keys:
                recovered: bool — kernel was re-enabled
                diagnosis: str — "transient" or "genuine"
                action_taken: str — description of action taken
        """
        result = {
            "recovered": False,
            "diagnosis": "unknown",
            "action_taken": "",
        }

        # Re-run verification
        verification_passed = False
        for attempt in range(1, max_retry_checks + 1):
            passed = self._verify_kernel(kernel_id, verification_pipeline)
            if passed:
                verification_passed = True
                break

        if verification_passed:
            # Transient issue — re-enable with increased monitoring
            result["recovered"] = True
            result["diagnosis"] = "transient"
            result["action_taken"] = (
                f"Re-enabled kernel '{kernel_id}' with increased monitoring "
                f"(check every {increased_check_interval} steps). "
                f"Original disable reason: {disable_reason}"
            )
            self._re_enable_kernel(kernel_id, increased_check_interval)
        else:
            # Genuine issue — keep disabled, trigger mutation
            result["recovered"] = False
            result["diagnosis"] = "genuine"
            result["action_taken"] = (
                f"Kernel '{kernel_id}' kept disabled. Genuine issue confirmed: "
                f"{disable_reason}. Queued for evolutionary replacement."
            )
            self._queue_for_replacement(kernel_id, disable_reason)

        self._log_recovery_attempt(kernel_id, result)
        return result

    def _verify_kernel(self, kernel_id: str, verification_pipeline) -> bool:
        """Run verification for a kernel."""
        if verification_pipeline is not None:
            try:
                vr = verification_pipeline(kernel_id)
                if isinstance(vr, dict):
                    return vr.get("passed", False)
                if hasattr(vr, "passed"):
                    return vr.passed
            except Exception:
                return False

        # Without a pipeline, check if a recent verification report exists
        report_path = os.path.join(
            self.data_dir, "verification_reports", f"{kernel_id}.json"
        )
        if os.path.exists(report_path):
            try:
                with open(report_path) as f:
                    report = json.load(f)
                return report.get("verdict") == "PASS"
            except (json.JSONDecodeError, IOError):
                pass
        return False

    def _re_enable_kernel(self, kernel_id: str, check_interval: int):
        """Re-enable kernel in config with increased monitoring."""
        config_path = os.path.join(self.data_dir, "kernel_config.json")
        config = {}
        if os.path.exists(config_path):
            try:
                with open(config_path) as f:
                    config = json.load(f)
            except (json.JSONDecodeError, IOError):
                pass

        if kernel_id in config and isinstance(config[kernel_id], dict):
            config[kernel_id]["enabled"] = True
            config[kernel_id]["backend"] = "triton"
            config[kernel_id]["check_interval"] = check_interval
            config[kernel_id]["recovery_timestamp"] = time.time()
        else:
            config[kernel_id] = {
                "enabled": True,
                "backend": "triton",
                "check_interval": check_interval,
                "recovery_timestamp": time.time(),
            }

        try:
            os.makedirs(os.path.dirname(config_path) or ".", exist_ok=True)
            with open(config_path, "w") as f:
                json.dump(config, f, indent=2)
        except OSError:
            pass

    def _queue_for_replacement(self, kernel_id: str, reason: str):
        """Queue disabled kernel for evolutionary replacement."""
        queue_path = os.path.join(self.data_dir, "mutation_queue.jsonl")
        entry = {
            "kernel_id": kernel_id,
            "action": "replace",
            "reason": reason,
            "timestamp": time.time(),
        }
        try:
            os.makedirs(os.path.dirname(queue_path) or ".", exist_ok=True)
            with open(queue_path, "a") as f:
                f.write(json.dumps(entry) + "\n")
        except OSError:
            pass

    def _log_recovery_attempt(self, kernel_id: str, result: dict):
        """Log recovery attempt to audit trail."""
        log_path = os.path.join(self.data_dir, "recovery_log.jsonl")
        entry = {
            "kernel_id": kernel_id,
            "timestamp": time.time(),
            **result,
        }
        try:
            os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
            with open(log_path, "a") as f:
                f.write(json.dumps(entry) + "\n")
        except OSError:
            pass
