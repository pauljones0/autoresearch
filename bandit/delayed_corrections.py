"""
Delayed correction receiver for kernel arm penalties.
"""

import time

from bandit.schemas import BanditState, ArmState


class DelayedCorrectionReceiver:
    """Queues delayed beta corrections when critical alerts disable a kernel."""

    def receive_alert(self, alert, state: BanditState, log_writer=None):
        """Queue a delayed beta correction for kernel arm.

        When a critical alert disables a kernel, this adds a delayed
        correction that will increase beta (penalize) the kernel arm
        on the next posterior update.

        Args:
            alert: Alert object or dict with severity, arm_id, message.
            state: Current bandit state (modified in place to add correction).
            log_writer: Optional log writer.
        """
        # Extract alert fields
        if isinstance(alert, dict):
            severity = alert.get("severity", "")
            arm_id = alert.get("arm_id", "")
            message = alert.get("message", "")
        else:
            severity = getattr(alert, "severity", "")
            arm_id = getattr(alert, "arm_id", "")
            message = getattr(alert, "message", "")

        # Only queue corrections for critical alerts on kernel arms
        if severity != "critical":
            return

        # Find the kernel arm to penalize
        target_arm = arm_id
        if not target_arm:
            # Try to find any kernel arm
            for aid, arm in state.arms.items():
                if isinstance(arm, ArmState) and arm.source_type == "kernel":
                    target_arm = aid
                    break

        if not target_arm:
            return

        # Queue the delayed correction
        correction = {
            "arm_id": target_arm,
            "beta_add": 2.0,  # Significant penalty
            "reason": f"critical alert: {message}",
            "queued_at": time.time(),
        }
        state.delayed_corrections.append(correction)

        # Log
        if log_writer is not None:
            log_entry = {
                "type": "delayed_correction_queued",
                "arm_id": target_arm,
                "beta_add": 2.0,
                "reason": message,
                "timestamp": time.time(),
            }
            if hasattr(log_writer, "write"):
                log_writer.write(log_entry)
            elif callable(log_writer):
                log_writer(log_entry)
