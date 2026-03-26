"""
Append-only structured log writer for the Adaptive Bandit pipeline.

Writes single JSON lines to ``bandit_log.jsonl`` with platform-aware
file locking (``msvcrt`` on Windows, ``fcntl`` on Unix).
"""

import json
import os
import time


# ---------------------------------------------------------------------------
# Platform-aware file locking helpers
# ---------------------------------------------------------------------------

def _lock_file(f):
    """Acquire an exclusive lock on open file *f*."""
    if os.name == "nt":
        import msvcrt
        msvcrt.locking(f.fileno(), msvcrt.LK_LOCK, 1)
    else:
        import fcntl
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)


def _unlock_file(f):
    """Release the lock on open file *f*."""
    if os.name == "nt":
        import msvcrt
        # Seek back to unlock the same byte range
        pos = f.tell()
        f.seek(0)
        try:
            msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
        except OSError:
            pass
        f.seek(pos)
    else:
        import fcntl
        fcntl.flock(f.fileno(), fcntl.LOCK_UN)


# ---------------------------------------------------------------------------
# Valid entry types
# ---------------------------------------------------------------------------

_VALID_ENTRY_TYPES = {
    "warm_start",
    "selection",
    "posterior_update",
    "acceptance_decision",
    "reheat",
    "regime_change",
    "config_change",
    "delayed_correction_queued",
    "regime_change_detected",
    "regime_change_recommendation",
}


# ---------------------------------------------------------------------------
# BanditLogWriter
# ---------------------------------------------------------------------------

class BanditLogWriter:
    """Append-only JSONL writer with file locking."""

    def __init__(self, log_path: str):
        self.log_path = log_path

    # ------------------------------------------------------------------
    # Core append
    # ------------------------------------------------------------------

    def _append(self, entry_type: str, data: dict) -> None:
        """Append a single JSON line with ``type`` and ``timestamp``."""
        record = {"type": entry_type, "timestamp": time.time()}
        record.update(data)
        line = json.dumps(record, separators=(",", ":")) + "\n"

        with open(self.log_path, "a") as f:
            _lock_file(f)
            try:
                f.write(line)
                f.flush()
                os.fsync(f.fileno())
            finally:
                _unlock_file(f)

    # ------------------------------------------------------------------
    # Typed helpers
    # ------------------------------------------------------------------

    def log_warm_start(self, state_dict: dict) -> None:
        """Log a warm-start snapshot (arms, regime, global_iteration)."""
        self._append("warm_start", state_dict)

    def log_selection(self, arm_id: str, selected_by: str = "thompson",
                      sample_values: dict | None = None,
                      dispatch_path: str = "internal",
                      **extra) -> None:
        """Log an arm selection event."""
        data = {
            "arm_id": arm_id,
            "selected_by": selected_by,
            "sample_values": sample_values or {},
            "dispatch_path": dispatch_path,
        }
        data.update(extra)
        self._append("selection", data)

    def log_posterior_update(self, arm_id: str, alpha: float, beta: float,
                             temperature: float, total_attempts: int,
                             total_successes: int,
                             consecutive_failures: int,
                             global_iteration: int,
                             **extra) -> None:
        """Log a posterior parameter update."""
        data = {
            "arm_id": arm_id,
            "alpha": alpha,
            "beta": beta,
            "temperature": temperature,
            "total_attempts": total_attempts,
            "total_successes": total_successes,
            "consecutive_failures": consecutive_failures,
            "global_iteration": global_iteration,
        }
        data.update(extra)
        self._append("posterior_update", data)

    def log_acceptance(self, arm_id: str, accepted: bool,
                       accepted_by: str, probability: float,
                       delta: float, T_effective: float,
                       **extra) -> None:
        """Log an annealing acceptance decision."""
        data = {
            "arm_id": arm_id,
            "accepted": accepted,
            "accepted_by": accepted_by,
            "probability": probability,
            "delta": delta,
            "T_effective": T_effective,
        }
        data.update(extra)
        self._append("acceptance_decision", data)

    def log_reheat(self, arm_id: str, temperature_before: float,
                   temperature_after: float,
                   consecutive_failures_at_trigger: int,
                   reheat_count: int, **extra) -> None:
        """Log a temperature reheat event."""
        data = {
            "arm_id": arm_id,
            "temperature_before": temperature_before,
            "temperature_after": temperature_after,
            "consecutive_failures_at_trigger": consecutive_failures_at_trigger,
            "reheat_count": reheat_count,
        }
        data.update(extra)
        self._append("reheat", data)

    def log_regime_change(self, old_regime: str, new_regime: str,
                          trigger: str = "", **extra) -> None:
        """Log a regime transition."""
        data = {
            "old_regime": old_regime,
            "new_regime": new_regime,
            "trigger": trigger,
        }
        data.update(extra)
        self._append("regime_change", data)

    def log_config_change(self, parameter: str, old_value, new_value,
                          reason: str = "", **extra) -> None:
        """Log a configuration parameter change."""
        data = {
            "parameter": parameter,
            "old_value": old_value,
            "new_value": new_value,
            "reason": reason,
        }
        data.update(extra)
        self._append("config_change", data)
