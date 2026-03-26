"""
Atomic state persistence for the Adaptive Bandit pipeline.

Provides crash-safe save/load with SHA-256 checksums and
log-based recovery.
"""

import hashlib
import json
import os
import time

from bandit.schemas import BanditState, ArmState, load_jsonl
from bandit.state import validate_state


class AtomicStateManager:
    """Crash-safe JSON persistence with checksum verification."""

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    def save(self, state: BanditState, state_path: str) -> None:
        """Atomically write *state* to *state_path*.

        Writes to a temporary file, fsyncs, then renames over the target
        so readers never see a partial write.
        """
        data = state.to_dict()
        data["metadata"]["last_updated"] = time.time()

        # Compute checksum over canonical JSON (without _checksum key)
        data.pop("_checksum", None)
        canonical = json.dumps(data, sort_keys=True, separators=(",", ":"))
        data["_checksum"] = hashlib.sha256(canonical.encode()).hexdigest()

        tmp_path = state_path + ".tmp"
        with open(tmp_path, "w") as f:
            json.dump(data, f, indent=2)
            f.flush()
            os.fsync(f.fileno())

        os.replace(tmp_path, state_path)

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------

    def load(self, state_path: str) -> BanditState:
        """Load and verify state from *state_path*.

        Raises ``ValueError`` on checksum mismatch or validation failure.
        """
        with open(state_path) as f:
            data = json.load(f)

        stored_checksum = data.pop("_checksum", None)
        canonical = json.dumps(data, sort_keys=True, separators=(",", ":"))
        computed = hashlib.sha256(canonical.encode()).hexdigest()

        if stored_checksum is not None and stored_checksum != computed:
            raise ValueError(
                f"Checksum mismatch: stored={stored_checksum}, "
                f"computed={computed}")

        state = BanditState.from_dict(data)
        issues = validate_state(state)
        if issues:
            raise ValueError(
                f"State validation failed: {'; '.join(issues)}")
        return state

    # ------------------------------------------------------------------
    # Corruption detection
    # ------------------------------------------------------------------

    def detect_corruption(self, state_path: str) -> bool:
        """Return True if the state file is corrupted or invalid."""
        try:
            self.load(state_path)
            return False
        except (ValueError, json.JSONDecodeError, FileNotFoundError,
                KeyError, TypeError):
            return True

    # ------------------------------------------------------------------
    # Recovery from log
    # ------------------------------------------------------------------

    def recover(self, state_path: str, log_path: str) -> BanditState:
        """Reconstruct state by replaying bandit_log.jsonl entries.

        Looks for ``warm_start`` entries to establish the initial state,
        then applies ``posterior_update`` entries on top.  The recovered
        state is saved atomically to *state_path*.
        """
        entries = load_jsonl(log_path)
        if not entries:
            raise ValueError(f"Cannot recover: log at {log_path} is empty")

        # Start from the last warm_start entry
        state = BanditState()
        for entry in entries:
            if entry.get("type") == "warm_start":
                arms_raw = entry.get("arms", {})
                state.arms = {
                    k: ArmState.from_dict(v) if isinstance(v, dict) else v
                    for k, v in arms_raw.items()
                }
                state.regime = entry.get("regime", "no_bandit")
                state.global_iteration = entry.get("global_iteration", 0)
                meta = entry.get("metadata")
                if isinstance(meta, dict):
                    state.metadata.update(meta)

        # Replay posterior updates
        for entry in entries:
            if entry.get("type") != "posterior_update":
                continue
            arm_id = entry.get("arm_id", "")
            if arm_id not in state.arms:
                state.arms[arm_id] = ArmState()
            arm = state.arms[arm_id]
            if not isinstance(arm, ArmState):
                arm = ArmState.from_dict(arm) if isinstance(arm, dict) else ArmState()
                state.arms[arm_id] = arm

            arm.alpha = entry.get("alpha", arm.alpha)
            arm.beta = entry.get("beta", arm.beta)
            arm.temperature = entry.get("temperature", arm.temperature)
            arm.total_attempts = entry.get("total_attempts", arm.total_attempts)
            arm.total_successes = entry.get("total_successes", arm.total_successes)
            arm.consecutive_failures = entry.get(
                "consecutive_failures", arm.consecutive_failures)
            state.global_iteration = max(
                state.global_iteration,
                entry.get("global_iteration", state.global_iteration))

        self.save(state, state_path)
        return state
