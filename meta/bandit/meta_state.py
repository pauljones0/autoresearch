"""
Atomic state persistence for the meta-bandit.

Same pattern as bandit/persistence.py: atomic writes with SHA-256
checksums and log-based recovery from meta_log.jsonl.
"""

import hashlib
import json
import os
import time

from meta.schemas import MetaBanditState, DimensionState, load_jsonl


class MetaStateManager:
    """Crash-safe persistence for MetaBanditState."""

    def save(self, state: MetaBanditState, state_path: str) -> None:
        """Atomically write state with checksum.

        Writes to a .tmp file, fsyncs, then renames so readers never
        see a partial write.
        """
        data = state.to_dict()
        data["metadata"]["last_updated"] = time.time()

        data.pop("_checksum", None)
        canonical = json.dumps(data, sort_keys=True, separators=(",", ":"))
        data["_checksum"] = hashlib.sha256(canonical.encode()).hexdigest()

        tmp_path = state_path + ".tmp"
        with open(tmp_path, "w") as f:
            json.dump(data, f, indent=2)
            f.flush()
            os.fsync(f.fileno())

        os.replace(tmp_path, state_path)

    def load(self, state_path: str) -> MetaBanditState:
        """Load and verify state, raising ValueError on checksum mismatch."""
        with open(state_path) as f:
            data = json.load(f)

        stored_checksum = data.pop("_checksum", None)
        canonical = json.dumps(data, sort_keys=True, separators=(",", ":"))
        computed = hashlib.sha256(canonical.encode()).hexdigest()

        if stored_checksum is not None and stored_checksum != computed:
            raise ValueError(
                f"Checksum mismatch: stored={stored_checksum}, "
                f"computed={computed}")

        state = MetaBanditState.from_dict(data)
        issues = state.validate()
        if issues:
            raise ValueError(
                f"State validation failed: {'; '.join(issues)}")
        return state

    def detect_corruption(self, state_path: str) -> bool:
        """Return True if the state file is corrupted or invalid."""
        try:
            self.load(state_path)
            return False
        except (ValueError, json.JSONDecodeError, FileNotFoundError,
                KeyError, TypeError):
            return True

    def recover(self, state_path: str, log_path: str) -> MetaBanditState:
        """Reconstruct state by replaying meta_log.jsonl entries.

        Looks for meta_experiment_completed and meta_posterior_update
        entries to rebuild dimension posteriors.  The recovered state
        is saved atomically to state_path.
        """
        entries = load_jsonl(log_path)
        if not entries:
            raise ValueError(f"Cannot recover: log at {log_path} is empty")

        state = MetaBanditState()

        for entry in entries:
            entry_type = entry.get("type", "")

            if entry_type == "meta_regime_change":
                state.meta_regime = entry.get("new_regime", state.meta_regime)

            elif entry_type == "meta_posterior_update":
                param_id = entry.get("param_id", "")
                if param_id and param_id in state.dimensions:
                    dim = state.dimensions[param_id]
                    var_key = entry.get("variant_key", "")
                    if var_key:
                        dim.variant_posteriors[var_key] = {
                            "alpha": entry.get("alpha", 1.0),
                            "beta": entry.get("beta", 1.0),
                        }
                elif param_id:
                    dim = DimensionState(param_id=param_id)
                    var_key = entry.get("variant_key", "")
                    if var_key:
                        dim.variant_posteriors[var_key] = {
                            "alpha": entry.get("alpha", 1.0),
                            "beta": entry.get("beta", 1.0),
                        }
                    state.dimensions[param_id] = dim

            elif entry_type == "meta_experiment_completed":
                state.total_meta_experiments += 1
                state.global_meta_iteration = max(
                    state.global_meta_iteration,
                    entry.get("meta_iteration", state.global_meta_iteration))

            elif entry_type == "meta_promotion":
                param_id = entry.get("param_id", "")
                if param_id in state.dimensions:
                    state.dimensions[param_id].current_best = entry.get("promoted_value")
                    state.dimensions[param_id].last_promoted = entry.get(
                        "timestamp", time.time())
                config_update = entry.get("config_update", {})
                state.best_config.update(config_update)

            elif entry_type == "meta_budget_status":
                state.budget_used = entry.get("budget_used", state.budget_used)

        self.save(state, state_path)
        return state
