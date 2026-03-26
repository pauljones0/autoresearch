"""
Hot configuration reloader: applies runtime overrides to bandit state.
"""

import json
import os
import copy

from bandit.schemas import BanditState


# Parameters that can be overridden at runtime
_OVERRIDABLE = {
    "T_base": float,
    "K_reheat_threshold": int,
    "reheat_factor": float,
    "min_temperature": float,
    "exploration_floor": float,
    "paper_preference_ratio": float,
    "enable_rollback_safety": bool,
}


class HotConfigReloader:
    """Loads and applies runtime configuration overrides to bandit state."""

    def check_and_reload(
        self,
        state: BanditState,
        config_override_path: str = "bandit_overrides.json",
    ) -> tuple:
        """Check for override file and apply changes.

        Args:
            state: Current bandit state.
            config_override_path: Path to JSON override file.

        Returns:
            (updated_state, list_of_change_descriptions)
        """
        changes = []

        if not os.path.exists(config_override_path):
            return state, changes

        try:
            with open(config_override_path) as f:
                overrides = json.load(f)
        except (json.JSONDecodeError, OSError):
            changes.append(f"Failed to parse override file: {config_override_path}")
            return state, changes

        if not isinstance(overrides, dict):
            changes.append("Override file does not contain a JSON object")
            return state, changes

        # Validate and apply
        new_state = copy.deepcopy(state)

        for key, value in overrides.items():
            if key not in _OVERRIDABLE:
                changes.append(f"Ignored unknown override: {key}")
                continue

            expected_type = _OVERRIDABLE[key]
            if not isinstance(value, expected_type):
                # Try type coercion for int/float compatibility
                try:
                    value = expected_type(value)
                except (ValueError, TypeError):
                    changes.append(
                        f"Ignored {key}: expected {expected_type.__name__}, "
                        f"got {type(value).__name__}")
                    continue

            # Validate bounds
            valid, reason = self._validate_value(key, value)
            if not valid:
                changes.append(f"Rejected {key}={value}: {reason}")
                continue

            old_value = getattr(new_state, key)
            if old_value != value:
                setattr(new_state, key, value)
                changes.append(f"Updated {key}: {old_value} -> {value}")

        return new_state, changes

    def _validate_value(self, key: str, value) -> tuple:
        """Validate a parameter value. Returns (valid, reason)."""
        if key == "T_base":
            if value <= 0:
                return False, "must be > 0"
        elif key == "K_reheat_threshold":
            if value < 1:
                return False, "must be >= 1"
        elif key == "reheat_factor":
            if value < 1:
                return False, "must be >= 1"
        elif key == "min_temperature":
            if value < 0:
                return False, "must be >= 0"
        elif key == "exploration_floor":
            if not (0 <= value <= 1):
                return False, "must be in [0, 1]"
        elif key == "paper_preference_ratio":
            if not (0 <= value <= 1):
                return False, "must be in [0, 1]"
        # enable_rollback_safety: bool, no bounds needed
        return True, ""
