"""Bridge for applying meta-config to the bandit subsystem."""

import json
import os

from meta.inventory.bandit_params import BanditParameterInventorist


class BanditConfigBridge:
    """Writes bandit-relevant meta-config parameters to bandit_overrides.json."""

    def __init__(self, overrides_path: str = "bandit_overrides.json"):
        self._overrides_path = overrides_path
        self._inventorist = BanditParameterInventorist()

    def apply(self, meta_config: dict) -> dict:
        """Extract bandit params from meta_config and write to overrides file.

        Args:
            meta_config: full meta-configuration dict.

        Returns:
            dict of bandit overrides that were written.
        """
        bandit_params = {}
        for param in self._inventorist.inventory():
            if param.param_id in meta_config:
                bandit_params[param.param_id] = meta_config[param.param_id]

        with open(self._overrides_path, "w") as f:
            json.dump(bandit_params, f, indent=2)

        return bandit_params
