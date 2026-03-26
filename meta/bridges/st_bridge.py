"""Bridge for applying meta-config to the surrogate triage subsystem."""

import json

from meta.inventory.st_params import SurrogateTriageParameterInventorist


class SurrogateTriageConfigBridge:
    """Writes surrogate triage meta-config parameters to st_overrides.json."""

    def __init__(self, overrides_path: str = "st_overrides.json"):
        self._overrides_path = overrides_path
        self._inventorist = SurrogateTriageParameterInventorist()

    def apply(self, meta_config: dict) -> dict:
        """Extract surrogate triage params from meta_config and write to overrides file.

        Args:
            meta_config: full meta-configuration dict.

        Returns:
            dict of surrogate triage overrides that were written.
        """
        st_params = {}
        for param in self._inventorist.inventory():
            if param.param_id in meta_config:
                st_params[param.param_id] = meta_config[param.param_id]

        with open(self._overrides_path, "w") as f:
            json.dump(st_params, f, indent=2)

        return st_params
