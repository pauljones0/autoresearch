"""Bridge for applying meta-config to the model scientist subsystem."""

import json

from meta.inventory.ms_params import ModelScientistParameterInventorist


class ModelScientistConfigBridge:
    """Writes model scientist meta-config parameters to ms_overrides.json."""

    def __init__(self, overrides_path: str = "ms_overrides.json"):
        self._overrides_path = overrides_path
        self._inventorist = ModelScientistParameterInventorist()

    def apply(self, meta_config: dict) -> dict:
        """Extract model scientist params from meta_config and write to overrides file.

        Args:
            meta_config: full meta-configuration dict.

        Returns:
            dict of model scientist overrides that were written.
        """
        ms_params = {}
        for param in self._inventorist.inventory():
            if param.param_id in meta_config:
                ms_params[param.param_id] = meta_config[param.param_id]

        with open(self._overrides_path, "w") as f:
            json.dump(ms_params, f, indent=2)

        return ms_params
