"""Bridge for applying meta-config to the GPU kernel subsystem."""

import json

from meta.inventory.gk_params import GPUKernelParameterInventorist


class GPUKernelConfigBridge:
    """Writes GPU kernel meta-config parameters to gk_overrides.json."""

    def __init__(self, overrides_path: str = "gk_overrides.json"):
        self._overrides_path = overrides_path
        self._inventorist = GPUKernelParameterInventorist()

    def apply(self, meta_config: dict) -> dict:
        """Extract GPU kernel params from meta_config and write to overrides file.

        Args:
            meta_config: full meta-configuration dict.

        Returns:
            dict of GPU kernel overrides that were written.
        """
        gk_params = {}
        for param in self._inventorist.inventory():
            if param.param_id in meta_config:
                gk_params[param.param_id] = meta_config[param.param_id]

        with open(self._overrides_path, "w") as f:
            json.dump(gk_params, f, indent=2)

        return gk_params
