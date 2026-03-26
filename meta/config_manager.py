"""Meta-config manager with atomic save, caching, validation, and diff."""

import json
import os
import tempfile
import time
import copy


class MetaConfigManager:
    """Manages loading, saving, and querying the meta configuration file."""

    def __init__(self, config_path: str = "meta_config.json", schema_path: str = "meta_config_schema.json"):
        self._config_path = config_path
        self._schema_path = schema_path
        self._cache = None
        self._cache_mtime = 0.0

    def load(self) -> dict:
        """Load config from disk, using cache if file hasn't changed."""
        try:
            mtime = os.path.getmtime(self._config_path)
        except OSError:
            self._cache = {}
            self._cache_mtime = 0.0
            return {}

        if self._cache is not None and mtime == self._cache_mtime:
            return copy.deepcopy(self._cache)

        with open(self._config_path, "r") as f:
            self._cache = json.load(f)
        self._cache_mtime = mtime
        return copy.deepcopy(self._cache)

    def save(self, config: dict) -> None:
        """Atomically save config to disk (write to temp, then rename)."""
        dir_name = os.path.dirname(os.path.abspath(self._config_path))
        fd, tmp_path = tempfile.mkstemp(dir=dir_name, suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(config, f, indent=2)
            # On Windows, os.replace handles atomic overwrite
            os.replace(tmp_path, self._config_path)
        except Exception:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise
        # Invalidate cache
        self._cache = copy.deepcopy(config)
        try:
            self._cache_mtime = os.path.getmtime(self._config_path)
        except OSError:
            self._cache_mtime = 0.0

    def get_param(self, param_id: str, default=None):
        """Get a single parameter value."""
        config = self.load()
        return config.get(param_id, default)

    def set_param(self, param_id: str, value) -> None:
        """Set a single parameter value and save."""
        config = self.load()
        config[param_id] = value
        self.save(config)

    def get_system_params(self, system: str, inventories: list) -> dict:
        """Get all parameters for a given system.

        Args:
            system: system name (e.g., 'bandit', 'model_scientist').
            inventories: list of inventorist objects to identify system params.

        Returns:
            dict of param_id -> current value for that system.
        """
        config = self.load()
        system_param_ids = set()
        for inv in inventories:
            for param in inv.inventory():
                if param.system == system:
                    system_param_ids.add(param.param_id)
        return {k: v for k, v in config.items() if k in system_param_ids}

    def diff(self, other_config: dict) -> list:
        """Compute diff between current config and another config.

        Returns:
            list of dicts with keys: param_id, old_value, new_value.
        """
        current = self.load()
        diffs = []
        all_keys = set(current.keys()) | set(other_config.keys())
        for key in sorted(all_keys):
            old = current.get(key)
            new = other_config.get(key)
            if old != new:
                diffs.append({"param_id": key, "old_value": old, "new_value": new})
        return diffs

    def validate(self, config: dict = None) -> list:
        """Validate config against the JSON schema.

        Returns:
            list of validation error strings (empty if valid).
        """
        if config is None:
            config = self.load()

        try:
            with open(self._schema_path, "r") as f:
                schema = json.load(f)
        except (OSError, json.JSONDecodeError):
            return ["Schema file not found or invalid"]

        errors = []
        properties = schema.get("properties", {})
        required = schema.get("required", [])

        # Check required fields
        for req in required:
            if req not in config:
                errors.append(f"Missing required parameter: {req}")

        # Check types and ranges
        for param_id, value in config.items():
            if param_id not in properties:
                if schema.get("additionalProperties") is False:
                    errors.append(f"Unknown parameter: {param_id}")
                continue

            prop = properties[param_id]
            expected_type = prop.get("type")

            # Type check
            type_ok = True
            if expected_type == "number":
                if not isinstance(value, (int, float)):
                    errors.append(f"{param_id}: expected number, got {type(value).__name__}")
                    type_ok = False
            elif expected_type == "integer":
                if not isinstance(value, int) or isinstance(value, bool):
                    errors.append(f"{param_id}: expected integer, got {type(value).__name__}")
                    type_ok = False
            elif expected_type == "boolean":
                if not isinstance(value, bool):
                    errors.append(f"{param_id}: expected boolean, got {type(value).__name__}")
                    type_ok = False
            elif expected_type == "string":
                if not isinstance(value, str):
                    errors.append(f"{param_id}: expected string, got {type(value).__name__}")
                    type_ok = False

            if not type_ok:
                continue

            # Range check
            if "enum" in prop and value not in prop["enum"]:
                errors.append(f"{param_id}: value {value!r} not in enum {prop['enum']}")
            if "minimum" in prop and isinstance(value, (int, float)) and value < prop["minimum"]:
                errors.append(f"{param_id}: value {value} below minimum {prop['minimum']}")
            if "maximum" in prop and isinstance(value, (int, float)) and value > prop["maximum"]:
                errors.append(f"{param_id}: value {value} above maximum {prop['maximum']}")

        return errors
