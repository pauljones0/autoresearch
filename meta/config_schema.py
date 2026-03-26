"""Meta-config JSON Schema builder and default config generator."""

import json
import os

from meta.schemas import MetaParameter


class MetaConfigSchemaBuilder:
    """Aggregates parameter inventories into a JSON Schema and generates default configs."""

    def build(self, params: list) -> dict:
        """Build a JSON Schema from a flat list of MetaParameter objects.

        Args:
            params: list of MetaParameter objects.

        Returns:
            JSON Schema dict describing all meta-parameters.
        """
        properties = {}
        required = []

        for param in params:
            prop = self._param_to_schema_property(param)
            properties[param.param_id] = prop
            required.append(param.param_id)

        schema = {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "title": "Meta-Autoresearch Configuration",
            "description": "Configuration schema for all meta-tunable harness parameters.",
            "type": "object",
            "properties": properties,
            "required": required,
            "additionalProperties": False,
        }
        return schema

    def generate_default_config(self, params: list) -> dict:
        """Generate a default configuration from a flat list of MetaParameter objects.

        Args:
            params: list of MetaParameter objects.

        Returns:
            dict mapping param_id -> default_value.
        """
        config = {}
        for param in params:
            config[param.param_id] = param.default_value
        return config

    def save_schema(self, schema: dict, path: str = "meta_config_schema.json") -> None:
        """Save schema to a JSON file."""
        with open(path, "w") as f:
            json.dump(schema, f, indent=2)

    def _param_to_schema_property(self, param: MetaParameter) -> dict:
        """Convert a MetaParameter to a JSON Schema property definition."""
        prop = {
            "description": param.impact_hypothesis,
        }

        type_map = {"float": "number", "int": "integer", "bool": "boolean", "str": "string"}
        prop["type"] = type_map.get(param.type, "string")

        if "enum" in param.valid_range:
            prop["enum"] = param.valid_range["enum"]
        else:
            if "min" in param.valid_range:
                prop["minimum"] = param.valid_range["min"]
            if "max" in param.valid_range:
                prop["maximum"] = param.valid_range["max"]

        prop["default"] = param.default_value

        prop["x-meta"] = {
            "system": param.system,
            "category": param.category,
            "sensitivity_estimate": param.sensitivity_estimate,
            "code_path": param.code_path,
            "display_name": param.display_name,
        }

        return prop
