"""
BottleneckMapper: static + extensible mapping from bottleneck types
to paper search terms. Custom mappings persist to bottleneck_mappings.json.
"""

import json
import os


# Default built-in mappings
_DEFAULT_MAPPINGS = {
    "attention_entropy_collapse": [
        "attention diversity", "head redundancy", "multi-head pruning",
        "attention routing", "head dropout",
    ],
    "gradient_vanishing_early": [
        "residual connections", "gradient flow", "initialization scheme",
        "normalization placement", "skip connections",
    ],
    "gradient_exploding": [
        "gradient clipping", "normalization", "weight initialization",
        "learning rate warmup",
    ],
    "dead_neurons": [
        "activation function", "dying relu", "leaky relu", "gelu",
        "neuron pruning",
    ],
    "loss_imbalance": [
        "token frequency", "loss weighting", "focal loss", "rare token",
        "vocabulary",
    ],
    "layer_redundancy": [
        "layer pruning", "layer sharing", "deep supervision", "auxiliary loss",
    ],
}


class BottleneckMapper:
    """Maps bottleneck types to relevant paper search terms."""

    def __init__(self, custom_mappings_path: str = ""):
        self._mappings = {k: list(v) for k, v in _DEFAULT_MAPPINGS.items()}
        self._custom_path = custom_mappings_path
        if custom_mappings_path:
            self._load_custom(custom_mappings_path)

    def _load_custom(self, path: str):
        """Load custom mappings from JSON, merging with defaults."""
        try:
            with open(path) as f:
                custom = json.load(f)
            if isinstance(custom, dict):
                for btype, terms in custom.items():
                    if isinstance(terms, list):
                        if btype in self._mappings:
                            existing = set(self._mappings[btype])
                            for t in terms:
                                if t not in existing:
                                    self._mappings[btype].append(t)
                        else:
                            self._mappings[btype] = list(terms)
        except (FileNotFoundError, json.JSONDecodeError):
            pass

    def get_search_terms(self, bottleneck_type: str) -> list:
        """Return search terms for a given bottleneck type."""
        return list(self._mappings.get(bottleneck_type, []))

    def add_mapping(self, bottleneck_type: str, terms: list):
        """Add or extend a mapping and persist custom additions."""
        if bottleneck_type not in self._mappings:
            self._mappings[bottleneck_type] = []
        existing = set(self._mappings[bottleneck_type])
        for t in terms:
            if t not in existing:
                self._mappings[bottleneck_type].append(t)
                existing.add(t)
        self._persist_custom()

    def _persist_custom(self):
        """Save non-default mappings to the custom mappings file."""
        if not self._custom_path:
            return
        # Compute diff from defaults
        custom = {}
        for btype, terms in self._mappings.items():
            if btype not in _DEFAULT_MAPPINGS:
                custom[btype] = terms
            else:
                extras = [t for t in terms if t not in _DEFAULT_MAPPINGS[btype]]
                if extras:
                    custom[btype] = extras
        try:
            parent = os.path.dirname(self._custom_path)
            if parent:
                os.makedirs(parent, exist_ok=True)
            with open(self._custom_path, "w") as f:
                json.dump(custom, f, indent=2)
        except OSError:
            pass

    def all_mappings(self) -> dict:
        """Return a copy of all current mappings."""
        return {k: list(v) for k, v in self._mappings.items()}
