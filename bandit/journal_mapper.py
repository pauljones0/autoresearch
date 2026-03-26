"""
Journal-to-arm mapper for the Adaptive Bandit pipeline.

Maps journal entries to canonical arm IDs using bandit_arm tags,
modification_category, or source:kernel heuristics.
"""

import json


# Canonical arm IDs that the mapper recognises.
_KNOWN_ARM_IDS = {
    "architecture", "optimizer", "hyperparameter", "activation",
    "initialization", "regularization", "scheduling",
    "kernel_discovery", "kernel_evolution",
}


class JournalArmMapper:
    """Maps journal entries to bandit arm IDs."""

    def map_entry_to_arm(self, entry: dict) -> str:
        """Map a single journal entry dict to an arm_id.

        Resolution order:
        1. Explicit ``bandit_arm`` tag.
        2. ``modification_category`` field.
        3. ``source`` == "kernel" with optional subtype.
        4. Falls back to ``"unknown"``.
        """
        # 1. Explicit bandit_arm tag
        bandit_arm = entry.get("bandit_arm", "")
        if bandit_arm in _KNOWN_ARM_IDS:
            return bandit_arm

        # 2. modification_category
        mod_cat = entry.get("modification_category", "")
        if mod_cat in _KNOWN_ARM_IDS:
            return mod_cat

        # 3. Kernel source heuristic
        source = entry.get("source", "")
        if source == "kernel":
            subtype = entry.get("subtype", "")
            if subtype == "evolution":
                return "kernel_evolution"
            return "kernel_discovery"

        return "unknown"

    def map_all(self, journal_path: str) -> dict[str, list]:
        """Group all journal entries by arm_id.

        Returns a dict with:
        - One key per arm_id (including ``"unknown"``).
        - Each value is a list of entry dicts.
        - A special ``"_meta"`` key with mapping statistics:
            ``total``, ``mapped``, ``unmapped``, ``mapping_rate``.
        """
        try:
            with open(journal_path) as f:
                entries = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {"_meta": {"total": 0, "mapped": 0, "unmapped": 0,
                              "mapping_rate": 0.0}}

        if not isinstance(entries, list):
            entries = []

        groups: dict[str, list] = {}
        mapped = 0
        unmapped = 0

        for entry in entries:
            if not isinstance(entry, dict):
                continue
            arm_id = self.map_entry_to_arm(entry)
            groups.setdefault(arm_id, []).append(entry)
            if arm_id == "unknown":
                unmapped += 1
            else:
                mapped += 1

        total = mapped + unmapped
        groups["_meta"] = {
            "total": total,
            "mapped": mapped,
            "unmapped": unmapped,
            "mapping_rate": mapped / total if total > 0 else 0.0,
        }
        return groups
