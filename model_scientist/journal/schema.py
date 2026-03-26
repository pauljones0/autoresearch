"""
Journal configuration, path management, and validation.
"""

import os
import uuid

# Schema version for forward compatibility
SCHEMA_VERSION = 1

# Default journal path (project root)
JOURNAL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "hypothesis_journal.jsonl",
)

# Required fields and their expected types
_REQUIRED_FIELDS = {
    "id": str,
    "timestamp": (int, float),
    "hypothesis": str,
    "verdict": str,
}

_VALID_VERDICTS = {"accepted", "rejected", "crashed"}


def generate_entry_id() -> str:
    """Generate a UUID-based unique ID for a journal entry."""
    return uuid.uuid4().hex[:12]


def validate_entry(entry: dict) -> tuple:
    """
    Validate a journal entry dict.

    Returns:
        (is_valid, errors): A tuple of a bool and a list of error strings.
    """
    errors = []

    for field, expected in _REQUIRED_FIELDS.items():
        if field not in entry:
            errors.append(f"missing required field: {field}")
        elif not isinstance(entry[field], expected):
            errors.append(f"field '{field}' has wrong type: expected {expected}, got {type(entry[field])}")

    if "verdict" in entry and entry["verdict"] not in _VALID_VERDICTS:
        errors.append(f"invalid verdict '{entry['verdict']}': must be one of {_VALID_VERDICTS}")

    if "id" in entry and isinstance(entry["id"], str) and not entry["id"]:
        errors.append("field 'id' must not be empty")

    if "predicted_delta" in entry and not isinstance(entry["predicted_delta"], (int, float)):
        errors.append(f"field 'predicted_delta' must be numeric, got {type(entry['predicted_delta'])}")

    if "actual_delta" in entry and not isinstance(entry["actual_delta"], (int, float)):
        errors.append(f"field 'actual_delta' must be numeric, got {type(entry['actual_delta'])}")

    return (len(errors) == 0, errors)
