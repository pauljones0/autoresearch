"""
Meta-loop sandbox enforcer — whitelist-based file write guard.
"""

import os
from meta.schemas import BoundaryViolationError

_WRITABLE_PATTERNS = [
    "meta_config.json", "bandit_overrides.json", "ms_overrides.json",
    "st_overrides.json", "gk_overrides.json", "meta_log.jsonl",
    "meta_state.json", "meta_config_schema.json", "meta_config_report.json",
    "meta_knowledge_base.json", "meta_impact_report.json",
]

_WRITABLE_DIRS = ["meta/", "bandit/prompt_templates/"]

_BLOCKED_SUBSTRINGS = ["train.py", "eval", "valid", "dataset"]


class MetaSandboxEnforcer:
    """Whitelist-based file write guard for the meta-loop."""

    def __init__(self, work_dir: str = "."):
        self.work_dir = os.path.abspath(work_dir)

    def check_write(self, path: str) -> bool:
        abs_path = os.path.abspath(path)
        rel_path = os.path.relpath(abs_path, self.work_dir)
        norm = rel_path.replace("\\", "/")

        for blocked in _BLOCKED_SUBSTRINGS:
            if blocked in norm.lower():
                raise BoundaryViolationError(
                    "unauthorized_write",
                    f"Path contains blocked substring '{blocked}': {path}",
                )

        basename = os.path.basename(norm)
        if basename in _WRITABLE_PATTERNS:
            return True

        for wdir in _WRITABLE_DIRS:
            if norm.startswith(wdir):
                return True

        raise BoundaryViolationError(
            "unauthorized_write", f"Path not in meta-loop whitelist: {path}",
        )

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False
