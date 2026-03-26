"""
JournalWriter — append-only, thread-safe writing of experiment entries
to the hypothesis journal (JSONL format).
"""

import json
import os
import re
import sys
import time
import threading

from ..schemas import JournalEntry
from .schema import JOURNAL_PATH, SCHEMA_VERSION, generate_entry_id, validate_entry


class JournalWriter:
    """Thread-safe, append-only writer for the hypothesis journal."""

    def __init__(self, path: str = None):
        self.path = path or JOURNAL_PATH
        self._lock = threading.Lock()

    def _write_entry(self, entry: JournalEntry):
        """Write a single entry to the journal file with file locking."""
        line = entry.to_json_line() + "\n"
        with self._lock:
            if sys.platform == "win32":
                self._write_entry_win32(line)
            else:
                self._write_entry_unix(line)

    def _write_entry_win32(self, line: str):
        """Windows file locking using msvcrt on a lockfile."""
        import msvcrt
        lockpath = self.path + ".lock"
        # Use a separate lock file to avoid byte-range issues with append
        lf = open(lockpath, "w")
        try:
            msvcrt.locking(lf.fileno(), msvcrt.LK_LOCK, 1)
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(line)
                f.flush()
            msvcrt.locking(lf.fileno(), msvcrt.LK_UNLCK, 1)
        finally:
            lf.close()

    def _write_entry_unix(self, line: str):
        """Unix file locking using fcntl."""
        import fcntl
        with open(self.path, "a", encoding="utf-8") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                f.write(line)
                f.flush()
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)

    def log_experiment(
        self,
        hypothesis: str,
        predicted_delta: float,
        actual_delta: float,
        modification_diff: str,
        verdict: str,
        diagnostics_summary: dict = None,
        tags: list = None,
        **extra_fields,
    ) -> JournalEntry:
        """
        Log a single experiment to the journal.

        Args:
            hypothesis: What was being tested.
            predicted_delta: Predicted change in val_bpb (negative = improvement).
            actual_delta: Actual change in val_bpb.
            modification_diff: Git diff or description of the code change.
            verdict: One of "accepted", "rejected", "crashed".
            diagnostics_summary: Optional diagnostics snapshot dict.
            tags: Optional list of string tags.
            **extra_fields: Any additional fields to store on the entry.

        Returns:
            The JournalEntry that was written.
        """
        entry = JournalEntry(
            id=generate_entry_id(),
            timestamp=time.time(),
            hypothesis=hypothesis,
            predicted_delta=predicted_delta,
            actual_delta=actual_delta,
            modification_diff=modification_diff,
            verdict=verdict,
            diagnostics_summary=diagnostics_summary or {},
            tags=tags or [],
        )
        # Apply any extra fields
        for k, v in extra_fields.items():
            if hasattr(entry, k):
                setattr(entry, k, v)

        # Validate before writing
        is_valid, errors = validate_entry(entry.to_dict())
        if not is_valid:
            raise ValueError(f"Invalid journal entry: {errors}")

        self._write_entry(entry)
        return entry

    def log_from_run_output(
        self,
        run_log: str,
        hypothesis: str,
        modification_diff: str,
        previous_val_bpb: float,
        predicted_delta: float = 0.0,
        tags: list = None,
    ) -> JournalEntry:
        """
        Parse training run output to extract metrics and log an experiment.

        Args:
            run_log: The full stdout/stderr text from `uv run train.py`.
            hypothesis: What was being tested.
            modification_diff: Git diff of the change.
            previous_val_bpb: The val_bpb before this experiment.
            predicted_delta: Predicted change in val_bpb.
            tags: Optional list of tags.

        Returns:
            The JournalEntry that was written.
        """
        val_bpb = _parse_metric(run_log, "val_bpb")
        peak_vram = _parse_metric(run_log, "peak_vram_mb")
        training_seconds = _parse_metric(run_log, "training_seconds")
        num_steps = _parse_metric(run_log, "num_steps")
        mfu = _parse_metric(run_log, "mfu_percent")

        # Determine verdict
        if val_bpb is None:
            verdict = "crashed"
            actual_delta = 0.0
        else:
            actual_delta = val_bpb - previous_val_bpb
            verdict = "accepted" if actual_delta < 0 else "rejected"

        diagnostics_summary = {}
        if val_bpb is not None:
            diagnostics_summary["val_bpb"] = val_bpb
        if peak_vram is not None:
            diagnostics_summary["peak_vram_mb"] = peak_vram
        if training_seconds is not None:
            diagnostics_summary["training_seconds"] = training_seconds
        if num_steps is not None:
            diagnostics_summary["num_steps"] = int(num_steps)
        if mfu is not None:
            diagnostics_summary["mfu_percent"] = mfu

        return self.log_experiment(
            hypothesis=hypothesis,
            predicted_delta=predicted_delta,
            actual_delta=actual_delta,
            modification_diff=modification_diff,
            verdict=verdict,
            diagnostics_summary=diagnostics_summary,
            tags=tags,
        )


def _parse_metric(log_text: str, metric_name: str):
    """Extract a numeric metric from run log output. Returns float or None."""
    pattern = rf"^{re.escape(metric_name)}:\s+([\d.eE+\-]+)"
    match = re.search(pattern, log_text, re.MULTILINE)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            return None
    return None


# ---------------------------------------------------------------------------
# Convenience function for the autoresearch loop
# ---------------------------------------------------------------------------

_default_writer = None
_default_writer_lock = threading.Lock()


def record_experiment(
    hypothesis: str,
    predicted_delta: float,
    actual_delta: float,
    modification_diff: str,
    verdict: str,
    diagnostics_summary: dict = None,
    tags: list = None,
    journal_path: str = None,
    **extra_fields,
) -> JournalEntry:
    """
    Simple function the autoresearch loop can call after each experiment.

    Uses a module-level singleton writer for convenience.
    """
    global _default_writer
    with _default_writer_lock:
        if _default_writer is None or (journal_path and _default_writer.path != journal_path):
            _default_writer = JournalWriter(path=journal_path)
    return _default_writer.log_experiment(
        hypothesis=hypothesis,
        predicted_delta=predicted_delta,
        actual_delta=actual_delta,
        modification_diff=modification_diff,
        verdict=verdict,
        diagnostics_summary=diagnostics_summary,
        tags=tags,
        **extra_fields,
    )
