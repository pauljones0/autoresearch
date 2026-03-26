"""
Phase 5 — ImpactTracker: track long-term impact of paper-sourced modifications,
including "stepping stone" value (improvements enabled by the architectural change).
"""

import json
import os
import time


class ImpactTracker:
    """Track long-term impact of accepted paper-sourced modifications.

    After a paper-sourced modification is accepted, compare the improvement
    rate in the subsequent *window* iterations against the baseline rate to
    estimate the modification's stepping-stone value.
    """

    def __init__(self, data_dir: str = "."):
        self.data_dir = data_dir
        self._impacts_path = os.path.join(data_dir, "paper_impacts.jsonl")

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def track(
        self,
        accepted_entry_id: str,
        journal_entries: list,
        window: int = 20,
    ) -> dict:
        """Compute the impact of an accepted paper-sourced modification.

        Args:
            accepted_entry_id: The journal entry ID of the accepted modification.
            journal_entries: Full list of journal entry dicts, ordered by timestamp.
            window: Number of subsequent iterations to examine.

        Returns:
            dict with immediate_delta, post_acceptance_improvement_rate,
            baseline_rate, stepping_stone_value.
        """
        # Find the accepted entry's index
        entry_idx = None
        for i, e in enumerate(journal_entries):
            if e.get("id") == accepted_entry_id:
                entry_idx = i
                break

        if entry_idx is None:
            return {
                "accepted_entry_id": accepted_entry_id,
                "immediate_delta": 0.0,
                "post_acceptance_improvement_rate": 0.0,
                "baseline_rate": 0.0,
                "stepping_stone_value": 0.0,
                "error": "entry_not_found",
            }

        entry = journal_entries[entry_idx]
        immediate_delta = entry.get("actual_delta", 0.0)

        # Post-acceptance window
        post_entries = journal_entries[entry_idx + 1: entry_idx + 1 + window]
        post_rate = self._improvement_rate(post_entries)

        # Baseline: entries before the accepted modification (same window size)
        pre_start = max(0, entry_idx - window)
        pre_entries = journal_entries[pre_start: entry_idx]
        baseline_rate = self._improvement_rate(pre_entries)

        # Stepping stone value: how much the improvement rate increased
        stepping_stone = post_rate - baseline_rate

        impact = {
            "accepted_entry_id": accepted_entry_id,
            "immediate_delta": immediate_delta,
            "post_acceptance_improvement_rate": post_rate,
            "baseline_rate": baseline_rate,
            "stepping_stone_value": stepping_stone,
            "window": window,
            "n_post_entries": len(post_entries),
            "n_pre_entries": len(pre_entries),
            "tracked_at": time.time(),
        }

        # Persist
        self._append_impact(impact)

        return impact

    def get_all_impacts(self) -> list:
        """Load all tracked impact records."""
        entries = []
        try:
            with open(self._impacts_path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            entries.append(json.loads(line))
                        except json.JSONDecodeError:
                            pass
        except FileNotFoundError:
            pass
        return entries

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _improvement_rate(entries: list) -> float:
        """Fraction of entries that are accepted (i.e., improvements).

        Args:
            entries: list of journal entry dicts.

        Returns:
            Float in [0, 1].
        """
        if not entries:
            return 0.0
        accepted = sum(1 for e in entries if e.get("verdict") == "accepted")
        return accepted / len(entries)

    def _append_impact(self, impact: dict) -> None:
        with open(self._impacts_path, "a") as f:
            f.write(json.dumps(impact) + "\n")
