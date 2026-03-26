"""
JournalReader — load, query, and summarize the hypothesis journal.

CLI usage:
    python -m model_scientist.journal.reader [--verdict X] [--tags X] [--recent N] [--stats]
"""

import argparse
import json
import os
import statistics
import sys

from .schema import JOURNAL_PATH


class JournalReader:
    """Loads and queries hypothesis_journal.jsonl."""

    def __init__(self, path: str = None):
        self.path = path or JOURNAL_PATH
        self._entries = []
        self.reload()

    def reload(self):
        """(Re)load entries from disk, skipping corrupted lines."""
        self._entries = []
        if not os.path.exists(self.path):
            return
        with open(self.path, "r", encoding="utf-8") as f:
            for lineno, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    self._entries.append(json.loads(line))
                except json.JSONDecodeError:
                    print(f"WARNING: skipping corrupted line {lineno} in {self.path}", file=sys.stderr)

    @property
    def entries(self) -> list:
        return list(self._entries)

    def __len__(self):
        return len(self._entries)

    # ------------------------------------------------------------------
    # Filters
    # ------------------------------------------------------------------

    def by_verdict(self, verdict: str) -> list:
        """Return entries matching the given verdict."""
        return [e for e in self._entries if e.get("verdict") == verdict]

    def by_tags(self, tags: list) -> list:
        """Return entries matching any of the given tags."""
        tag_set = set(tags)
        return [e for e in self._entries if tag_set & set(e.get("tags", []))]

    def by_date_range(self, start: float, end: float) -> list:
        """Return entries with timestamp in [start, end]."""
        return [e for e in self._entries if start <= e.get("timestamp", 0) <= end]

    def by_prediction_accuracy(self, max_error: float) -> list:
        """Return entries where |predicted_delta - actual_delta| < max_error."""
        results = []
        for e in self._entries:
            predicted = e.get("predicted_delta", 0.0)
            actual = e.get("actual_delta", 0.0)
            if abs(predicted - actual) < max_error:
                results.append(e)
        return results

    def search(self, query: str) -> list:
        """Full-text search in hypothesis and modification_diff fields."""
        q = query.lower()
        results = []
        for e in self._entries:
            text = (e.get("hypothesis", "") + " " + e.get("modification_diff", "")).lower()
            if q in text:
                results.append(e)
        return results

    # ------------------------------------------------------------------
    # Aggregations
    # ------------------------------------------------------------------

    def success_rate(self) -> float:
        """Fraction of experiments with verdict 'accepted'."""
        if not self._entries:
            return 0.0
        accepted = sum(1 for e in self._entries if e.get("verdict") == "accepted")
        return accepted / len(self._entries)

    def prediction_correlation(self) -> float:
        """Pearson correlation between predicted_delta and actual_delta.

        Returns 0.0 if fewer than 3 non-crashed entries exist.
        """
        pairs = [
            (e.get("predicted_delta", 0.0), e.get("actual_delta", 0.0))
            for e in self._entries
            if e.get("verdict") != "crashed"
        ]
        if len(pairs) < 3:
            return 0.0
        predicted = [p for p, _ in pairs]
        actual = [a for _, a in pairs]
        try:
            return statistics.correlation(predicted, actual)
        except (statistics.StatisticsError, ZeroDivisionError):
            return 0.0

    def recent(self, n: int = 10) -> list:
        """Return the last n entries (by position in file)."""
        return self._entries[-n:]

    def summary_stats(self) -> dict:
        """Return a dict with counts, rates, and average deltas."""
        total = len(self._entries)
        if total == 0:
            return {
                "total": 0,
                "accepted": 0,
                "rejected": 0,
                "crashed": 0,
                "success_rate": 0.0,
                "avg_predicted_delta": 0.0,
                "avg_actual_delta": 0.0,
            }
        accepted = sum(1 for e in self._entries if e.get("verdict") == "accepted")
        rejected = sum(1 for e in self._entries if e.get("verdict") == "rejected")
        crashed = sum(1 for e in self._entries if e.get("verdict") == "crashed")

        non_crashed = [e for e in self._entries if e.get("verdict") != "crashed"]
        avg_predicted = (
            statistics.mean(e.get("predicted_delta", 0.0) for e in non_crashed) if non_crashed else 0.0
        )
        avg_actual = (
            statistics.mean(e.get("actual_delta", 0.0) for e in non_crashed) if non_crashed else 0.0
        )

        return {
            "total": total,
            "accepted": accepted,
            "rejected": rejected,
            "crashed": crashed,
            "success_rate": accepted / total,
            "avg_predicted_delta": avg_predicted,
            "avg_actual_delta": avg_actual,
        }

    # ------------------------------------------------------------------
    # Context generation for LLM prompts
    # ------------------------------------------------------------------

    def generate_context(self, max_tokens: int = 2000) -> str:
        """
        Generate a formatted summary of recent experiments suitable for
        injecting into an LLM prompt. Estimates ~4 chars per token.
        """
        max_chars = max_tokens * 4
        parts = []

        # Summary stats header
        stats = self.summary_stats()
        header = (
            f"## Experiment Journal Summary\n"
            f"Total experiments: {stats['total']} | "
            f"Accepted: {stats['accepted']} | "
            f"Rejected: {stats['rejected']} | "
            f"Crashed: {stats['crashed']} | "
            f"Success rate: {stats['success_rate']:.1%}\n"
        )
        parts.append(header)

        # Recent experiments (most recent first)
        parts.append("\n## Recent Experiments\n")
        recent = list(reversed(self.recent(20)))
        for e in recent:
            verdict = e.get("verdict", "?")
            hyp = e.get("hypothesis", "")[:120]
            predicted = e.get("predicted_delta", 0.0)
            actual = e.get("actual_delta", 0.0)
            tags = ", ".join(e.get("tags", []))
            line = f"- [{verdict}] {hyp}"
            if verdict != "crashed":
                line += f" (predicted: {predicted:+.4f}, actual: {actual:+.4f})"
            if tags:
                line += f" [{tags}]"
            line += "\n"
            parts.append(line)

            # Check approximate length
            if sum(len(p) for p in parts) > max_chars:
                break

        result = "".join(parts)
        if len(result) > max_chars:
            result = result[:max_chars] + "\n... (truncated)"
        return result


# ---------------------------------------------------------------------------
# CLI interface
# ---------------------------------------------------------------------------


def _format_entry(e: dict) -> str:
    verdict = e.get("verdict", "?")
    hyp = e.get("hypothesis", "")
    predicted = e.get("predicted_delta", 0.0)
    actual = e.get("actual_delta", 0.0)
    entry_id = e.get("id", "?")[:8]
    tags = ", ".join(e.get("tags", []))
    line = f"[{entry_id}] [{verdict:>8}] {hyp}"
    if verdict != "crashed":
        line += f"  (predicted: {predicted:+.6f}, actual: {actual:+.6f})"
    if tags:
        line += f"  tags: {tags}"
    return line


def main():
    parser = argparse.ArgumentParser(description="Query the hypothesis journal")
    parser.add_argument("--path", default=None, help="Path to journal JSONL file")
    parser.add_argument("--verdict", default=None, help="Filter by verdict")
    parser.add_argument("--tags", default=None, help="Filter by tags (comma-separated)")
    parser.add_argument("--search", default=None, help="Full-text search query")
    parser.add_argument("--recent", type=int, default=None, help="Show last N entries")
    parser.add_argument("--stats", action="store_true", help="Show summary statistics")
    parser.add_argument("--context", action="store_true", help="Generate LLM context string")
    args = parser.parse_args()

    reader = JournalReader(path=args.path)

    if len(reader) == 0:
        print("Journal is empty.")
        return

    if args.stats:
        stats = reader.summary_stats()
        print("=== Journal Summary ===")
        for k, v in stats.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.4f}")
            else:
                print(f"  {k}: {v}")
        corr = reader.prediction_correlation()
        print(f"  prediction_correlation: {corr:.4f}")
        return

    if args.context:
        print(reader.generate_context())
        return

    # Apply filters
    if args.verdict:
        entries = reader.by_verdict(args.verdict)
    elif args.tags:
        entries = reader.by_tags(args.tags.split(","))
    elif args.search:
        entries = reader.search(args.search)
    elif args.recent:
        entries = reader.recent(args.recent)
    else:
        entries = reader.recent(20)

    print(f"Showing {len(entries)} entries:\n")
    for e in entries:
        print(_format_entry(e))


if __name__ == "__main__":
    main()
