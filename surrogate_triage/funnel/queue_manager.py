"""
Phase 3.2 — QueueManager: manage the evaluation queue of paper candidates.

Stores an ordered list of QueueEntry objects in evaluation_queue.json,
with deduplication, priority ordering, and max queue size enforcement.
"""

import json
import logging
import math
import os
import time

from surrogate_triage.schemas import QueueEntry

logger = logging.getLogger(__name__)

_DEFAULT_MAX_SIZE = 50
_DEDUP_THRESHOLD = 0.95


def _cosine_similarity(a: list, b: list) -> float:
    """Compute cosine similarity between two numeric vectors."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


class QueueManager:
    """Priority queue for paper-sourced evaluation candidates."""

    def __init__(
        self,
        queue_path: str = "evaluation_queue.json",
        journal_path: str = None,
        max_size: int = _DEFAULT_MAX_SIZE,
        dedup_threshold: float = _DEDUP_THRESHOLD,
    ):
        self.queue_path = queue_path
        self.journal_path = journal_path
        self.max_size = max_size
        self.dedup_threshold = dedup_threshold
        self._queue: list[QueueEntry] = []

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self):
        """Persist queue to disk."""
        os.makedirs(os.path.dirname(self.queue_path) or ".", exist_ok=True)
        data = [e.to_dict() for e in self._queue]
        with open(self.queue_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def load(self):
        """Load queue from disk."""
        if not os.path.exists(self.queue_path):
            self._queue = []
            return
        try:
            with open(self.queue_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self._queue = [QueueEntry.from_dict(d) for d in data]
        except (json.JSONDecodeError, IOError) as exc:
            logger.warning("Failed to load queue from %s: %s", self.queue_path, exc)
            self._queue = []

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------

    def add(self, entry: QueueEntry) -> bool:
        """Insert a candidate into the queue.

        Returns True if added, False if deduplicated or queue full with lower priority.
        """
        # Deduplication against existing queue
        if self._is_duplicate(entry):
            logger.info("Duplicate detected, skipping diff %s", entry.diff_id)
            return False

        # Deduplication against journal entries tagged source:paper
        if self._is_journal_duplicate(entry):
            logger.info(
                "Diff %s already evaluated (journal duplicate), skipping",
                entry.diff_id,
            )
            return False

        entry.priority = entry.adjusted_score
        self._queue.append(entry)
        self._sort()

        # Enforce max size — keep the best (lowest adjusted_score)
        if len(self._queue) > self.max_size:
            self._queue = self._queue[: self.max_size]

        return True

    def pop_next(self):
        """Remove and return the highest-priority (lowest adjusted_score) entry.

        Returns QueueEntry or None if queue is empty.
        """
        pending = [e for e in self._queue if e.status == "pending"]
        if not pending:
            return None
        entry = pending[0]
        entry.status = "evaluating"
        return entry

    def remove(self, queue_id: str):
        """Remove a specific entry by queue_id."""
        self._queue = [e for e in self._queue if e.queue_id != queue_id]

    def get_all(self) -> list:
        """Return all queue entries."""
        return list(self._queue)

    def stats(self) -> dict:
        """Return queue statistics for PipelineMonitor."""
        total = len(self._queue)
        pending = sum(1 for e in self._queue if e.status == "pending")
        evaluating = sum(1 for e in self._queue if e.status == "evaluating")
        evaluated = sum(1 for e in self._queue if e.status == "evaluated")
        return {
            "total": total,
            "pending": pending,
            "evaluating": evaluating,
            "evaluated": evaluated,
            "max_size": self.max_size,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _sort(self):
        """Sort queue by adjusted_score ascending (most negative first)."""
        self._queue.sort(key=lambda e: e.adjusted_score)

    def _is_duplicate(self, entry: QueueEntry) -> bool:
        """Check if entry is a near-duplicate of an existing queue item."""
        for existing in self._queue:
            if existing.diff_id == entry.diff_id:
                return True
            # Use diff text similarity as a proxy when no embedding available
            sim = _text_jaccard(existing.diff_text, entry.diff_text)
            if sim > self.dedup_threshold:
                return True
        return False

    def _is_journal_duplicate(self, entry: QueueEntry) -> bool:
        """Check if this diff was already evaluated (in journal with source:paper tag)."""
        if not self.journal_path or not os.path.exists(self.journal_path):
            return False
        try:
            with open(self.journal_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        j = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    tags = j.get("tags", [])
                    if "source:paper" not in tags:
                        continue
                    journal_diff = j.get("modification_diff", "")
                    sim = _text_jaccard(journal_diff, entry.diff_text)
                    if sim > self.dedup_threshold:
                        return True
        except IOError:
            pass
        return False


def _text_jaccard(a: str, b: str) -> float:
    """Token-level Jaccard similarity as a lightweight dedup proxy."""
    if not a or not b:
        return 0.0
    tokens_a = set(a.split())
    tokens_b = set(b.split())
    if not tokens_a or not tokens_b:
        return 0.0
    intersection = tokens_a & tokens_b
    union = tokens_a | tokens_b
    return len(intersection) / len(union)
