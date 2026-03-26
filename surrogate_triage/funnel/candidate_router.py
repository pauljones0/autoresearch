"""
Phase 3.4 — PaperCandidateRouter: route a QueueEntry through
ModelScientistPipeline.evaluate_modification().
"""

import difflib
import logging

from surrogate_triage.schemas import QueueEntry

logger = logging.getLogger(__name__)


class PaperCandidateRouter:
    """Route paper candidates through the Model Scientist evaluation pipeline."""

    def route(
        self,
        entry: QueueEntry,
        base_source: str,
        pipeline=None,
    ) -> dict:
        """Route a queue entry through the evaluation pipeline.

        Args:
            entry: QueueEntry with diff_text, hypothesis, surrogate_score, etc.
            base_source: Current train.py source code.
            pipeline: ModelScientistPipeline instance (or None for dry-run).

        Returns:
            Dict with evaluation result including verdict, delta, etc.
            On failure returns a dict with verdict="crashed" and error info.
        """
        # Apply the diff to get modified source
        modified_source = self.apply_diff(entry.diff_text, base_source)
        if modified_source is None:
            logger.warning(
                "Diff application failed for queue entry %s (diff_id=%s)",
                entry.queue_id, entry.diff_id,
            )
            return {
                "verdict": "crashed",
                "error": "diff_application_failed",
                "queue_id": entry.queue_id,
                "diff_id": entry.diff_id,
            }

        # Build hypothesis string from technique metadata
        hypothesis = entry.hypothesis
        if not hypothesis:
            hypothesis = (
                f"Paper technique: {entry.technique_name} "
                f"(paper: {entry.paper_title or entry.paper_id})"
            )

        # Build tags
        tags = [
            "source:paper",
            f"paper_id:{entry.paper_id}",
            f"surrogate_score:{entry.surrogate_score:.4f}",
        ]

        # Route through pipeline
        if pipeline is None:
            logger.info(
                "No pipeline provided, dry-run for entry %s", entry.queue_id
            )
            return {
                "verdict": "dry_run",
                "queue_id": entry.queue_id,
                "diff_id": entry.diff_id,
                "hypothesis": hypothesis,
                "tags": tags,
                "modified_source_length": len(modified_source),
            }

        try:
            result = pipeline.evaluate_modification(
                modified_source=modified_source,
                hypothesis=hypothesis,
                predicted_delta=entry.surrogate_score,
                modification_diff=entry.diff_text,
                tags=tags,
            )
            result["queue_id"] = entry.queue_id
            result["diff_id"] = entry.diff_id
            return result

        except Exception as exc:
            logger.error(
                "Pipeline evaluation failed for entry %s: %s",
                entry.queue_id, exc,
            )
            return {
                "verdict": "crashed",
                "error": str(exc),
                "queue_id": entry.queue_id,
                "diff_id": entry.diff_id,
            }

    @staticmethod
    def apply_diff(diff_text: str, base_source: str) -> str:
        """Apply a unified diff to base_source.

        Args:
            diff_text: Unified diff text.
            base_source: Original source code.

        Returns:
            Modified source string, or None if the diff cannot be applied.
        """
        if not diff_text or not base_source:
            return None

        try:
            base_lines = base_source.splitlines(keepends=True)
            # Parse the unified diff and apply hunks
            patched = _apply_unified_diff(base_lines, diff_text)
            if patched is not None:
                return "".join(patched)
        except Exception as exc:
            logger.debug("Unified diff application failed: %s", exc)

        # Fallback: if the diff is actually a full replacement source
        # (some synthetic diffs are generated as complete files)
        if not diff_text.startswith(("---", "@@", "diff ")):
            # Likely a full source, not a diff
            return diff_text if len(diff_text) > 50 else None

        return None


def _apply_unified_diff(base_lines: list, diff_text: str) -> list:
    """Apply a unified diff to a list of lines.

    Returns the patched lines or None on failure.
    """
    result = list(base_lines)
    offset = 0

    hunks = _parse_hunks(diff_text)
    if not hunks:
        return None

    for start, count, new_lines in hunks:
        idx = start - 1 + offset  # Convert 1-based to 0-based
        if idx < 0:
            idx = 0

        # Remove old lines
        del result[idx: idx + count]
        # Insert new lines
        for i, line in enumerate(new_lines):
            result.insert(idx + i, line)

        offset += len(new_lines) - count

    return result


def _parse_hunks(diff_text: str) -> list:
    """Parse unified diff hunks.

    Returns list of (start_line, old_count, new_lines) tuples.
    """
    import re

    hunks = []
    lines = diff_text.splitlines(keepends=True)
    i = 0

    while i < len(lines):
        line = lines[i]
        match = re.match(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@", line)
        if match:
            old_start = int(match.group(1))
            old_count = int(match.group(2)) if match.group(2) else 1
            i += 1

            new_lines = []
            removed = 0
            while i < len(lines):
                hline = lines[i]
                if hline.startswith("@@") or hline.startswith("diff "):
                    break
                if hline.startswith("-"):
                    removed += 1
                    i += 1
                elif hline.startswith("+"):
                    new_lines.append(hline[1:])
                    i += 1
                elif hline.startswith(" "):
                    new_lines.append(hline[1:])
                    i += 1
                else:
                    # Context or unrecognized line
                    i += 1

            hunks.append((old_start, old_count, new_lines))
        else:
            i += 1

    return hunks
