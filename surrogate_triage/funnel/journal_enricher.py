"""
Phase 3.5 — PaperJournalEnricher: enrich journal entries with paper metadata
after evaluation completes.
"""

import logging

from surrogate_triage.schemas import QueueEntry

logger = logging.getLogger(__name__)


class PaperJournalEnricher:
    """Enrich journal entries with paper-specific metadata after evaluation."""

    def enrich(
        self,
        journal_entry_dict: dict,
        queue_entry: QueueEntry,
        actual_delta: float,
    ) -> dict:
        """Enrich a journal entry dict with paper metadata.

        Args:
            journal_entry_dict: The journal entry as a dict (from pipeline result).
            queue_entry: The QueueEntry that was evaluated.
            actual_delta: The actual val_bpb delta observed.

        Returns:
            Enriched journal entry dict with additional tags and diagnostics fields.
        """
        if not isinstance(journal_entry_dict, dict):
            logger.warning("journal_entry_dict is not a dict, returning as-is")
            return journal_entry_dict

        entry = dict(journal_entry_dict)

        # Enrich tags
        tags = list(entry.get("tags", []))
        paper_tags = [
            "source:paper",
            f"paper_id:{queue_entry.paper_id}",
            f"paper_title:{queue_entry.paper_title}",
            f"technique_name:{queue_entry.technique_name}",
            f"surrogate_score:{queue_entry.surrogate_score:.4f}",
            f"diff_id:{queue_entry.diff_id}",
            f"technique_id:{queue_entry.technique_id}",
        ]
        for tag in paper_tags:
            if tag not in tags:
                tags.append(tag)
        entry["tags"] = tags

        # Enrich diagnostics_summary with paper-specific fields
        diag = dict(entry.get("diagnostics_summary", {}))
        diag["paper_metadata"] = {
            "arxiv_id": queue_entry.paper_id,
            "paper_title": queue_entry.paper_title,
            "technique_name": queue_entry.technique_name,
            "technique_id": queue_entry.technique_id,
            "diff_id": queue_entry.diff_id,
        }
        diag["surrogate_comparison"] = {
            "predicted_delta": queue_entry.surrogate_score,
            "actual_delta": actual_delta,
            "prediction_error": queue_entry.surrogate_score - actual_delta,
            "constraint_penalty": queue_entry.constraint_penalty,
            "adjusted_score": queue_entry.adjusted_score,
        }
        entry["diagnostics_summary"] = diag

        return entry
