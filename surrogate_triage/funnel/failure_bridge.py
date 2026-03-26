"""
Phase 3.10 — FailureMiningBridge: feed paper rejections into the existing
FailureExtractor/FailureClusterer pipeline with paper-specific tagging.
"""

import json
import logging
import os
import time

logger = logging.getLogger(__name__)


class FailureMiningBridge:
    """Bridge between paper evaluation rejections and failure mining pipeline."""

    def feed_rejection(
        self,
        journal_entry: dict,
        paper_metadata: dict,
        journal_path: str,
    ):
        """Feed a paper rejection into the failure mining pipeline.

        Appends a tagged journal entry to the journal so that
        FailureExtractor picks it up on the next clustering pass.

        Args:
            journal_entry: The journal entry dict for the rejected experiment.
            paper_metadata: Paper metadata dict (arxiv_id, title, authors, etc.).
            journal_path: Path to the hypothesis journal JSONL file.
        """
        if not journal_entry or not journal_path:
            logger.warning("Missing journal_entry or journal_path, skipping rejection feed")
            return

        verdict = journal_entry.get("verdict", "")
        if verdict not in ("rejected", "crashed"):
            logger.debug("Entry verdict is '%s', not a rejection — skipping", verdict)
            return

        # Enrich the entry with paper-sourced failure tags
        tags = list(journal_entry.get("tags", []))
        paper_tags = [
            "source:paper",
            "failure_source:paper",
            f"paper_id:{paper_metadata.get('arxiv_id', '')}",
        ]
        for tag in paper_tags:
            if tag not in tags:
                tags.append(tag)

        enriched = dict(journal_entry)
        enriched["tags"] = tags

        # Add paper context to diagnostics_summary
        diag = dict(enriched.get("diagnostics_summary", {}))
        diag["paper_rejection_context"] = {
            "arxiv_id": paper_metadata.get("arxiv_id", ""),
            "paper_title": paper_metadata.get("title", ""),
            "technique_category": paper_metadata.get("modification_category", ""),
            "fed_to_failure_mining": True,
            "fed_at": time.time(),
        }
        enriched["diagnostics_summary"] = diag

        # Write enriched entry to journal (FailureExtractor reads from journal)
        try:
            os.makedirs(os.path.dirname(journal_path) or ".", exist_ok=True)
            with open(journal_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(enriched) + "\n")
            logger.info(
                "Fed paper rejection to failure mining: paper_id=%s",
                paper_metadata.get("arxiv_id", "?"),
            )
        except IOError as exc:
            logger.error("Failed to write rejection to journal: %s", exc)

    def update_failure_patterns(self, journal_path: str):
        """Trigger re-extraction and re-clustering of failure patterns.

        Args:
            journal_path: Path to the hypothesis journal JSONL file.
        """
        if not journal_path or not os.path.exists(journal_path):
            logger.warning("Journal path %s does not exist, skipping update", journal_path)
            return

        try:
            from model_scientist.failure_mining.extractor import FailureExtractor
            from model_scientist.failure_mining.clusterer import FailureClusterer

            extractor = FailureExtractor()
            failures = extractor.extract(journal_path)

            if not failures:
                logger.info("No failures extracted from journal, skipping clustering")
                return

            clusterer = FailureClusterer()
            patterns = clusterer.cluster(failures)

            # Save updated patterns
            data_dir = os.path.dirname(journal_path)
            patterns_path = os.path.join(data_dir, "failure_patterns.json")
            pattern_dicts = [
                p.to_dict() if hasattr(p, "to_dict") else p
                for p in patterns
            ]
            with open(patterns_path, "w", encoding="utf-8") as f:
                json.dump(pattern_dicts, f, indent=2)

            logger.info(
                "Updated failure patterns: %d failures → %d patterns",
                len(failures), len(patterns),
            )

        except ImportError as exc:
            logger.warning(
                "Could not import failure mining components: %s. "
                "Failure pattern update skipped.",
                exc,
            )
        except Exception as exc:
            logger.error("Failed to update failure patterns: %s", exc)
