"""
Phase 2 — JournalDataExtractor: extract training data for the surrogate
model from the hypothesis journal.
"""

import json
import logging
import os

from model_scientist.schemas import JournalEntry, load_jsonl
from surrogate_triage.schemas import SurrogateTrainingExample
from surrogate_triage.surrogate.feature_enricher import FeatureEnricher

logger = logging.getLogger(__name__)


class JournalDataExtractor:
    """Extract surrogate training examples from hypothesis_journal.jsonl.

    For each journal entry, extracts:
        - modification_diff -> code embedding (via FeatureEnricher)
        - diagnostics_summary -> failure features
        - actual_delta -> training label
    """

    def extract(
        self,
        journal_path: str,
        enricher: FeatureEnricher | None = None,
        metric_registry=None,
    ) -> list[SurrogateTrainingExample]:
        """Extract training examples from the journal.

        Args:
            journal_path: Path to hypothesis_journal.jsonl.
            enricher: FeatureEnricher instance. Creates a default one if None.
            metric_registry: Optional MetricRegistry for metric features.

        Returns:
            List of SurrogateTrainingExample with feature vectors and labels.
        """
        raw = load_jsonl(journal_path)
        if not raw:
            logger.warning("No entries found in %s", journal_path)
            return []

        if enricher is None:
            enricher = FeatureEnricher()

        examples = []
        skipped = 0

        for data in raw:
            entry = JournalEntry.from_dict(data)

            # Skip crashed entries (no reliable delta) and entries without diffs
            if entry.verdict == "crashed":
                skipped += 1
                continue
            if not entry.modification_diff:
                skipped += 1
                continue

            try:
                enriched = enricher.enrich(
                    diff_text=entry.modification_diff,
                    diagnostics_snapshot=entry.diagnostics_summary,
                    metric_registry=metric_registry,
                    diff_id=entry.id,
                )

                # Determine source: check tags for paper-sourced entries
                source = "internal"
                tags = entry.tags or []
                if any("paper" in t.lower() for t in tags):
                    source = "paper"

                example = SurrogateTrainingExample(
                    journal_id=entry.id,
                    feature_vector=enriched.combined_vector,
                    actual_delta=entry.actual_delta or 0.0,
                    source=source,
                    tags=tags,
                )
                examples.append(example)

            except Exception as exc:
                logger.warning(
                    "Failed to extract features for entry %s: %s",
                    entry.id, exc,
                )
                skipped += 1

        logger.info(
            "Extracted %d training examples from %s (%d skipped)",
            len(examples), journal_path, skipped,
        )
        return examples

    @staticmethod
    def save_training_data(examples: list[SurrogateTrainingExample], path: str):
        """Save training examples to a JSON file.

        Args:
            examples: List of SurrogateTrainingExample.
            path: Output file path.
        """
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)

        data = []
        for ex in examples:
            data.append({
                "journal_id": ex.journal_id,
                "feature_vector": ex.feature_vector,
                "actual_delta": ex.actual_delta,
                "source": ex.source,
                "tags": ex.tags,
            })

        with open(path, "w") as f:
            json.dump(data, f)

        logger.info("Saved %d training examples to %s", len(examples), path)

    @staticmethod
    def load_training_data(path: str) -> list[SurrogateTrainingExample]:
        """Load training examples from a JSON file.

        Args:
            path: Path to the JSON file.

        Returns:
            List of SurrogateTrainingExample.
        """
        if not os.path.exists(path):
            logger.warning("Training data file not found: %s", path)
            return []

        with open(path) as f:
            data = json.load(f)

        examples = []
        for item in data:
            ex = SurrogateTrainingExample(
                journal_id=item.get("journal_id", ""),
                feature_vector=item.get("feature_vector", []),
                actual_delta=item.get("actual_delta", 0.0),
                source=item.get("source", ""),
                tags=item.get("tags", []),
            )
            examples.append(ex)

        logger.info("Loaded %d training examples from %s", len(examples), path)
        return examples
