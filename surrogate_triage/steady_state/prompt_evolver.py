"""
Phase 5 — ExtractionPromptEvolver: analyze extraction failures, classify
failure modes, and suggest prompt refinements for technique extraction.
"""

import json
import os
import time


# Canonical failure-mode categories
FAILURE_MODES = (
    "missed_critical_detail",
    "diff_misinterpreted_pseudocode",
    "technique_incompatible_with_train",
)


class ExtractionPromptEvolver:
    """Analyze paper extraction failures and evolve prompts accordingly.

    A failure is defined as a paper-sourced evaluation where the absolute
    prediction error exceeds *threshold* — meaning the surrogate prediction
    diverged significantly from the actual outcome, indicating the extracted
    representation was poor.
    """

    def __init__(self, data_dir: str = "."):
        self.data_dir = data_dir
        self._prompt_versions_path = os.path.join(data_dir, "prompt_versions.jsonl")

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def analyze_failures(
        self,
        evaluation_records: list,
        threshold: float = 0.05,
    ) -> dict:
        """Analyze evaluation records for extraction failures.

        Args:
            evaluation_records: list of dicts with at least:
                surrogate_predicted_delta, actual_delta,
                extraction_confidence, technique_id, diff_text (optional),
                modification_category (optional).
            threshold: absolute prediction error above which we flag a failure.

        Returns:
            dict with 'failure_modes' (categorised counts) and
            'recommendations' (list of str).
        """
        failures = []
        for rec in evaluation_records:
            pred = rec.get("surrogate_predicted_delta", 0.0)
            actual = rec.get("actual_delta", 0.0)
            error = abs(pred - actual)
            if error > threshold:
                failures.append({**rec, "_error": error})

        mode_counts = {m: 0 for m in FAILURE_MODES}
        mode_examples: dict[str, list] = {m: [] for m in FAILURE_MODES}

        for f in failures:
            mode = self._classify_failure(f)
            mode_counts[mode] += 1
            mode_examples[mode].append(f.get("technique_id", ""))

        recommendations = self._generate_recommendations(mode_counts, len(failures))

        return {
            "total_evaluated": len(evaluation_records),
            "total_failures": len(failures),
            "failure_modes": mode_counts,
            "failure_examples": mode_examples,
            "recommendations": recommendations,
        }

    def suggest_prompt_refinements(self, failure_analysis: dict) -> list:
        """Generate concrete prompt refinement suggestions from failure analysis.

        Args:
            failure_analysis: output of analyze_failures().

        Returns:
            List of refinement suggestion strings.
        """
        suggestions = []
        modes = failure_analysis.get("failure_modes", {})
        total = failure_analysis.get("total_failures", 0)
        if total == 0:
            return ["No failures detected — current prompt is performing well."]

        if modes.get("missed_critical_detail", 0) > 0:
            frac = modes["missed_critical_detail"] / total
            suggestions.append(
                f"Add explicit instruction to extract ALL hyperparameter values, "
                f"initialization schemes, and architectural dimensions mentioned "
                f"in the paper. ({frac:.0%} of failures are missed-detail.)"
            )
            suggestions.append(
                "Include a checklist in the prompt: 'For each technique, confirm "
                "you have captured: (1) exact layer/module affected, (2) numerical "
                "constants, (3) activation function changes, (4) any conditional logic.'"
            )

        if modes.get("diff_misinterpreted_pseudocode", 0) > 0:
            frac = modes["diff_misinterpreted_pseudocode"] / total
            suggestions.append(
                f"Add instruction: 'When the paper uses pseudocode, map each "
                f"pseudocode variable to the corresponding variable in train.py "
                f"before generating the diff.' ({frac:.0%} of failures are "
                f"pseudocode misinterpretations.)"
            )

        if modes.get("technique_incompatible_with_train", 0) > 0:
            frac = modes["technique_incompatible_with_train"] / total
            suggestions.append(
                f"Add a pre-check step: 'Before generating the diff, verify that "
                f"the technique is applicable to the model architecture in train.py. "
                f"If it requires components not present (e.g., convolutions in a "
                f"transformer), mark as inapplicable.' ({frac:.0%} of failures "
                f"are incompatibility issues.)"
            )

        return suggestions

    def track_prompt_versions(
        self,
        version: int,
        quality_metrics: dict,
    ) -> None:
        """Append a prompt version record to prompt_versions.jsonl.

        Args:
            version: Integer prompt version number.
            quality_metrics: dict of quality metrics for this version.
        """
        record = {
            "version": version,
            "quality_metrics": quality_metrics,
            "timestamp": time.time(),
        }
        with open(self._prompt_versions_path, "a") as f:
            f.write(json.dumps(record) + "\n")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _classify_failure(record: dict) -> str:
        """Heuristic classification of a failure into one of the three modes."""
        confidence = record.get("extraction_confidence", 1.0)
        diff_text = record.get("diff_text", "")
        category = record.get("modification_category", "")
        error = record.get("_error", 0.0)

        # Low extraction confidence + large error → likely missed detail
        if confidence < 0.5:
            return "missed_critical_detail"

        # If the diff is empty or very short despite having a technique,
        # pseudocode was likely misinterpreted
        if diff_text and len(diff_text.strip()) < 50:
            return "diff_misinterpreted_pseudocode"

        # If the category doesn't match common train.py modification types,
        # the technique may be incompatible
        compatible_categories = {
            "architecture", "optimizer", "hyperparameter", "activation",
            "initialization", "normalization", "regularization", "attention",
            "embedding", "loss",
        }
        if category and category.lower() not in compatible_categories:
            return "technique_incompatible_with_train"

        # If prediction error is very large, default to pseudocode misinterpretation
        if error > 0.1:
            return "diff_misinterpreted_pseudocode"

        return "missed_critical_detail"

    @staticmethod
    def _generate_recommendations(mode_counts: dict, total: int) -> list:
        """Generate high-level recommendations from failure mode distribution."""
        if total == 0:
            return ["No failures — extraction quality is good."]

        recs = []
        dominant = max(mode_counts, key=mode_counts.get)

        if dominant == "missed_critical_detail":
            recs.append(
                "Primary issue: extraction is missing critical details. "
                "Consider adding structured extraction templates."
            )
        elif dominant == "diff_misinterpreted_pseudocode":
            recs.append(
                "Primary issue: pseudocode-to-code translation errors. "
                "Consider adding explicit variable-mapping instructions."
            )
        elif dominant == "technique_incompatible_with_train":
            recs.append(
                "Primary issue: techniques incompatible with train.py. "
                "Consider adding architecture-compatibility pre-filtering."
            )

        if total > 20:
            recs.append(
                f"High failure count ({total}). Consider tightening the "
                f"extraction confidence threshold to filter low-quality extractions."
            )

        return recs
