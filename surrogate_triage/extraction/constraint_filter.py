"""
Phase 1.3 -- ConstraintPreFilter: run diffs against failure constraints before
surrogate scoring.  Produces a penalty score (not a hard filter).
"""

import ast
import math
import re

from surrogate_triage.schemas import SyntheticDiff

# Import FailureExtractor for feature vector extraction
from model_scientist.failure_mining.extractor import (
    FailureExtractor,
    _CATEGORY_KEYWORDS as _FAILURE_CATEGORY_KEYWORDS,
    _classify_category,
    _CATEGORY_LIST,
    _FAILURE_MODE_LIST,
    _DIAG_KEYS,
)
from model_scientist.schemas import FailureFeatures


def _diff_to_failure_features(diff: SyntheticDiff,
                              diagnostics_snapshot: dict | None = None) -> FailureFeatures:
    """Synthesize a FailureFeatures object from a SyntheticDiff.

    Since the diff hasn't been run yet, we use the diff text for category
    classification and supply any available diagnostics snapshot.
    """
    category = _classify_category(diff.diff_text) if diff.diff_text else "other"
    # Prefer the diff's own category if it's set and specific
    if diff.modification_category and diff.modification_category != "other":
        category = diff.modification_category

    return FailureFeatures(
        journal_id=diff.diff_id,
        modification_category=category,
        diagnostics_snapshot=diagnostics_snapshot or {},
        predicted_delta=0.0,
        actual_delta=0.0,
        failure_mode="",  # unknown -- hasn't run yet
    )


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    if len(a) != len(b) or not a:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a < 1e-12 or norm_b < 1e-12:
        return 0.0
    return dot / (norm_a * norm_b)


class ConstraintPreFilter:
    """Run diffs against failure constraints before surrogate scoring.

    Produces a penalty score (0 = no match, higher = stronger match to known
    failure patterns).  Diffs are NOT discarded -- the penalty feeds into
    surrogate ranking.
    """

    def compute_penalty(
        self,
        diff: SyntheticDiff,
        constraints: list,
        diagnostics_snapshot: dict | None = None,
    ) -> float:
        """Compute constraint penalty for a single diff.

        Parameters
        ----------
        diff : SyntheticDiff
        constraints : list
            List of NegativeConstraint objects (from model_scientist.schemas).
        diagnostics_snapshot : dict, optional
            Current diagnostics state to incorporate into the feature vector.

        Returns
        -------
        float
            Penalty score (0.0 = clean, higher = more matches).
        """
        if not constraints:
            return 0.0

        # Build synthetic failure features from the diff
        synth_features = _diff_to_failure_features(diff, diagnostics_snapshot)
        feature_vec = FailureExtractor.extract_features_vector(synth_features)

        # Store on the diff for downstream use
        diff.failure_feature_vector = feature_vec

        penalty = 0.0

        for constraint in constraints:
            match_score = self._score_constraint_match(
                diff, synth_features, feature_vec, constraint
            )
            penalty += match_score

        return penalty

    def filter_batch(
        self,
        diffs: list[SyntheticDiff],
        constraints: list,
        diagnostics_snapshot: dict | None = None,
    ) -> list[SyntheticDiff]:
        """Compute constraint penalties for a batch of diffs.

        Parameters
        ----------
        diffs : list[SyntheticDiff]
        constraints : list
            List of NegativeConstraint objects.
        diagnostics_snapshot : dict, optional
            Current diagnostics state.

        Returns
        -------
        list[SyntheticDiff]
            Same list with ``constraint_penalty`` and ``failure_feature_vector``
            populated on each diff.
        """
        for diff in diffs:
            diff.constraint_penalty = self.compute_penalty(
                diff, constraints, diagnostics_snapshot
            )
        return diffs

    def _score_constraint_match(
        self,
        diff: SyntheticDiff,
        synth_features: FailureFeatures,
        feature_vec: list[float],
        constraint,
    ) -> float:
        """Score how well a diff matches a single constraint.

        Uses category matching and text overlap to produce a score in [0, 1].
        """
        score = 0.0
        text = getattr(constraint, "text", "") or ""
        text_lower = text.lower()

        # 1. Category match: if the constraint mentions the same modification type
        cat = synth_features.modification_category or ""
        if cat and cat != "other" and cat in text_lower:
            score += 0.4

        # 2. Keyword overlap between diff text and constraint text
        diff_words = set(re.findall(r'\b\w{4,}\b', (diff.diff_text or "").lower()))
        constraint_words = set(re.findall(r'\b\w{4,}\b', text_lower))
        if diff_words and constraint_words:
            overlap = len(diff_words & constraint_words)
            union = len(diff_words | constraint_words)
            if union > 0:
                jaccard = overlap / union
                score += 0.3 * jaccard

        # 3. If the constraint has a centroid feature vector embedded in it,
        #    compute cosine similarity (constraints carry pattern_id which links
        #    back to FailurePattern centroid, but we don't have direct access
        #    here, so we use text-based heuristics instead).

        # 4. Weight by constraint validity
        is_valid = getattr(constraint, "is_valid", False)
        precision = getattr(constraint, "precision", 0.0) or 0.0
        if is_valid and precision > 0:
            score *= (0.5 + 0.5 * precision)  # validated constraints count more

        return min(score, 1.0)
