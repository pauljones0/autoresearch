"""
Phase 1.2 -- ExtractionValidator: validate technique extractions for
completeness, quality, and constraint compliance.
"""

import re

from surrogate_triage.schemas import TechniqueDescription

_VALID_CATEGORIES = {
    "architecture", "optimizer", "hyperparameter", "activation",
    "initialization", "regularization", "scheduling", "other",
}

# Comparative language indicating a reported baseline
_COMPARATIVE_PATTERNS = [
    re.compile(r"(?:over|compared|than|versus|vs\.?|relative to|baseline|outperform)", re.I),
    re.compile(r"\d+(?:\.\d+)?\s*%", re.I),
    re.compile(r"(?:better|worse|higher|lower|improvement|reduction|gain)", re.I),
]


class ExtractionValidator:
    """Validate technique extractions for completeness and quality."""

    def validate(
        self,
        technique: TechniqueDescription,
        constraints: list = None,
    ) -> tuple[bool, list[str], list]:
        """Validate a single technique extraction.

        Parameters
        ----------
        technique : TechniqueDescription
            The technique to validate.
        constraints : list, optional
            List of NegativeConstraint objects to check against.

        Returns
        -------
        tuple[bool, list[str], list]
            (is_valid, warnings, constraint_matches)
            - is_valid: True if the extraction passes all critical checks.
            - warnings: list of warning strings for non-critical issues.
            - constraint_matches: list of matched NegativeConstraint objects.
        """
        warnings: list[str] = []
        constraint_matches: list = []
        is_valid = True

        # Check: technique name is non-empty
        if not technique.name or not technique.name.strip():
            warnings.append("Technique name is empty.")
            is_valid = False

        # Check: category is valid
        if technique.modification_category not in _VALID_CATEGORIES:
            warnings.append(
                f"Invalid category '{technique.modification_category}'. "
                f"Valid: {sorted(_VALID_CATEGORIES)}"
            )
            is_valid = False

        # Check: description has substance (>20 chars)
        desc = technique.description or ""
        if len(desc.strip()) <= 20:
            warnings.append(
                f"Description too short ({len(desc.strip())} chars, need >20)."
            )
            is_valid = False

        # Check: reported improvements include comparative language
        improvement = technique.reported_improvement or ""
        if improvement:
            has_comparative = any(
                p.search(improvement) for p in _COMPARATIVE_PATTERNS
            )
            if not has_comparative:
                warnings.append(
                    "Reported improvement lacks comparative language "
                    "(no baseline reference found)."
                )
        else:
            # Not a hard failure, but note it
            warnings.append("No reported improvement extracted.")

        # Check: extraction confidence is reasonable
        if technique.extraction_confidence < 0.1:
            warnings.append(
                f"Very low extraction confidence: {technique.extraction_confidence:.2f}"
            )

        # Check against negative constraints
        if constraints:
            constraint_matches = self._check_constraints(technique, constraints)
            if constraint_matches:
                warnings.append(
                    f"Matched {len(constraint_matches)} negative constraint(s)."
                )

        return is_valid, warnings, constraint_matches

    def _check_constraints(self, technique: TechniqueDescription,
                           constraints: list) -> list:
        """Check technique against active NegativeConstraints.

        A constraint matches if its modification_type (extracted from its text)
        aligns with the technique category, or if keyword overlap is detected.
        """
        matches = []
        cat = (technique.modification_category or "").lower()
        desc_lower = (technique.description or "").lower()
        name_lower = (technique.name or "").lower()

        for constraint in constraints:
            text = getattr(constraint, "text", "") or ""
            text_lower = text.lower()

            # Match if the constraint mentions the same modification category
            if cat and cat != "other" and cat in text_lower:
                matches.append(constraint)
                continue

            # Match if significant keyword overlap
            constraint_words = set(re.findall(r'\b\w{4,}\b', text_lower))
            technique_words = set(re.findall(r'\b\w{4,}\b', f"{name_lower} {desc_lower}"))
            if constraint_words and technique_words:
                overlap = len(constraint_words & technique_words)
                if overlap >= 2:
                    matches.append(constraint)

        return matches
