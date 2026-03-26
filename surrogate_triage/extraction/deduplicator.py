"""
Phase 1.2 -- TechniqueDeduplicator: detect duplicate techniques across papers
using Jaccard similarity on word sets.
"""

import json
import re

from surrogate_triage.schemas import TechniqueDescription


def _tokenize(text: str) -> set[str]:
    """Extract a set of lowercase words (length >= 3) from text."""
    if not text:
        return set()
    return set(re.findall(r'\b\w{3,}\b', text.lower()))


def _jaccard(a: set, b: set) -> float:
    """Jaccard similarity between two sets."""
    if not a or not b:
        return 0.0
    intersection = len(a & b)
    union = len(a | b)
    return intersection / union if union > 0 else 0.0


def _technique_text(tech: TechniqueDescription) -> str:
    """Combine technique fields into a single text for comparison."""
    parts = [
        tech.name or "",
        tech.description or "",
        tech.modification_category or "",
    ]
    return " ".join(parts)


class TechniqueDeduplicator:
    """Detect duplicate techniques across papers using text similarity."""

    def __init__(self, threshold: float = 0.6):
        """
        Parameters
        ----------
        threshold : float
            Jaccard similarity threshold above which two techniques are
            considered duplicates. Default 0.6.
        """
        self.threshold = threshold

    def deduplicate(self, techniques: list[TechniqueDescription]) -> list[TechniqueDescription]:
        """Deduplicate a list of techniques, marking duplicates.

        The first occurrence is kept as canonical. Later duplicates get
        ``deduplicated=True`` and ``duplicate_of`` set to the canonical
        technique_id.

        Parameters
        ----------
        techniques : list[TechniqueDescription]

        Returns
        -------
        list[TechniqueDescription]
            The same list with duplicate fields updated in-place.
        """
        if len(techniques) <= 1:
            return techniques

        # Precompute token sets
        token_sets = [_tokenize(_technique_text(t)) for t in techniques]

        # Mark duplicates (quadratic, fine for expected scale)
        canonical_indices: list[int] = []  # indices of canonical techniques

        for i, tech in enumerate(techniques):
            if tech.deduplicated:
                # Already marked from a previous run
                continue

            is_dup = False
            for ci in canonical_indices:
                sim = _jaccard(token_sets[i], token_sets[ci])
                if sim >= self.threshold:
                    tech.deduplicated = True
                    tech.duplicate_of = techniques[ci].technique_id
                    is_dup = True
                    break

            if not is_dup:
                canonical_indices.append(i)

        return techniques

    def check_against_journal(
        self,
        techniques: list[TechniqueDescription],
        journal_path: str,
    ) -> list[TechniqueDescription]:
        """Check techniques against a hypothesis journal for already-explored ideas.

        Parameters
        ----------
        techniques : list[TechniqueDescription]
        journal_path : str
            Path to a JSONL hypothesis journal file.

        Returns
        -------
        list[TechniqueDescription]
            The same list with ``already_explored`` and ``journal_entry_id``
            updated where matches are found.
        """
        journal_entries = self._load_journal(journal_path)
        if not journal_entries:
            return techniques

        # Build token sets for journal hypotheses
        journal_tokens = []
        for entry in journal_entries:
            hypothesis = entry.get("hypothesis", "")
            diff = entry.get("modification_diff", "")
            combined = f"{hypothesis} {diff}"
            journal_tokens.append((_tokenize(combined), entry))

        for tech in techniques:
            if tech.already_explored:
                continue

            tech_tokens = _tokenize(_technique_text(tech))
            for jtokens, entry in journal_tokens:
                sim = _jaccard(tech_tokens, jtokens)
                if sim >= self.threshold:
                    tech.already_explored = True
                    tech.journal_entry_id = entry.get("id", "")
                    break

        return techniques

    def _load_journal(self, path: str) -> list[dict]:
        """Load JSONL journal file."""
        entries = []
        try:
            with open(path) as f:
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
