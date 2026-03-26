"""
Phase 2.1 — ConstraintValidator: back-test NegativeConstraints against
historical journal entries and compute precision / recall.
"""

from model_scientist.schemas import JournalEntry, NegativeConstraint


class ConstraintValidator:
    """Validate constraints by checking them against historical experiments."""

    def __init__(self):
        self._stats: dict = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def validate(
        self,
        constraints: list[NegativeConstraint],
        journal_entries: list,
    ) -> list[NegativeConstraint]:
        """Back-test each constraint. Returns constraints with filled metrics.

        A constraint "flags" an experiment if the constraint's modification
        type matches the experiment's diff category AND the experiment's
        diagnostics overlap with the centroid conditions encoded in the
        constraint text.

        A constraint is valid when:
          - precision > 0.5  (majority of flagged experiments were failures)
          - recall does not filter > 10% of successes
        """
        if not constraints or not journal_entries:
            return constraints

        entries = [
            JournalEntry.from_dict(e) if isinstance(e, dict) else e
            for e in journal_entries
        ]

        failures = [e for e in entries if e.verdict in ("rejected", "crashed")]
        successes = [e for e in entries if e.verdict == "accepted"]

        validated: list[NegativeConstraint] = []
        total_valid = 0

        for c in constraints:
            flagged_failures = 0
            flagged_successes = 0

            for entry in failures:
                if self._constraint_matches(c, entry):
                    flagged_failures += 1

            for entry in successes:
                if self._constraint_matches(c, entry):
                    flagged_successes += 1

            total_flagged = flagged_failures + flagged_successes
            precision = flagged_failures / total_flagged if total_flagged > 0 else 0.0
            recall = flagged_failures / len(failures) if failures else 0.0

            # Check that we don't filter too many successes.
            success_filter_rate = (
                flagged_successes / len(successes) if successes else 0.0
            )
            is_valid = precision > 0.5 and success_filter_rate <= 0.10

            c.precision = precision
            c.recall = recall
            c.is_valid = is_valid

            if is_valid:
                total_valid += 1

            validated.append(c)

        self._stats = {
            "total_constraints": len(validated),
            "valid_constraints": total_valid,
            "total_journal_entries": len(entries),
            "total_failures": len(failures),
            "total_successes": len(successes),
        }

        return validated

    def summary(self) -> dict:
        """Return validation summary statistics."""
        return dict(self._stats)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _constraint_matches(constraint: NegativeConstraint, entry: JournalEntry) -> bool:
        """Heuristic: does *constraint* flag *entry*?

        Matching criteria:
        1. The constraint's modification type appears in the entry's diff.
        2. At least one diagnostic keyword from the constraint text appears
           in the entry's diagnostics summary.
        """
        text_lower = constraint.text.lower()
        diff_lower = (entry.modification_diff or "").lower()
        diag_str = str(entry.diagnostics_summary or {}).lower()

        # Extract the modification type from the constraint text.
        # Template: "... {mod_type} modifications have failed ..."
        mod_type = ""
        if "modifications have failed" in text_lower:
            before = text_lower.split("modifications have failed")[0]
            parts = before.strip().rsplit(",", 1)
            mod_type = parts[-1].strip().split()[-1] if parts else ""

        if not mod_type:
            return False

        # Check if the modification type keyword is in the diff.
        if mod_type not in diff_lower:
            # Also check common keywords for the category.
            from model_scientist.failure_mining.extractor import _CATEGORY_KEYWORDS

            category_keywords = _CATEGORY_KEYWORDS.get(mod_type, [])
            if not any(kw in diff_lower for kw in category_keywords):
                return False

        # Check if any diagnostic condition from the constraint text
        # has overlap with the entry's diagnostics.
        # Extract diagnostic keys mentioned in constraint text.
        diag_keywords = [
            "val_bpb", "train_loss", "grad_norm", "grad_dead",
            "activation", "attention_entropy", "attention_collapse",
        ]
        matched_diag = any(
            kw in text_lower and kw in diag_str for kw in diag_keywords
        )

        # If the constraint mentions diagnostics, require at least one match.
        has_diag_in_constraint = any(kw in text_lower for kw in diag_keywords)
        if has_diag_in_constraint and not matched_diag:
            return False

        return True
