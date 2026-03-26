"""
Phase 1.3 -- DiffApplicabilityChecker: validate generated diffs by applying
them to train.py and checking syntax + DiffParser compatibility.
"""

import ast
import difflib
import re

from surrogate_triage.schemas import SyntheticDiff

# Import DiffParser for decomposition check
from ...model_scientist.ablation.diff_parser import DiffParser


def _apply_unified_diff(diff_text: str, base_source: str) -> str | None:
    """Apply a unified diff to base source, returning the new source or None."""
    base_lines = base_source.splitlines(keepends=True)

    hunks = []
    current_hunk = None
    for line in diff_text.splitlines(keepends=True):
        if line.startswith('@@'):
            match = re.match(r'@@ -(\d+)(?:,\d+)? \+(\d+)(?:,\d+)? @@', line)
            if match:
                current_hunk = {
                    'old_start': int(match.group(1)),
                    'lines': [],
                }
                hunks.append(current_hunk)
        elif current_hunk is not None:
            if line.startswith('---') or line.startswith('+++'):
                continue
            current_hunk['lines'].append(line)

    if not hunks:
        return None

    result_lines = list(base_lines)
    for hunk in reversed(hunks):
        old_idx = hunk['old_start'] - 1
        remove_count = 0
        add_lines = []

        for line in hunk['lines']:
            if line.startswith('-'):
                remove_count += 1
            elif line.startswith('+'):
                content = line[1:]
                # Ensure line ends with newline
                if not content.endswith('\n'):
                    content += '\n'
                add_lines.append(content)
            elif line.startswith(' '):
                remove_count += 1
                content = line[1:]
                if not content.endswith('\n'):
                    content += '\n'
                add_lines.append(content)

        result_lines[old_idx:old_idx + remove_count] = add_lines

    return ''.join(result_lines)


class DiffApplicabilityChecker:
    """Test generated diffs for validity and decomposability."""

    def __init__(self):
        self._diff_parser = DiffParser()

    def check(self, diff: SyntheticDiff, base_source: str) -> SyntheticDiff:
        """Check a single diff for applicability.

        Fills in ``applies_cleanly``, ``is_decomposable``, and ``n_components``
        on the diff object.

        Parameters
        ----------
        diff : SyntheticDiff
            The diff to check.
        base_source : str
            The current content of train.py.

        Returns
        -------
        SyntheticDiff
            The same object with validity fields populated.
        """
        diff.applies_cleanly = False
        diff.is_decomposable = False
        diff.n_components = 0

        if not diff.diff_text or not base_source:
            return diff

        # Step 1: Apply diff to a copy of base_source
        new_source = _apply_unified_diff(diff.diff_text, base_source)
        if new_source is None:
            return diff

        # Step 2: Verify syntax with ast.parse
        try:
            ast.parse(new_source)
        except SyntaxError:
            return diff

        diff.applies_cleanly = True

        # Step 3: Check if DiffParser can decompose it
        try:
            components = self._diff_parser.parse(base_source, new_source)
            diff.is_decomposable = len(components) > 0
            diff.n_components = len(components)
        except Exception:
            # DiffParser failed -- still applies cleanly, just not decomposable
            diff.is_decomposable = False
            diff.n_components = 0

        return diff

    def check_batch(self, diffs: list[SyntheticDiff],
                    base_source: str) -> list[SyntheticDiff]:
        """Check a batch of diffs.

        Parameters
        ----------
        diffs : list[SyntheticDiff]
        base_source : str

        Returns
        -------
        list[SyntheticDiff]
            Same list with validity fields populated on each diff.
        """
        for diff in diffs:
            self.check(diff, base_source)
        return diffs
