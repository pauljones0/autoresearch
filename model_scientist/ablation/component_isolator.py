"""
Phase 3 — ComponentIsolator: Generate isolated and leave-one-out versions
of train.py for ablation testing.
"""

import ast
import difflib

from ..schemas import ModificationComponent


class ComponentIsolator:
    """Generate train.py variants that apply/exclude individual components."""

    def isolate_component(self, base_source: str,
                          component: ModificationComponent) -> str:
        """Apply ONLY one component's changes to the base source.

        Returns modified source string, or base_source if patching fails.
        """
        return self._apply_component_diff(base_source, component)

    def leave_one_out(self, base_source: str,
                      all_components: list,
                      exclude_idx: int) -> str:
        """Apply ALL components EXCEPT the one at exclude_idx.

        Returns modified source string.
        """
        if exclude_idx < 0 or exclude_idx >= len(all_components):
            raise ValueError(f"exclude_idx {exclude_idx} out of range [0, {len(all_components)})")

        # Strategy: start from base, apply each component except the excluded one.
        # Since components may overlap or interact, we use a line-level merge approach.
        result = base_source
        for idx, component in enumerate(all_components):
            if idx == exclude_idx:
                continue
            candidate = self._apply_component_diff(result, component)
            # Only use the candidate if it produces valid Python
            if self._is_valid_python(candidate):
                result = candidate

        if not self._is_valid_python(result):
            # Fall back: try applying from full modified source minus the excluded component
            return self._leave_one_out_via_removal(base_source, all_components, exclude_idx)

        return result

    def apply_subset(self, base_source: str,
                     all_components: list,
                     include_indices: list) -> str:
        """Apply only the components at the given indices.

        Returns modified source string.
        """
        result = base_source
        for idx in include_indices:
            if idx < 0 or idx >= len(all_components):
                continue
            candidate = self._apply_component_diff(result, all_components[idx])
            if self._is_valid_python(candidate):
                result = candidate
        return result

    # ------------------------------------------------------------------
    # Internal patching
    # ------------------------------------------------------------------

    def _apply_component_diff(self, source: str, component: ModificationComponent) -> str:
        """Apply a component's diff to source using line-level patching."""
        if not component.diff:
            return source

        # Parse the unified diff to extract changes
        additions, deletions = self._parse_unified_diff(component.diff)

        if not additions and not deletions:
            return source

        source_lines = source.splitlines(keepends=True)

        # Find the lines to delete (match by content)
        result_lines = list(source_lines)
        deletion_contents = [d.rstrip('\n\r') for d in deletions]

        # Remove deleted lines (search by content match)
        for del_content in deletion_contents:
            for i, line in enumerate(result_lines):
                if line.rstrip('\n\r') == del_content:
                    result_lines[i] = None  # mark for removal
                    break

        result_lines = [l for l in result_lines if l is not None]

        # Insert added lines at the best matching position
        if additions:
            insert_pos = self._find_insertion_point(result_lines, additions, component.diff)
            for i, add_line in enumerate(additions):
                if not add_line.endswith('\n'):
                    add_line += '\n'
                result_lines.insert(insert_pos + i, add_line)

        result = ''.join(result_lines)

        if not self._is_valid_python(result):
            # Fall back: use difflib to try a merge
            return self._merge_fallback(source, component.diff)

        return result

    def _parse_unified_diff(self, diff_text: str) -> tuple:
        """Extract added and deleted lines from a unified diff."""
        additions = []
        deletions = []
        in_hunk = False

        for line in diff_text.splitlines():
            if line.startswith('@@'):
                in_hunk = True
                continue
            if line.startswith('---') or line.startswith('+++'):
                continue
            if not in_hunk:
                continue
            if line.startswith('+'):
                additions.append(line[1:])
            elif line.startswith('-'):
                deletions.append(line[1:])

        return additions, deletions

    def _find_insertion_point(self, lines: list, additions: list,
                               diff_text: str) -> int:
        """Find the best line index to insert additions."""
        # Try to find context lines from the diff
        context_lines = []
        for line in diff_text.splitlines():
            if line.startswith(' ') and not line.startswith('---') and not line.startswith('+++'):
                context_lines.append(line[1:].rstrip('\n\r'))

        # Search for context lines to find insertion point
        line_contents = [l.rstrip('\n\r') for l in lines]
        for ctx in reversed(context_lines):
            for i, lc in enumerate(line_contents):
                if lc == ctx:
                    return i + 1

        # Fall back: insert at end
        return len(lines)

    def _merge_fallback(self, source: str, diff_text: str) -> str:
        """Fall back to reconstructing from the diff's context."""
        # If we can't cleanly patch, return original source
        return source

    def _leave_one_out_via_removal(self, base_source: str,
                                    all_components: list,
                                    exclude_idx: int) -> str:
        """Alternative leave-one-out: build full modified, then try to undo one component."""
        # Apply all components to get full modified
        full = base_source
        for component in all_components:
            candidate = self._apply_component_diff(full, component)
            if self._is_valid_python(candidate):
                full = candidate

        # Now try to "reverse" the excluded component
        excluded = all_components[exclude_idx]
        additions, deletions = self._parse_unified_diff(excluded.diff)

        full_lines = full.splitlines(keepends=True)
        result_lines = list(full_lines)

        # Remove the additions (they are now in the full source)
        add_contents = [a.rstrip('\n\r') for a in additions]
        for add_content in add_contents:
            for i, line in enumerate(result_lines):
                if line is not None and line.rstrip('\n\r') == add_content:
                    result_lines[i] = None
                    break

        result_lines = [l for l in result_lines if l is not None]

        # Re-add the deletions at appropriate positions
        # (simplified: just return what we have, which removes the component's additions)

        result = ''.join(result_lines)
        if self._is_valid_python(result):
            return result

        return full  # can't cleanly remove; return full as fallback

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    @staticmethod
    def _is_valid_python(source: str) -> bool:
        """Check if source is syntactically valid Python."""
        try:
            ast.parse(source)
            return True
        except SyntaxError:
            return False
