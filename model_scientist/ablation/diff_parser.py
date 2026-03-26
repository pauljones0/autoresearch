"""
Phase 3 — DiffParser: Parse unified diffs and identify semantically independent
modification components using AST analysis and heuristic grouping.
"""

import ast
import difflib
import re
from dataclasses import dataclass, field

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from model_scientist.schemas import ModificationComponent


# Section markers in train.py used for category classification
_SECTION_MARKERS = {
    "GPT Model": "architecture",
    "Optimizer": "optimizer",
    "Hyperparameters": "hyperparameter",
    "Setup": "setup",
    "Training loop": "training_loop",
}

# Patterns for fine-grained category detection
_CATEGORY_PATTERNS = [
    (r'\bactivation\b|F\.relu|F\.gelu|F\.silu|\.square\(\)', "activation"),
    (r'\binit_weights\b|torch\.nn\.init\.|\.fill_\(|\.zeros_\(', "initialization"),
    (r'\boptimizer\b|AdamW|Muon|\.step\(\)|param_groups|weight_decay|lr\b', "optimizer"),
    (r'\bn_layer\b|n_head\b|n_embd\b|n_kv_head\b|head_dim|DEPTH|ASPECT_RATIO|HEAD_DIM', "hyperparameter"),
    (r'\bclass\s+\w+\(nn\.Module\)|def forward\(|self\.transformer|self\.lm_head', "architecture"),
    (r'\bfor micro_step\b|loss\.backward|optimizer\.step|train_loader|grad_accum', "training_loop"),
]


@dataclass
class _ChangedRegion:
    """A contiguous region of changed lines in the new file."""
    start_line: int  # 1-indexed in new file
    end_line: int    # inclusive
    old_lines: list = field(default_factory=list)
    new_lines: list = field(default_factory=list)


class DiffParser:
    """Parse modifications to train.py and decompose into independent components."""

    def parse(self, old_source: str, new_source: str) -> list:
        """Parse old and new source, return list of ModificationComponent."""
        regions = self._find_changed_regions(old_source, new_source)
        if not regions:
            return []

        # Try AST-level decomposition first
        components = self._ast_decompose(old_source, new_source, regions)

        # If AST decomposition yields nothing useful, fall back to heuristic
        if not components:
            components = self._heuristic_decompose(old_source, new_source, regions)

        return components

    def parse_from_diff(self, diff_text: str, base_source: str) -> list:
        """Parse from a unified diff string and base source."""
        new_source = self._apply_diff(diff_text, base_source)
        if new_source is None:
            return []
        return self.parse(base_source, new_source)

    # ------------------------------------------------------------------
    # Region detection via difflib
    # ------------------------------------------------------------------

    def _find_changed_regions(self, old_source: str, new_source: str) -> list:
        """Identify contiguous changed regions between old and new source."""
        old_lines = old_source.splitlines(keepends=True)
        new_lines = new_source.splitlines(keepends=True)
        matcher = difflib.SequenceMatcher(None, old_lines, new_lines)
        regions = []

        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == 'equal':
                continue
            region = _ChangedRegion(
                start_line=j1 + 1,
                end_line=max(j2, j1 + 1),
                old_lines=old_lines[i1:i2],
                new_lines=new_lines[j1:j2],
            )
            regions.append(region)

        return regions

    # ------------------------------------------------------------------
    # AST-level decomposition
    # ------------------------------------------------------------------

    def _ast_decompose(self, old_source: str, new_source: str, regions: list) -> list:
        """Decompose changes by mapping regions to AST nodes."""
        try:
            old_tree = ast.parse(old_source)
            new_tree = ast.parse(new_source)
        except SyntaxError:
            return []

        new_lines = new_source.splitlines()
        new_node_map = self._build_node_map(new_tree)

        # Map each region to the AST node(s) it touches
        node_to_regions = {}
        for region in regions:
            node_key = self._find_enclosing_node(region, new_node_map, new_lines)
            if node_key not in node_to_regions:
                node_to_regions[node_key] = []
            node_to_regions[node_key].append(region)

        if len(node_to_regions) <= 1:
            # All changes are in one node — try heuristic split
            return []

        components = []
        for idx, (node_key, node_regions) in enumerate(node_to_regions.items()):
            diff_text = self._regions_to_diff(node_regions, old_source, new_source)
            description = self._describe_change(node_key, node_regions, new_lines)
            category = self._classify_category(node_key, node_regions, new_lines)

            components.append(ModificationComponent(
                component_id=idx,
                description=description,
                diff=diff_text,
                category=category,
            ))

        return components

    def _build_node_map(self, tree: ast.AST) -> list:
        """Build list of (name, start_line, end_line) for top-level AST nodes."""
        nodes = []
        for node in ast.iter_child_nodes(tree):
            if not hasattr(node, 'lineno'):
                continue
            name = getattr(node, 'name', None)
            if name is None:
                if isinstance(node, ast.Assign):
                    targets = [t.id for t in node.targets if isinstance(t, ast.Name)]
                    name = ', '.join(targets) if targets else f"line_{node.lineno}"
                else:
                    name = f"line_{node.lineno}"
            end_line = getattr(node, 'end_lineno', node.lineno)
            nodes.append((name, node.lineno, end_line))
        return nodes

    def _find_enclosing_node(self, region: _ChangedRegion, node_map: list, lines: list) -> str:
        """Find the AST node that encloses a changed region."""
        for name, start, end in node_map:
            if region.start_line >= start and region.start_line <= end:
                return name
        # Check section by comment headers
        section = self._find_section(region.start_line, lines)
        return section or f"region_{region.start_line}"

    def _find_section(self, line_num: int, lines: list) -> str:
        """Find which section a line belongs to based on comment headers."""
        for i in range(min(line_num - 1, len(lines) - 1), -1, -1):
            line = lines[i]
            for marker, _category in _SECTION_MARKERS.items():
                if marker in line and line.strip().startswith('#'):
                    return marker
        return None

    # ------------------------------------------------------------------
    # Heuristic decomposition
    # ------------------------------------------------------------------

    def _heuristic_decompose(self, old_source: str, new_source: str,
                              regions: list) -> list:
        """Split changes using heuristic grouping at semantic boundaries."""
        new_lines = new_source.splitlines()

        # Group regions that are close together (within 3 lines)
        groups = []
        current_group = [regions[0]]
        for region in regions[1:]:
            prev = current_group[-1]
            if region.start_line - prev.end_line <= 3:
                current_group.append(region)
            else:
                groups.append(current_group)
                current_group = [region]
        groups.append(current_group)

        # Further split groups at semantic boundaries
        split_groups = []
        for group in groups:
            split_groups.extend(self._split_at_boundaries(group, new_lines))

        if len(split_groups) <= 1:
            # Cannot decompose further — return as single component
            diff_text = self._regions_to_diff(regions, old_source, new_source)
            category = self._classify_from_lines(
                [l for r in regions for l in r.new_lines], new_lines,
                regions[0].start_line)
            return [ModificationComponent(
                component_id=0,
                description=self._summarize_lines(regions, new_lines),
                diff=diff_text,
                category=category,
            )]

        components = []
        for idx, group in enumerate(split_groups):
            diff_text = self._regions_to_diff(group, old_source, new_source)
            description = self._summarize_lines(group, new_lines)
            category = self._classify_from_lines(
                [l for r in group for l in r.new_lines], new_lines,
                group[0].start_line)
            components.append(ModificationComponent(
                component_id=idx,
                description=description,
                diff=diff_text,
                category=category,
            ))

        return components

    def _split_at_boundaries(self, group: list, lines: list) -> list:
        """Split a group of regions at function/class/section boundaries."""
        if len(group) <= 1:
            return [group]

        result = []
        current = [group[0]]
        prev_section = self._find_section(group[0].start_line, lines)

        for region in group[1:]:
            section = self._find_section(region.start_line, lines)
            # Split if different section or if there's a blank line gap
            if section != prev_section:
                result.append(current)
                current = [region]
                prev_section = section
            else:
                current.append(region)

        result.append(current)
        return result

    # ------------------------------------------------------------------
    # Category classification
    # ------------------------------------------------------------------

    def _classify_category(self, node_key: str, regions: list, lines: list) -> str:
        """Classify a component's category from its AST node and content."""
        # Check node name against known patterns
        for marker, category in _SECTION_MARKERS.items():
            if marker.lower() in node_key.lower():
                return category

        # Check content
        content = '\n'.join(l for r in regions for l in r.new_lines)
        return self._match_category_patterns(content, lines, regions[0].start_line)

    def _classify_from_lines(self, changed_lines: list, all_lines: list,
                              start_line: int) -> str:
        """Classify category from changed line content."""
        content = '\n'.join(changed_lines)
        return self._match_category_patterns(content, all_lines, start_line)

    def _match_category_patterns(self, content: str, all_lines: list,
                                  start_line: int) -> str:
        """Match content against category patterns."""
        for pattern, category in _CATEGORY_PATTERNS:
            if re.search(pattern, content):
                return category

        # Fall back to section-based classification
        section = self._find_section(start_line, all_lines)
        if section:
            for marker, category in _SECTION_MARKERS.items():
                if marker == section:
                    return category

        return "other"

    # ------------------------------------------------------------------
    # Description generation
    # ------------------------------------------------------------------

    def _describe_change(self, node_key: str, regions: list, lines: list) -> str:
        """Generate a human-readable description of a change."""
        total_added = sum(len(r.new_lines) for r in regions)
        total_removed = sum(len(r.old_lines) for r in regions)

        if total_added > 0 and total_removed == 0:
            action = "Add"
        elif total_added == 0 and total_removed > 0:
            action = "Remove"
        else:
            action = "Modify"

        return f"{action} {node_key} (lines {regions[0].start_line}-{regions[-1].end_line})"

    def _summarize_lines(self, regions: list, lines: list) -> str:
        """Summarize a group of regions."""
        total_added = sum(len(r.new_lines) for r in regions)
        total_removed = sum(len(r.old_lines) for r in regions)

        if total_added > 0 and total_removed == 0:
            action = "Add"
        elif total_added == 0 and total_removed > 0:
            action = "Remove"
        else:
            action = "Modify"

        start = regions[0].start_line
        end = regions[-1].end_line
        return f"{action} lines {start}-{end} ({total_added} added, {total_removed} removed)"

    # ------------------------------------------------------------------
    # Diff generation and application
    # ------------------------------------------------------------------

    def _regions_to_diff(self, regions: list, old_source: str, new_source: str) -> str:
        """Generate a unified diff string for specific regions."""
        old_lines = old_source.splitlines(keepends=True)
        new_lines = new_source.splitlines(keepends=True)
        
        target_regions = {(r.start_line, r.end_line) for r in regions}
        synthetic_new_lines = []
        
        matcher = difflib.SequenceMatcher(None, old_lines, new_lines)
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == 'equal':
                synthetic_new_lines.extend(old_lines[i1:i2])
            else:
                region_key = (j1 + 1, max(j2, j1 + 1))
                if region_key in target_regions:
                    synthetic_new_lines.extend(new_lines[j1:j2])
                else:
                    synthetic_new_lines.extend(old_lines[i1:i2])
                    
        diff = difflib.unified_diff(old_lines, synthetic_new_lines, lineterm='')
        return ''.join(diff)

    def _apply_diff(self, diff_text: str, base_source: str) -> str:
        """Apply a unified diff to base source. Returns new source or None on failure."""
        base_lines = base_source.splitlines(keepends=True)

        # Parse hunks from unified diff
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

        # Apply hunks in reverse order to preserve line numbers
        result_lines = list(base_lines)
        for hunk in reversed(hunks):
            old_idx = hunk['old_start'] - 1
            remove_count = 0
            add_lines = []

            for line in hunk['lines']:
                if line.startswith('-'):
                    remove_count += 1
                elif line.startswith('+'):
                    add_lines.append(line[1:])
                elif line.startswith(' '):
                    # Context line — counts as both old and new
                    remove_count += 1
                    add_lines.append(line[1:])

            result_lines[old_idx:old_idx + remove_count] = add_lines

        result = ''.join(result_lines)

        # Validate
        try:
            ast.parse(result)
        except SyntaxError:
            return None

        return result
