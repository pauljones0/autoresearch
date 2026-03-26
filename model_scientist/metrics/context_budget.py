"""
Phase 4: ContextBudgetManager — allocates diagnostic context tokens
proportional to each metric's predictive power.
"""

from ..schemas import DiagnosticsReport, MetricDefinition
from .registry import MetricRegistry
from .implementer import MetricImplementer


class ContextBudgetManager:
    """Formats diagnostic context for the research agent, weighted by metric value."""

    def __init__(self):
        self._implementer = MetricImplementer()

    def allocate(
        self,
        registry: MetricRegistry,
        diagnostics: DiagnosticsReport,
        max_tokens: int = 2000,
    ) -> str:
        """Build a formatted context string within the token budget.

        Active metrics are allocated tokens proportional to |correlation|.
        Candidate metrics get a minimal one-line summary.
        Retired metrics are omitted.

        Args:
            registry: The MetricRegistry.
            diagnostics: Current DiagnosticsReport.
            max_tokens: Approximate token budget (chars / 4 as proxy).

        Returns:
            Formatted string for inclusion in agent prompts.
        """
        max_chars = max_tokens * 4  # rough chars-per-token estimate

        active = registry.get_active()
        candidates = registry.get_candidates()

        if not active and not candidates:
            return "[No diagnostic metrics available]"

        # Compute correlation weights for active metrics
        weights: list[tuple[MetricDefinition, float]] = []
        total_weight = 0.0
        for m in active:
            w = max(abs(m.correlation_with_success), 0.05)  # floor so every active metric gets some space
            weights.append((m, w))
            total_weight += w

        parts: list[str] = []
        chars_used = 0

        # --- Active metrics: proportional allocation ---
        if weights and total_weight > 0:
            parts.append("=== Diagnostic Metrics (active) ===")
            chars_used += 40

            # Reserve 20% for candidates
            active_budget = int(max_chars * 0.8) if candidates else max_chars

            for metric, w in sorted(weights, key=lambda x: -x[1]):
                frac = w / total_weight
                token_budget = max(int(active_budget * frac), 80)
                line = self.format_metric(metric, diagnostics, token_budget)
                if chars_used + len(line) + 2 > max_chars:
                    break
                parts.append(line)
                chars_used += len(line) + 1

        # --- Candidate metrics: one-line summaries ---
        if candidates:
            parts.append("")
            parts.append("=== Candidate Metrics ===")
            chars_used += 30
            for m in candidates:
                if chars_used + 100 > max_chars:
                    parts.append(f"  ... and {len(candidates) - len([p for p in parts if p.startswith('  ')])} more candidates")
                    break
                try:
                    val = self._implementer.compute_metric(m, diagnostics)
                    line = f"  {m.name}: {val:.4f} (r={m.correlation_with_success:.2f})"
                except (ValueError, Exception):
                    line = f"  {m.name}: [error computing]"
                parts.append(line)
                chars_used += len(line) + 1

        return "\n".join(parts)

    def format_metric(
        self,
        metric: MetricDefinition,
        diagnostics: DiagnosticsReport,
        token_budget: int,
    ) -> str:
        """Format a single metric's display, scaled to the token budget.

        Higher budgets get description + value + rationale.
        Lower budgets get just name + value.
        """
        try:
            value = self._implementer.compute_metric(metric, diagnostics)
            val_str = f"{value:.4f}"
        except (ValueError, Exception):
            val_str = "[error]"

        char_budget = token_budget * 4

        # Minimal: name + value
        short = f"  {metric.name}: {val_str}"
        if char_budget < 120:
            return short

        # Medium: add correlation
        medium = f"  {metric.name}: {val_str}  (r={metric.correlation_with_success:.2f})"
        if char_budget < 250:
            return medium

        # Full: add description
        desc = metric.description[:char_budget - len(medium) - 10]
        return f"  {metric.name}: {val_str}  (r={metric.correlation_with_success:.2f})\n    {desc}"
