"""
Context budget explorer — generate context token allocation strategies.
"""

from meta.schemas import ContextAllocation


class ContextBudgetExplorer:
    """Generate context token allocation strategies."""

    def generate_allocations(self, total_tokens: int = 8000) -> list:
        return [
            ContextAllocation(
                allocation_id="code_heavy",
                code_fraction=0.60, history_fraction=0.20,
                diagnostics_fraction=0.10, constraints_fraction=0.10,
            ),
            ContextAllocation(
                allocation_id="history_heavy",
                code_fraction=0.30, history_fraction=0.40,
                diagnostics_fraction=0.15, constraints_fraction=0.15,
            ),
            ContextAllocation(
                allocation_id="diagnostics_heavy",
                code_fraction=0.30, history_fraction=0.20,
                diagnostics_fraction=0.35, constraints_fraction=0.15,
            ),
            ContextAllocation(
                allocation_id="constraints_heavy",
                code_fraction=0.30, history_fraction=0.15,
                diagnostics_fraction=0.15, constraints_fraction=0.40,
            ),
            ContextAllocation(
                allocation_id="balanced",
                code_fraction=0.30, history_fraction=0.25,
                diagnostics_fraction=0.25, constraints_fraction=0.20,
            ),
            ContextAllocation(
                allocation_id="dynamic_early",
                code_fraction=0.45, history_fraction=0.25,
                diagnostics_fraction=0.15, constraints_fraction=0.15,
                is_dynamic=True,
                dynamic_rule="code_heavy for first 50 iterations, then history_heavy",
            ),
        ]
