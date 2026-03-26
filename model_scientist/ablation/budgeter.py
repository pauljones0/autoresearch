"""
Phase 3 — AblationBudgeter: Decide whether to run ablation and how deep,
based on component count, improvement magnitude, and compute budget.
"""


class AblationBudgeter:
    """Control ablation compute spending."""

    def __init__(self,
                 min_improvement_threshold: float = 0.005,
                 max_top_k: int = 3,
                 max_budget_fraction: float = 0.15):
        """
        Args:
            min_improvement_threshold: Minimum val_bpb improvement to warrant ablation.
            max_top_k: Max number of modifications to ablate per cycle.
            max_budget_fraction: Max fraction of total training compute for ablation.
        """
        self.min_improvement_threshold = min_improvement_threshold
        self.max_top_k = max_top_k
        self.max_budget_fraction = max_budget_fraction
        self._cumulative_compute = 0.0

    def should_ablate(self, n_components: int, improvement: float,
                      compute_used: float, compute_budget: float) -> tuple:
        """Decide whether to run ablation.

        Args:
            n_components: Number of components in the modification.
            improvement: val_bpb improvement (positive = better).
            compute_used: Compute already used for ablations this cycle (seconds).
            compute_budget: Total compute budget for ablations (seconds). 0 = unlimited.

        Returns:
            (should_run: bool, reason: str)
        """
        # Single-component modifications have nothing to ablate
        if n_components <= 1:
            return False, "Single component — nothing to ablate"

        # Improvement below threshold
        if improvement < self.min_improvement_threshold:
            return False, (f"Improvement {improvement:.6f} below threshold "
                          f"{self.min_improvement_threshold:.6f}")

        # Check compute budget
        if compute_budget > 0 and compute_used >= compute_budget:
            return False, (f"Compute budget exhausted: {compute_used:.1f}s "
                          f"used of {compute_budget:.1f}s budget")

        return True, f"Ablation warranted: {n_components} components, improvement {improvement:.6f}"

    def max_variants(self, compute_remaining: float,
                     time_per_variant: float) -> int:
        """Calculate maximum number of variants we can run within remaining budget.

        Args:
            compute_remaining: Remaining compute budget in seconds.
            time_per_variant: Estimated time per variant run in seconds.

        Returns:
            Maximum number of variants (0 if budget exhausted).
        """
        if time_per_variant <= 0:
            return 0
        if compute_remaining <= 0:
            return 0
        return int(compute_remaining / time_per_variant)

    def select_top_k(self, candidates: list, k: int = None) -> list:
        """Select top-K modifications to ablate from a list of (id, improvement) tuples.

        Args:
            candidates: List of (modification_id, improvement) tuples.
            k: Number to select (defaults to self.max_top_k).

        Returns:
            Top-K candidates sorted by improvement (descending).
        """
        if k is None:
            k = self.max_top_k
        sorted_candidates = sorted(candidates, key=lambda x: x[1], reverse=True)
        return sorted_candidates[:k]

    def compute_budget_for_cycle(self, total_training_time: float) -> float:
        """Calculate ablation compute budget for a cycle.

        Args:
            total_training_time: Total training compute time in seconds.

        Returns:
            Ablation budget in seconds.
        """
        return total_training_time * self.max_budget_fraction

    def record_compute(self, seconds: float):
        """Record compute time used for ablation."""
        self._cumulative_compute += seconds

    @property
    def cumulative_compute(self) -> float:
        """Total compute used for ablations across all cycles."""
        return self._cumulative_compute

    def reset_cycle(self):
        """Reset per-cycle tracking (cumulative compute persists)."""
        pass
