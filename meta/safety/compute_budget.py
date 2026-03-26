"""
Meta compute budget enforcer — enforce production/meta split.
"""


class MetaComputeBudgetEnforcer:
    """Enforce configurable budget split between production and meta."""

    def __init__(self, state_path: str = "meta_state.json",
                 budget_fraction: float = 0.2,
                 budget_cycle_length: int = 500):
        self.state_path = state_path
        self.budget_fraction = budget_fraction
        self.budget_cycle_length = budget_cycle_length
        self._total_iterations = 0
        self._meta_iterations = 0

    def can_run_meta_experiment(self, total_iterations_run: int,
                                meta_iterations_run: int) -> bool:
        self._total_iterations = total_iterations_run
        self._meta_iterations = meta_iterations_run
        if total_iterations_run == 0:
            return True
        current_fraction = meta_iterations_run / max(1, total_iterations_run)
        return current_fraction < self.budget_fraction

    def get_budget_status(self) -> dict:
        total = max(1, self._total_iterations)
        current_fraction = self._meta_iterations / total
        max_meta = int(self.budget_cycle_length * self.budget_fraction)
        meta_in_cycle = min(self._meta_iterations, max_meta)
        return {
            "total_iterations": self._total_iterations,
            "meta_iterations": self._meta_iterations,
            "budget_fraction": self.budget_fraction,
            "current_fraction": current_fraction,
            "remaining_meta_iterations": max(0, max_meta - meta_in_cycle),
            "exhausted": current_fraction >= self.budget_fraction,
        }
