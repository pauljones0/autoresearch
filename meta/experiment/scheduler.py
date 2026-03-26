"""
Meta-experiment scheduling.

Decides whether to run a meta-experiment based on the current
regime and budget constraints:
  - baseline  -> never run (False)
  - maintenance -> 5% probability
  - active    -> 20% probability
"""

import random

from meta.schemas import MetaBanditState


class MetaExperimentScheduler:
    """Regime-aware scheduling of meta-experiments."""

    # Probability of running a meta-experiment per inner iteration
    REGIME_RATES = {
        "baseline": 0.0,
        "maintenance": 0.05,
        "active": 0.20,
    }

    def should_run_meta_experiment(self, meta_state: MetaBanditState,
                                   inner_iteration: int,
                                   budget_enforcer=None,
                                   rng: random.Random = None) -> bool:
        """Decide whether to run a meta-experiment now.

        Args:
            meta_state: Current meta-bandit state (contains regime).
            inner_iteration: Current inner-loop iteration number.
            budget_enforcer: Optional budget enforcer with
                ``can_afford(cost) -> bool`` method.
            rng: Random number generator.

        Returns:
            True if a meta-experiment should run.
        """
        if rng is None:
            rng = random.Random()

        regime = meta_state.meta_regime
        rate = self.REGIME_RATES.get(regime, 0.0)

        if rate <= 0.0:
            return False

        # Budget check
        if budget_enforcer is not None:
            if not budget_enforcer.can_run_meta_experiment(
                    meta_state.total_inner_iterations,
                    meta_state.total_meta_experiments):
                return False

        # Stochastic decision
        return rng.random() < rate
