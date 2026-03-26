"""
Parallel per-dimension Thompson Sampling for meta-parameter optimization.

Each tunable dimension maintains independent Beta posteriors for its variants.
Selection samples from all variants' posteriors and picks the highest.
"""

import math
import random

from meta.schemas import MetaBanditState, DimensionState


class MetaBandit:
    """Thompson Sampling bandit that selects independently per dimension."""

    def select(self, state: MetaBanditState, rng: random.Random = None) -> dict:
        """Return a full config by selecting the best variant per dimension.

        Args:
            state: Current meta-bandit state with all dimension posteriors.
            rng: Random number generator for reproducibility.

        Returns:
            Dict mapping param_id -> selected variant value.
        """
        if rng is None:
            rng = random.Random()

        config = {}
        for param_id, dim_state in state.dimensions.items():
            config[param_id] = self.select_single_dimension(dim_state, rng)
        return config

    def select_single_dimension(self, dim_state: DimensionState,
                                rng: random.Random) -> object:
        """Sample from each variant's Beta posterior, return the best.

        Args:
            dim_state: Per-dimension state with variants and posteriors.
            rng: Random number generator.

        Returns:
            The variant value with the highest Thompson sample.
        """
        if not dim_state.variants:
            return dim_state.current_best

        best_sample = -1.0
        best_variant = dim_state.variants[0] if dim_state.variants else None

        for variant in dim_state.variants:
            var_key = str(variant)
            posterior = dim_state.variant_posteriors.get(var_key, {})
            alpha = posterior.get("alpha", 1.0)
            beta = posterior.get("beta", 1.0)
            sample = _beta_sample(alpha, beta, rng)
            if sample > best_sample:
                best_sample = sample
                best_variant = variant

        return best_variant


def _beta_sample(alpha: float, beta: float, rng: random.Random) -> float:
    """Sample from Beta(alpha, beta) using the gamma-ratio method."""
    # random.gammavariate is available in stdlib
    x = rng.gammavariate(alpha, 1.0)
    y = rng.gammavariate(beta, 1.0)
    if x + y == 0:
        return 0.5
    return x / (x + y)
