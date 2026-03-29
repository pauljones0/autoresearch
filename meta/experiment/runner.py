"""
Meta-experiment runner.

Samples a config from the meta-bandit, applies it via bridges,
runs K inner-loop iterations, computes improvement rate, and
restores the best config.
"""

import random
import time
import uuid

from meta.schemas import (
    MetaBanditState, MetaContext, MetaExperimentResult, ParamDiff,
)
from meta.bandit.meta_bandit import MetaBandit


class MetaExperimentRunner:
    """Execute a single meta-experiment."""

    def __init__(self):
        self.bandit = MetaBandit()

    def run_experiment(self, meta_state: MetaBanditState,
                       context: MetaContext,
                       experiment_length: int = 50,
                       rng: random.Random = None) -> MetaExperimentResult:
        """Run one meta-experiment cycle.

        1. Sample a new config from the meta-bandit.
        2. Record the diff from current config.
        3. Apply the config via bridges.
        4. Run experiment_length inner-loop iterations.
        5. Compute improvement rate from raw deltas.
        6. Restore the best known config.

        Args:
            meta_state: Current meta-bandit state.
            context: Execution context with pipeline references.
            experiment_length: Number of inner iterations to run.
            rng: Random number generator for reproducibility.

        Returns:
            MetaExperimentResult with computed metrics.
        """
        if rng is None:
            rng = random.Random()

        experiment_id = f"meta_exp_{uuid.uuid4().hex[:8]}"

        # 1. Sample config from meta-bandit
        sampled_config = self.bandit.select(meta_state, rng)

        # 2. Build config diff
        config_diff = []
        for param_id, new_value in sampled_config.items():
            old_value = meta_state.current_config.get(param_id)
            if old_value != new_value:
                config_diff.append(ParamDiff(
                    param_id=param_id,
                    old_value=old_value,
                    new_value=new_value,
                ).to_dict())

        # 3. Apply config via bridges (if available)
        self._apply_config(sampled_config, context)

        # 4. Run inner-loop iterations and collect deltas
        raw_deltas = self._run_inner_loop(context, experiment_length)

        # 5. Compute improvement rate (negative delta = improvement in val_bpb)
        n_improved = sum(1 for d in raw_deltas if d < 0)
        improvement_rate = n_improved / len(raw_deltas) if raw_deltas else 0.0
        n_accepted = sum(1 for d in raw_deltas if d <= 0)
        acceptance_rate = n_accepted / len(raw_deltas) if raw_deltas else 0.0

        # 6. Restore best config
        if meta_state.best_config:
            self._apply_config(meta_state.best_config, context)

        return MetaExperimentResult(
            experiment_id=experiment_id,
            config_diff=config_diff,
            n_iterations=experiment_length,
            improvement_rate=improvement_rate,
            acceptance_rate=acceptance_rate,
            raw_deltas=raw_deltas,
            timestamp=time.time(),
        )

    def _apply_config(self, config: dict, context: MetaContext) -> None:
        """Apply a config dict via bridges in the context."""
        # Bridges are optional — in standalone mode this is a no-op.
        # When integrated, each bridge's apply() method is called.
        bridges = []
        for attr in ("bandit_pipeline", "model_scientist_pipeline",
                      "surrogate_triage_pipeline", "gpu_kernel_pipeline"):
            bridge = getattr(context, attr, None)
            if bridge is not None and hasattr(bridge, "apply"):
                bridges.append(bridge)

        for bridge in bridges:
            bridge.apply(config)

    def _run_inner_loop(self, context: MetaContext,
                        n_iterations: int) -> list:
        """Run inner-loop iterations and return raw deltas.

        In production this delegates to the actual pipeline.
        Returns a list of floats (val_bpb deltas per iteration).
        """
        # Stub: when no real pipeline is connected, return empty.
        # Real integration calls context pipelines' step() methods.
        runner = None
        for attr in ("bandit_pipeline", "model_scientist_pipeline"):
            candidate = getattr(context, attr, None)
            if candidate is not None and hasattr(candidate, "run_iterations"):
                runner = candidate
                break

        if runner is not None:
            return runner.run_iterations(n_iterations)

        return []
