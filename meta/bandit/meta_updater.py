"""
Posterior updater for the meta-bandit.

Uses three-zone scoring based on how the experiment IR compares
to the baseline distribution:
  - Better than baseline_mean + 1 std  -> success  (alpha += 1)
  - Worse  than baseline_mean - 1 std  -> failure  (beta  += 1)
  - Within +/- 1 std                   -> inconclusive (alpha += 0.3, beta += 0.3)
"""

import time

from meta.schemas import MetaBanditState, MetaExperimentResult, DimensionState


class MetaPosteriorUpdater:
    """Update meta-bandit posteriors from experiment results."""

    def update(self, state: MetaBanditState,
               experiment_result: MetaExperimentResult,
               baseline_ir,
               baseline_std: float = None) -> MetaBanditState:
        """Update posteriors for all dimensions changed in the experiment.

        Args:
            state: Current meta-bandit state (modified in place and returned).
            experiment_result: Result of the meta-experiment.
            baseline_ir: Baseline mean IR (float, dict, or AggregateIR).
            baseline_std: Baseline standard deviation of IR (overrides if given).

        Returns:
            Updated MetaBanditState.
        """
        experiment_ir = experiment_result.improvement_rate

        # Extract mean/std from baseline_ir
        if hasattr(baseline_ir, "mean_ir"):
            bl_mean = baseline_ir.mean_ir
            bl_std = baseline_ir.std_ir if baseline_std is None else baseline_std
        elif isinstance(baseline_ir, dict):
            bl_mean = baseline_ir.get("mean_ir", 0.0)
            bl_std = baseline_ir.get("std_ir", 0.0) if baseline_std is None else baseline_std
        else:
            bl_mean = float(baseline_ir)
            bl_std = baseline_std if baseline_std is not None else 0.0

        # Classify result into three zones
        zone = self._classify(experiment_ir, bl_mean, bl_std)
        experiment_result.compared_to_baseline = zone

        # Determine posterior deltas based on zone
        if zone == "better":
            d_alpha, d_beta = 1.0, 0.0
        elif zone == "worse":
            d_alpha, d_beta = 0.0, 1.0
        else:  # inconclusive
            d_alpha, d_beta = 0.3, 0.3

        # Apply to every dimension that was part of this experiment
        for diff in experiment_result.config_diff:
            param_id = diff.get("param_id", "") if isinstance(diff, dict) else getattr(diff, "param_id", "")
            new_value = diff.get("new_value") if isinstance(diff, dict) else getattr(diff, "new_value", None)

            if not param_id:
                continue

            if param_id not in state.dimensions:
                state.dimensions[param_id] = DimensionState(param_id=param_id)

            dim = state.dimensions[param_id]
            var_key = str(new_value)

            posterior = dim.variant_posteriors.get(var_key, {"alpha": 1.0, "beta": 1.0})
            posterior["alpha"] = posterior.get("alpha", 1.0) + d_alpha
            posterior["beta"] = posterior.get("beta", 1.0) + d_beta
            dim.variant_posteriors[var_key] = posterior

        state.metadata["last_updated"] = time.time()

        return state

    def _classify(self, experiment_ir: float, baseline_mean: float,
                  baseline_std: float) -> str:
        """Classify experiment result into three zones."""
        if baseline_std <= 0:
            # Without variance info, only clear wins/losses count
            if experiment_ir > baseline_mean:
                return "better"
            elif experiment_ir < baseline_mean:
                return "worse"
            return "inconclusive"

        upper = baseline_mean + baseline_std
        lower = baseline_mean - baseline_std

        if experiment_ir > upper:
            return "better"
        elif experiment_ir <= lower:
            return "worse"
        return "inconclusive"
