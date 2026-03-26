"""
A/B evaluation of prompt variants.

Installs a variant prompt, runs iterations across multiple seeds,
extracts per-arm improvement rate, then restores the default prompt.
"""

import time

from meta.schemas import PromptVariant, PromptEvalResult


class PromptABEvaluator:
    """Evaluate prompt variants via controlled A/B experiments."""

    def evaluate(self, arm_id: str, variant: PromptVariant,
                 experiment_length: int = 50,
                 n_seeds: int = 2,
                 pipeline=None) -> PromptEvalResult:
        """Evaluate a single prompt variant.

        1. Save the current default prompt template.
        2. Install the variant template.
        3. For each seed, run experiment_length iterations.
        4. Compute per-arm IR and success rate.
        5. Restore the default template.

        Args:
            arm_id: The bandit arm being evaluated.
            variant: The prompt variant to test.
            experiment_length: Iterations per seed.
            n_seeds: Number of random seeds to average over.
            pipeline: Optional pipeline with prompt installation support.

        Returns:
            PromptEvalResult with per-seed and aggregate metrics.
        """
        default_template = None
        if pipeline is not None and hasattr(pipeline, "get_prompt_template"):
            default_template = pipeline.get_prompt_template(arm_id)

        per_seed_ir = []
        total_selections = 0
        total_successes = 0

        try:
            # Install variant
            if pipeline is not None and hasattr(pipeline, "set_prompt_template"):
                pipeline.set_prompt_template(arm_id, variant.template_text)

            for seed in range(n_seeds):
                seed_result = self._run_seed(
                    arm_id, experiment_length, seed, pipeline)
                per_seed_ir.append(seed_result["ir"])
                total_selections += seed_result["selections"]
                total_successes += seed_result["successes"]

        finally:
            # Always restore default
            if (default_template is not None and pipeline is not None
                    and hasattr(pipeline, "set_prompt_template")):
                pipeline.set_prompt_template(arm_id, default_template)

        mean_ir = sum(per_seed_ir) / len(per_seed_ir) if per_seed_ir else 0.0
        variance = (sum((x - mean_ir) ** 2 for x in per_seed_ir)
                     / len(per_seed_ir)) if per_seed_ir else 0.0
        std_ir = variance ** 0.5

        arm_success_rate = (total_successes / total_selections
                            if total_selections > 0 else 0.0)

        return PromptEvalResult(
            variant_id=variant.variant_id,
            arm_id=arm_id,
            per_seed_ir=per_seed_ir,
            mean_ir=mean_ir,
            std_ir=std_ir,
            n_arm_selections=total_selections,
            n_arm_successes=total_successes,
            arm_success_rate=arm_success_rate,
        )

    def _run_seed(self, arm_id: str, n_iterations: int,
                  seed: int, pipeline) -> dict:
        """Run iterations for a single seed and extract arm-specific metrics."""
        if pipeline is not None and hasattr(pipeline, "run_iterations_with_seed"):
            result = pipeline.run_iterations_with_seed(
                n_iterations, seed, track_arm=arm_id)
            return {
                "ir": result.get("improvement_rate", 0.0),
                "selections": result.get("arm_selections", 0),
                "successes": result.get("arm_successes", 0),
            }

        # No pipeline — return zeros
        return {"ir": 0.0, "selections": 0, "successes": 0}
