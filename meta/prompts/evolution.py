"""
Multi-generation evolutionary prompt optimization.

Runs multiple generations: generate variants, evaluate, select best,
generate from best, repeat.
"""

from meta.schemas import PromptVariant, PromptEvalResult
from meta.prompts.variant_generator import PromptVariantGenerator
from meta.prompts.evaluator import PromptABEvaluator


class PromptEvolutionController:
    """Evolutionary optimization of prompt templates."""

    def __init__(self):
        self.generator = PromptVariantGenerator()
        self.evaluator = PromptABEvaluator()

    def evolve(self, arm_id: str,
               current_template: str = "",
               journal_context: str = "",
               n_generations: int = 3,
               n_variants_per_gen: int = 5,
               experiment_length: int = 50,
               n_seeds: int = 2,
               pipeline=None) -> PromptVariant:
        """Run multi-generation evolutionary prompt optimization.

        Each generation:
        1. Generate n_variants from the current best template.
        2. Evaluate each variant.
        3. Select the best-performing variant as the new parent.

        Args:
            arm_id: Bandit arm to optimize prompts for.
            current_template: Starting template text.
            journal_context: Journal context for variant generation.
            n_generations: Number of evolutionary generations.
            n_variants_per_gen: Variants to generate per generation.
            experiment_length: Iterations per evaluation.
            n_seeds: Seeds per evaluation.
            pipeline: Optional pipeline for real evaluation.

        Returns:
            The best PromptVariant found across all generations.
        """
        best_template = current_template
        best_variant = None
        best_ir = float("-inf")

        for gen in range(n_generations):
            variants = self.generator.generate_variants(
                arm_id=arm_id,
                current_template=best_template,
                journal_context=journal_context,
                n_variants=n_variants_per_gen,
            )

            for variant in variants:
                result = self.evaluator.evaluate(
                    arm_id=arm_id,
                    variant=variant,
                    experiment_length=experiment_length,
                    n_seeds=n_seeds,
                    pipeline=pipeline,
                )

                if result.mean_ir > best_ir:
                    best_ir = result.mean_ir
                    best_variant = variant
                    best_template = variant.template_text

        # If no variant beat the original, return a variant wrapping
        # the original template
        if best_variant is None:
            best_variant = PromptVariant(
                variant_id=f"{arm_id}_original",
                arm_id=arm_id,
                template_text=current_template,
                variation_dimension="none",
                variation_description="Original template (no improvement found)",
            )

        return best_variant
