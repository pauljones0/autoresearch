"""
Prompt variant generation across five systematic dimensions.

Dimensions:
  1. Instruction specificity  — vague vs. precise directives
  2. History format            — how past results are presented
  3. Code context              — amount and style of code snippets
  4. Constraint emphasis       — how strongly constraints are stated
  5. Reasoning guidance        — depth of chain-of-thought prompting
"""

import hashlib
import uuid

from meta.schemas import PromptVariant


VARIATION_DIMENSIONS = [
    "instruction_specificity",
    "history_format",
    "code_context",
    "constraint_emphasis",
    "reasoning_guidance",
]


class PromptVariantGenerator:
    """Generate prompt template variants for a given arm."""

    def generate_variants(self, arm_id: str,
                          current_template: str,
                          journal_context: str = "",
                          n_variants: int = 5) -> list:
        """Generate n_variants prompt variants, one per dimension.

        Args:
            arm_id: The bandit arm to generate variants for.
            current_template: Current prompt template text.
            journal_context: Recent journal entries for context.
            n_variants: Number of variants to generate (one per dimension).

        Returns:
            List of PromptVariant objects.
        """
        parent_hash = hashlib.sha256(current_template.encode()).hexdigest()[:12]
        variants = []

        for i, dimension in enumerate(VARIATION_DIMENSIONS[:n_variants]):
            template_text = self._vary_template(
                current_template, dimension, journal_context)
            variant = PromptVariant(
                variant_id=f"{arm_id}_{dimension}_{uuid.uuid4().hex[:6]}",
                arm_id=arm_id,
                template_text=template_text,
                variation_dimension=dimension,
                variation_description=self._describe_variation(dimension),
                parent_template_hash=parent_hash,
            )
            variants.append(variant)

        return variants

    def _vary_template(self, template: str, dimension: str,
                       journal_context: str) -> str:
        """Apply a systematic variation along the given dimension."""
        if dimension == "instruction_specificity":
            return self._vary_specificity(template)
        elif dimension == "history_format":
            return self._vary_history(template, journal_context)
        elif dimension == "code_context":
            return self._vary_code_context(template)
        elif dimension == "constraint_emphasis":
            return self._vary_constraints(template)
        elif dimension == "reasoning_guidance":
            return self._vary_reasoning(template)
        return template

    def _vary_specificity(self, template: str) -> str:
        """More precise instructions with explicit step-by-step directives."""
        prefix = (
            "You MUST follow these exact steps:\n"
            "1. Analyze the current code and identify the specific bottleneck.\n"
            "2. Propose exactly ONE targeted modification.\n"
            "3. Explain why this change will improve performance.\n"
            "4. Write the complete modified code.\n\n"
        )
        return prefix + template

    def _vary_history(self, template: str, journal_context: str) -> str:
        """Restructure history presentation as a ranked summary."""
        history_block = (
            "\n## Recent Results (ranked by impact)\n"
            "Focus on the patterns in these results rather than "
            "individual values. Identify what types of changes "
            "have been most effective.\n"
        )
        if journal_context:
            history_block += f"\n{journal_context}\n"
        return template + history_block

    def _vary_code_context(self, template: str) -> str:
        """Emphasize code analysis with explicit structure markers."""
        code_block = (
            "\n## Code Analysis Protocol\n"
            "Before proposing changes:\n"
            "- Identify the HOT PATH in the current code\n"
            "- Note which operations dominate compute time\n"
            "- Consider numerical stability of all operations\n"
            "- Check for redundant computations that can be eliminated\n"
        )
        return template + code_block

    def _vary_constraints(self, template: str) -> str:
        """Strengthen constraint language."""
        constraint_block = (
            "\n## CRITICAL CONSTRAINTS (violations = automatic rejection)\n"
            "- Do NOT change the model architecture dimensions\n"
            "- Do NOT modify the evaluation metric or dataset\n"
            "- Do NOT introduce external dependencies\n"
            "- Changes MUST be backwards-compatible\n"
            "- Total parameter count MUST remain within 5% of original\n"
        )
        return template + constraint_block

    def _vary_reasoning(self, template: str) -> str:
        """Add chain-of-thought scaffolding."""
        reasoning_block = (
            "\n## Reasoning Framework\n"
            "Think through your approach step by step:\n"
            "1. What is the current performance bottleneck?\n"
            "2. What first-principles insight suggests a fix?\n"
            "3. What is the expected magnitude of improvement?\n"
            "4. What could go wrong with this change?\n"
            "5. How would you verify the improvement is real?\n"
            "Show your reasoning before writing code.\n"
        )
        return template + reasoning_block

    def _describe_variation(self, dimension: str) -> str:
        """Return a human-readable description of the variation."""
        descriptions = {
            "instruction_specificity": "More precise step-by-step directives",
            "history_format": "Restructured history as ranked impact summary",
            "code_context": "Added explicit code analysis protocol",
            "constraint_emphasis": "Strengthened constraint language",
            "reasoning_guidance": "Added chain-of-thought reasoning scaffolding",
        }
        return descriptions.get(dimension, dimension)
