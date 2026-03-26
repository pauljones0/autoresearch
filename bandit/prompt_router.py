"""
Category-aware prompt router for bandit arm dispatch.
"""

import os

from bandit.schemas import BanditState


TEMPLATE_DIR = os.path.join(os.path.dirname(__file__), "prompt_templates")


class CategoryPromptRouter:
    """Builds prompts by loading arm-specific templates and injecting dynamic context."""

    def build_prompt(self, arm_id: str, diagnostics_summary: str,
                     journal_context: str, base_source: str) -> str:
        """Load template and inject dynamic sections.

        Args:
            arm_id: The arm identifier (maps to prompt_templates/{arm_id}.txt).
            diagnostics_summary: Recent diagnostics as text.
            journal_context: Recent accepted journal entries as text.
            base_source: Current model source code.

        Returns:
            Assembled prompt string.
        """
        template = self._load_template(arm_id)

        # Inject dynamic sections
        prompt = template
        prompt = prompt.replace("{{recent_examples}}", journal_context or "(none)")
        prompt = prompt.replace("{{constraints}}", self._build_constraints(arm_id))
        prompt = prompt.replace("{{diagnostics}}", diagnostics_summary or "(no diagnostics available)")
        prompt = prompt.replace("{{source_code}}", base_source or "(source not provided)")

        return prompt

    def _load_template(self, arm_id: str) -> str:
        """Load template file for the given arm_id."""
        # Try exact match first, then strip common prefixes
        candidates = [
            arm_id,
            arm_id.replace("_paper", ""),
            arm_id.split("_")[0],
        ]
        for candidate in candidates:
            path = os.path.join(TEMPLATE_DIR, f"{candidate}.txt")
            if os.path.exists(path):
                with open(path, 'r') as f:
                    return f.read()

        # Fallback generic template
        return (
            f"You are an ML research assistant specializing in {arm_id} modifications.\n\n"
            "## Recent Accepted Modifications\n{{recent_examples}}\n\n"
            "## Constraints\n{{constraints}}\n\n"
            "## Current Diagnostics\n{{diagnostics}}\n\n"
            "## Current Source Code\n{{source_code}}\n\n"
            "Propose a single concrete modification. Be specific and actionable."
        )

    def _build_constraints(self, arm_id: str) -> str:
        """Build negative constraints to avoid repeating past failures."""
        # Constraints are injected by the caller via the template placeholder;
        # this method provides arm-specific static constraints.
        constraints = [
            "- Do NOT propose changes that have been previously rejected.",
            "- Do NOT combine multiple unrelated changes in one proposal.",
            "- Ensure the modification is testable and measurable.",
        ]
        if "architecture" in arm_id:
            constraints.append("- Do NOT increase parameter count by more than 10% without justification.")
        if "optimizer" in arm_id:
            constraints.append("- Do NOT change the optimizer without testing on a reduced schedule first.")
        if "hyperparameter" in arm_id:
            constraints.append("- Do NOT tune more than 2 hyperparameters simultaneously.")

        return "\n".join(constraints)
