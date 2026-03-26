"""
Phase 3.6 — AblationInsightExtractor: analyze which component carried value
when a paper modification is ablated, and compare claimed vs revealed mechanisms.
"""

import logging

logger = logging.getLogger(__name__)


class AblationInsightExtractor:
    """Extract insights from ablation reports for paper-sourced modifications."""

    def extract_insights(
        self,
        technique: dict,
        ablation_report: dict,
    ) -> dict:
        """Analyze ablation results against the paper's claimed mechanism.

        Args:
            technique: Dict with technique metadata including:
                - name: technique name
                - description: what the paper claims
                - modification_category: category of the modification
                - pseudo_code: optional pseudo-code from the paper
            ablation_report: Dict from the ablation phase including:
                - components: list of component dicts with {name, delta_when_removed, kept}
                - stripped: list of stripped component names
                - final_delta: overall delta after ablation

        Returns:
            Dict with:
                - component_match: float 0-1 indicating how well paper's claim
                  matches ablation reality
                - discrepancies: list of discrepancy descriptions
                - insight_text: human-readable summary
                - value_components: list of components that carried value
                - neutral_components: list of neutral/stripped components
        """
        if not ablation_report or not technique:
            return {
                "component_match": 0.0,
                "discrepancies": ["missing ablation_report or technique data"],
                "insight_text": "Insufficient data for ablation insight extraction.",
                "value_components": [],
                "neutral_components": [],
            }

        components = ablation_report.get("components", [])
        stripped = set(ablation_report.get("stripped", []))

        value_components = []
        neutral_components = []

        for comp in components:
            name = comp.get("name", "unknown")
            delta = comp.get("delta_when_removed", 0.0)
            kept = comp.get("kept", True)

            if kept and abs(delta) > 1e-5:
                value_components.append({
                    "name": name,
                    "delta_when_removed": delta,
                })
            else:
                neutral_components.append({
                    "name": name,
                    "delta_when_removed": delta,
                })

        # Compute component match score
        total_components = len(components)
        if total_components == 0:
            component_match = 0.0
        else:
            # Higher match = more components carried value (paper's idea worked broadly)
            component_match = len(value_components) / total_components

        # Identify discrepancies
        discrepancies = []
        technique_name = technique.get("name", "unknown technique")
        technique_category = technique.get("modification_category", "")

        if not value_components:
            discrepancies.append(
                f"No components from '{technique_name}' carried measurable value "
                f"after ablation — paper's claimed benefit may be an artifact."
            )

        if len(neutral_components) > len(value_components) and value_components:
            discrepancies.append(
                f"Majority of components ({len(neutral_components)}/{total_components}) "
                f"were neutral — the benefit is concentrated in "
                f"{len(value_components)} component(s)."
            )

        if stripped:
            discrepancies.append(
                f"{len(stripped)} component(s) stripped as neutral: "
                f"{', '.join(sorted(stripped)[:5])}"
            )

        # Build insight text
        if value_components:
            value_names = [c["name"] for c in value_components[:3]]
            insight_text = (
                f"Paper technique '{technique_name}' ({technique_category}): "
                f"{len(value_components)}/{total_components} components carried value. "
                f"Key contributors: {', '.join(value_names)}. "
            )
            if neutral_components:
                insight_text += (
                    f"{len(neutral_components)} component(s) were neutral and stripped."
                )
        else:
            insight_text = (
                f"Paper technique '{technique_name}' ({technique_category}): "
                f"No individual component carried significant value after ablation. "
                f"The claimed mechanism may not be the actual driver of improvement."
            )

        return {
            "component_match": round(component_match, 3),
            "discrepancies": discrepancies,
            "insight_text": insight_text,
            "value_components": value_components,
            "neutral_components": neutral_components,
        }
