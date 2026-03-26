"""Compile validated insights into a knowledge base."""

import time

from meta.schemas import save_json


class MetaKnowledgeBaseWriter:
    """Compiles validated insights, experiment summary, and config
    documentation into a meta_knowledge_base.json structure."""

    def compile(
        self,
        validated_insights: list,
        experiment_summary: dict,
        config_doc: dict,
    ) -> dict:
        """Produce a knowledge base dict.

        Structure:
        - validated_insights: {universal: [...], conditional: [...]}
        - campaign_specific_findings: [...]
        - interaction_map: {pair_key: {...}}
        - sensitivity_classification: {param_id: class}
        - recommended_defaults: {param_id: value}
        - anti_patterns: [...]
        """
        universal = []
        conditional = []
        campaign_specific = []
        interaction_map = {}
        recommended_defaults = {}
        anti_patterns = []

        for entry in validated_insights:
            insight = entry.get("insight", {})
            validation = entry.get("validation", {})
            itype = insight.get("type", "")
            record = {
                "insight_id": insight.get("insight_id", ""),
                "description": insight.get("description", ""),
                "confidence": insight.get("confidence", "low"),
                "recommended_default": insight.get("recommended_default"),
                "validated": validation.get("validated", False),
                "validation_ir": validation.get("validation_ir", 0.0),
                "improvement": validation.get("improvement", 0.0),
            }

            if not validation.get("validated", False):
                anti_patterns.append({
                    "insight_id": insight.get("insight_id", ""),
                    "description": insight.get("description", ""),
                    "reason": "Failed transfer validation",
                })
                continue

            if itype == "universal":
                universal.append(record)
                if insight.get("recommended_default") is not None:
                    # Use first token of insight_id as param key fallback
                    param_id = self._extract_param_id(insight)
                    recommended_defaults[param_id] = insight["recommended_default"]
            elif itype == "interaction":
                pair_key = insight.get("insight_id", "")
                interaction_map[pair_key] = record
            elif itype in ("phase_dependent", "scale_dependent"):
                conditional.append(record)
            else:
                campaign_specific.append(record)

        # Sensitivity classification from config_doc
        sensitivity_classification = {}
        doc_dict = config_doc.to_dict() if hasattr(config_doc, "to_dict") else (config_doc if isinstance(config_doc, dict) else {})
        for param_id, dim_info in doc_dict.get("dimensions", {}).items():
            sensitivity_classification[param_id] = (
                dim_info.get("sensitivity", "unknown") if isinstance(dim_info, dict) else "unknown"
            )

        return {
            "schema_version": "1.0",
            "compiled_at": time.time(),
            "validated_insights": {
                "universal": universal,
                "conditional": conditional,
            },
            "campaign_specific_findings": campaign_specific,
            "interaction_map": interaction_map,
            "sensitivity_classification": sensitivity_classification,
            "recommended_defaults": recommended_defaults,
            "anti_patterns": anti_patterns,
            "experiment_summary": experiment_summary,
        }

    def _extract_param_id(self, insight: dict) -> str:
        """Best-effort extraction of param_id from insight description."""
        desc = insight.get("description", "")
        if "'" in desc:
            parts = desc.split("'")
            if len(parts) >= 2:
                return parts[1]
        return insight.get("insight_id", "unknown")
