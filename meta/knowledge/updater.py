"""Incremental updates to an existing knowledge base."""

import time


class KnowledgeBaseUpdater:
    """Merges new validated insights and deprecates invalidated ones."""

    def update(
        self,
        knowledge_base: dict,
        new_insights: list,
        new_validations: list,
    ) -> dict:
        """Update a knowledge base with new insights and validations.

        - Merge new validated insights into the appropriate category.
        - Deprecate insights that failed re-validation.

        Args:
            knowledge_base: Existing knowledge base dict.
            new_insights: List of MetaInsight.to_dict() dicts.
            new_validations: List of TransferValidationResult.to_dict() dicts.

        Returns:
            Updated knowledge base dict.
        """
        kb = dict(knowledge_base)
        kb.setdefault("validated_insights", {"universal": [], "conditional": []})
        kb.setdefault("campaign_specific_findings", [])
        kb.setdefault("anti_patterns", [])
        kb.setdefault("recommended_defaults", {})

        # Index validations by insight_id
        val_by_id = {v.get("insight_id", ""): v for v in new_validations}

        existing_ids = self._all_insight_ids(kb)

        for insight in new_insights:
            iid = insight.get("insight_id", "")
            validation = val_by_id.get(iid, {})
            validated = validation.get("validated", False)

            if iid in existing_ids:
                if not validated:
                    # Deprecate existing insight
                    self._deprecate(kb, iid)
                continue  # don't duplicate

            if not validated:
                kb["anti_patterns"].append({
                    "insight_id": iid,
                    "description": insight.get("description", ""),
                    "reason": "Failed transfer validation",
                    "deprecated_at": time.time(),
                })
                continue

            record = {
                "insight_id": iid,
                "description": insight.get("description", ""),
                "confidence": insight.get("confidence", "low"),
                "recommended_default": insight.get("recommended_default"),
                "validated": True,
                "validation_ir": validation.get("validation_ir", 0.0),
                "improvement": validation.get("improvement", 0.0),
            }

            itype = insight.get("type", "")
            if itype == "universal":
                kb["validated_insights"]["universal"].append(record)
                if insight.get("recommended_default") is not None:
                    desc = insight.get("description", "")
                    param_id = self._extract_param_id(desc, iid)
                    kb["recommended_defaults"][param_id] = insight["recommended_default"]
            elif itype in ("phase_dependent", "scale_dependent"):
                kb["validated_insights"]["conditional"].append(record)
            else:
                kb["campaign_specific_findings"].append(record)

        kb["last_updated"] = time.time()
        return kb

    def _all_insight_ids(self, kb: dict) -> set:
        ids = set()
        for entry in kb.get("validated_insights", {}).get("universal", []):
            ids.add(entry.get("insight_id", ""))
        for entry in kb.get("validated_insights", {}).get("conditional", []):
            ids.add(entry.get("insight_id", ""))
        for entry in kb.get("campaign_specific_findings", []):
            ids.add(entry.get("insight_id", ""))
        for entry in kb.get("anti_patterns", []):
            ids.add(entry.get("insight_id", ""))
        return ids

    def _deprecate(self, kb: dict, insight_id: str) -> None:
        """Move an insight to anti_patterns."""
        for category in ["universal", "conditional"]:
            entries = kb.get("validated_insights", {}).get(category, [])
            for entry in entries:
                if entry.get("insight_id") == insight_id:
                    entries.remove(entry)
                    kb["anti_patterns"].append({
                        "insight_id": insight_id,
                        "description": entry.get("description", ""),
                        "reason": "Failed re-validation; deprecated",
                        "deprecated_at": time.time(),
                    })
                    return
        for entry in kb.get("campaign_specific_findings", []):
            if entry.get("insight_id") == insight_id:
                kb["campaign_specific_findings"].remove(entry)
                kb["anti_patterns"].append({
                    "insight_id": insight_id,
                    "description": entry.get("description", ""),
                    "reason": "Failed re-validation; deprecated",
                    "deprecated_at": time.time(),
                })
                return

    def _extract_param_id(self, desc: str, fallback: str) -> str:
        if "'" in desc:
            parts = desc.split("'")
            if len(parts) >= 2:
                return parts[1]
        return fallback
