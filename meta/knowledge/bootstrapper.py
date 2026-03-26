"""Bootstrap new campaigns from an existing knowledge base."""

import os
import time

from meta.schemas import MetaBanditState, DimensionState, load_json, save_json


class NewCampaignBootstrapper:
    """Applies knowledge base insights to initialise a new campaign."""

    def bootstrap(self, knowledge_base_path: str, new_campaign_dir: str) -> dict:
        """Bootstrap a new campaign from a previous knowledge base.

        Steps:
        1. Read the knowledge base JSON.
        2. Apply universal insights as recommended defaults.
        3. Apply conditional insights based on campaign_profile.json.
        4. Set regime to 'active'.
        5. Warm-start meta-bandit posteriors from knowledge base data.

        Args:
            knowledge_base_path: Path to meta_knowledge_base.json.
            new_campaign_dir: Path to the new campaign directory.

        Returns:
            Dict with the bootstrapped MetaBanditState and applied insights.
        """
        kb = load_json(knowledge_base_path)
        profile_path = os.path.join(new_campaign_dir, "campaign_profile.json")
        profile = load_json(profile_path)

        config = {}
        applied_insights = []

        # Apply universal insights
        for insight in kb.get("validated_insights", {}).get("universal", []):
            rec = insight.get("recommended_default")
            if rec is not None:
                param_id = self._extract_param_id(insight)
                config[param_id] = rec
                applied_insights.append({
                    "insight_id": insight.get("insight_id", ""),
                    "type": "universal",
                    "param_id": param_id,
                    "value": rec,
                })

        # Apply conditional insights if profile matches
        for insight in kb.get("validated_insights", {}).get("conditional", []):
            if self._matches_profile(insight, profile):
                rec = insight.get("recommended_default")
                if rec is not None:
                    param_id = self._extract_param_id(insight)
                    config[param_id] = rec
                    applied_insights.append({
                        "insight_id": insight.get("insight_id", ""),
                        "type": "conditional",
                        "param_id": param_id,
                        "value": rec,
                    })

        # Also apply recommended_defaults from KB directly
        for param_id, val in kb.get("recommended_defaults", {}).items():
            if param_id not in config:
                config[param_id] = val

        # Build initial meta-bandit state
        state = MetaBanditState()
        state.meta_regime = "active"
        state.current_config = config
        state.best_config = dict(config)
        state.metadata["created_at"] = time.time()
        state.metadata["last_updated"] = time.time()
        state.metadata["bootstrapped_from"] = knowledge_base_path

        # Warm-start posteriors from KB sensitivity data
        sensitivity = kb.get("sensitivity_classification", {})
        for param_id, val in config.items():
            dim = DimensionState(
                param_id=param_id,
                variants=[val],
                current_best=val,
            )
            # Set initial posteriors based on sensitivity
            sens_class = sensitivity.get(param_id, "unknown")
            alpha, beta = self._initial_posteriors(sens_class)
            dim.variant_posteriors[str(val)] = {"alpha": alpha, "beta": beta}
            state.dimensions[param_id] = dim

        # Save bootstrapped state
        os.makedirs(new_campaign_dir, exist_ok=True)
        state_path = os.path.join(new_campaign_dir, "meta_bandit_state.json")
        save_json(state, state_path)

        result = {
            "state": state.to_dict(),
            "applied_insights": applied_insights,
            "config": config,
            "knowledge_base_used": knowledge_base_path,
            "anti_patterns_avoided": [
                ap.get("insight_id", "") for ap in kb.get("anti_patterns", [])
            ],
        }

        # Save bootstrap report
        report_path = os.path.join(new_campaign_dir, "bootstrap_report.json")
        save_json(result, report_path)

        return result

    def _extract_param_id(self, insight: dict) -> str:
        desc = insight.get("description", "")
        if "'" in desc:
            parts = desc.split("'")
            if len(parts) >= 2:
                return parts[1]
        return insight.get("insight_id", "unknown")

    def _matches_profile(self, insight: dict, profile: dict) -> bool:
        """Check if a conditional insight applies to the campaign profile."""
        if not profile:
            return False
        desc = insight.get("description", "").lower()
        # Match scale-dependent insights by model size
        if "scale" in desc or "model size" in desc:
            return "model_size" in profile
        # Match phase-dependent insights
        if "phase" in desc or "early" in desc or "late" in desc:
            return True  # phase insights generally apply
        return False

    def _initial_posteriors(self, sensitivity_class: str) -> tuple:
        """Return (alpha, beta) warm-start values based on sensitivity."""
        if sensitivity_class == "critical":
            return (2.0, 2.0)  # uncertain, need more data
        if sensitivity_class == "moderate":
            return (5.0, 3.0)  # slightly optimistic
        if sensitivity_class == "robust":
            return (10.0, 3.0)  # confident in current value
        return (3.0, 3.0)  # unknown
