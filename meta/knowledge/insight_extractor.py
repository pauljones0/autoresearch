"""Extract transferable insights from meta-optimization history."""

import hashlib
import time

from meta.schemas import MetaInsight


class InsightExtractor:
    """Extracts structured insights from experiment history and documentation."""

    def extract(
        self,
        experiment_history: list,
        config_doc: dict,
        sensitivity_report: dict,
    ) -> list:
        """Extract insights of four types: universal, phase_dependent,
        scale_dependent, interaction.

        Args:
            experiment_history: List of experiment result dicts.
            config_doc: ConfigDocumentation.to_dict() output.
            sensitivity_report: SensitivityReport.to_dict() output.

        Returns:
            List of MetaInsight objects.
        """
        insights = []
        insights.extend(self._extract_universal(experiment_history, config_doc))
        insights.extend(self._extract_phase_dependent(experiment_history, config_doc))
        insights.extend(self._extract_scale_dependent(experiment_history, config_doc))
        insights.extend(self._extract_interaction(experiment_history, sensitivity_report))
        return insights

    # ------------------------------------------------------------------
    # Universal insights: consistent across progress
    # ------------------------------------------------------------------

    def _extract_universal(self, history: list, config_doc: dict) -> list:
        insights = []
        dims = config_doc.get("dimensions", {})
        for param_id, info in dims.items():
            evidence = info.get("promotion_evidence", {})
            if not evidence.get("promoted", False):
                continue
            # Check consistency: was this variant always better?
            variants = info.get("all_variants_tested", [])
            if len(variants) < 2:
                continue
            promoted_val = info.get("promoted_value")
            best_variant = max(variants, key=lambda v: v.get("mean_ir", 0))
            if str(best_variant.get("value")) == str(promoted_val):
                insights.append(MetaInsight(
                    insight_id=self._make_id("universal", param_id),
                    type="universal",
                    description=(
                        f"Parameter '{param_id}' performs best at value "
                        f"{promoted_val} consistently across all experiments."
                    ),
                    evidence_experiments=[
                        e.get("experiment_id", "") for e in history
                        if any(d.get("param_id") == param_id for d in e.get("config_diff", []))
                    ],
                    confidence=self._compute_confidence(variants),
                    transferability="universal",
                    recommended_default=promoted_val,
                ))
        return insights

    # ------------------------------------------------------------------
    # Phase-dependent insights: behaviour changes during campaign
    # ------------------------------------------------------------------

    def _extract_phase_dependent(self, history: list, config_doc: dict) -> list:
        insights = []
        if len(history) < 4:
            return insights

        mid = len(history) // 2
        early = history[:mid]
        late = history[mid:]

        dims = config_doc.get("dimensions", {})
        for param_id in dims:
            early_irs = self._param_irs(param_id, early)
            late_irs = self._param_irs(param_id, late)
            if not early_irs or not late_irs:
                continue
            early_mean = sum(early_irs) / len(early_irs)
            late_mean = sum(late_irs) / len(late_irs)
            if abs(early_mean - late_mean) > 0.1 * max(abs(early_mean), abs(late_mean), 1e-9):
                insights.append(MetaInsight(
                    insight_id=self._make_id("phase_dependent", param_id),
                    type="phase_dependent",
                    description=(
                        f"Parameter '{param_id}' shows different optimal values "
                        f"in early vs late campaign phases "
                        f"(early IR={early_mean:.4f}, late IR={late_mean:.4f})."
                    ),
                    evidence_experiments=[
                        e.get("experiment_id", "") for e in history
                        if any(d.get("param_id") == param_id for d in e.get("config_diff", []))
                    ],
                    confidence="medium",
                    transferability="conditional",
                ))
        return insights

    # ------------------------------------------------------------------
    # Scale-dependent insights
    # ------------------------------------------------------------------

    def _extract_scale_dependent(self, history: list, config_doc: dict) -> list:
        insights = []
        dims = config_doc.get("dimensions", {})
        for param_id in dims:
            # Look for experiments that record model size metadata
            size_irs = {}
            for exp in history:
                for diff in exp.get("config_diff", []):
                    if diff.get("param_id") == param_id:
                        size = exp.get("model_size", exp.get("scale", None))
                        if size is not None:
                            size_irs.setdefault(str(size), []).append(
                                exp.get("improvement_rate", 0.0)
                            )
            if len(size_irs) >= 2:
                means = {s: sum(v) / len(v) for s, v in size_irs.items()}
                vals = list(means.values())
                spread = max(vals) - min(vals)
                avg = sum(vals) / len(vals) if vals else 1.0
                if avg != 0 and spread / abs(avg) > 0.15:
                    insights.append(MetaInsight(
                        insight_id=self._make_id("scale_dependent", param_id),
                        type="scale_dependent",
                        description=(
                            f"Parameter '{param_id}' sensitivity varies with model scale: "
                            f"{means}."
                        ),
                        evidence_experiments=[
                            e.get("experiment_id", "") for e in history
                            if any(d.get("param_id") == param_id for d in e.get("config_diff", []))
                        ],
                        confidence="low",
                        transferability="conditional",
                    ))
        return insights

    # ------------------------------------------------------------------
    # Interaction insights
    # ------------------------------------------------------------------

    def _extract_interaction(self, history: list, sensitivity_report: dict) -> list:
        insights = []
        # Find experiments that changed multiple params simultaneously
        multi_param_exps = [
            e for e in history if len(e.get("config_diff", [])) >= 2
        ]
        seen_pairs = set()
        for exp in multi_param_exps:
            params = [d.get("param_id", "") for d in exp.get("config_diff", [])]
            for i in range(len(params)):
                for j in range(i + 1, len(params)):
                    pair = tuple(sorted([params[i], params[j]]))
                    if pair in seen_pairs:
                        continue
                    seen_pairs.add(pair)
                    insights.append(MetaInsight(
                        insight_id=self._make_id("interaction", f"{pair[0]}_{pair[1]}"),
                        type="interaction",
                        description=(
                            f"Parameters '{pair[0]}' and '{pair[1]}' were jointly "
                            f"modified in experiment {exp.get('experiment_id', '?')}; "
                            f"potential interaction detected."
                        ),
                        evidence_experiments=[exp.get("experiment_id", "")],
                        confidence="low",
                        transferability="campaign_specific",
                    ))
        return insights

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _param_irs(self, param_id: str, history: list) -> list:
        irs = []
        for exp in history:
            for diff in exp.get("config_diff", []):
                if diff.get("param_id") == param_id:
                    irs.append(exp.get("improvement_rate", 0.0))
        return irs

    def _compute_confidence(self, variants: list) -> str:
        total_exps = sum(v.get("n_experiments", 0) for v in variants)
        if total_exps >= 10:
            return "high"
        if total_exps >= 5:
            return "medium"
        return "low"

    def _make_id(self, insight_type: str, key: str) -> str:
        raw = f"{insight_type}:{key}"
        return hashlib.md5(raw.encode()).hexdigest()[:12]
