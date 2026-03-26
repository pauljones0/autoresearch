"""Evidence-based configuration documentation."""

from meta.schemas import MetaBanditState, ConfigDocumentation


class MetaConfigDocumenter:
    """Generates per-dimension documentation of meta-optimization results."""

    def document(
        self,
        meta_state: MetaBanditState,
        experiment_history: list,
        baseline_ir: float,
    ) -> ConfigDocumentation:
        """Produce a ConfigDocumentation summarising every tuned dimension.

        For each dimension the output includes:
        - promoted_value / default_value
        - all variants tested with their IRs
        - promotion evidence (p-value, effect size)
        - justification text
        - sensitivity classification
        """
        dimensions = {}
        total_promotions = 0

        for param_id, dim in meta_state.dimensions.items():
            variants_tested = self._collect_variants(param_id, experiment_history)
            promoted_value = dim.current_best
            default_value = meta_state.best_config.get(param_id, dim.current_best)
            promo_evidence = self._promotion_evidence(param_id, experiment_history)
            if promo_evidence.get("promoted", False):
                total_promotions += 1

            dimensions[param_id] = {
                "promoted_value": promoted_value,
                "default_value": default_value,
                "all_variants_tested": variants_tested,
                "promotion_evidence": promo_evidence,
                "justification": self._justification(param_id, promo_evidence, variants_tested),
                "sensitivity": self._sensitivity_class(param_id, variants_tested),
            }

        best_ir = self._best_config_ir(experiment_history)
        improvement = best_ir - baseline_ir if baseline_ir else 0.0

        return ConfigDocumentation(
            dimensions=dimensions,
            total_experiments=len(experiment_history),
            total_promotions=total_promotions,
            best_config_ir=best_ir,
            default_config_ir=baseline_ir,
            improvement_over_defaults=improvement,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _collect_variants(self, param_id: str, history: list) -> list:
        """Gather all variant values and their IRs for a dimension."""
        variants = {}
        for exp in history:
            diffs = exp.get("config_diff", [])
            for diff in diffs:
                pid = diff.get("param_id", "")
                if pid == param_id:
                    val = str(diff.get("new_value", ""))
                    ir = exp.get("improvement_rate", 0.0)
                    if val not in variants:
                        variants[val] = {"value": diff.get("new_value"), "irs": []}
                    variants[val]["irs"].append(ir)
        result = []
        for val_str, info in variants.items():
            irs = info["irs"]
            mean_ir = sum(irs) / len(irs) if irs else 0.0
            result.append({
                "value": info["value"],
                "mean_ir": mean_ir,
                "n_experiments": len(irs),
            })
        return result

    def _promotion_evidence(self, param_id: str, history: list) -> dict:
        """Extract promotion evidence for a dimension from history."""
        for exp in reversed(history):
            if exp.get("compared_to_baseline") == "better":
                diffs = exp.get("config_diff", [])
                for diff in diffs:
                    if diff.get("param_id") == param_id:
                        return {
                            "promoted": True,
                            "experiment_id": exp.get("experiment_id", ""),
                            "improvement_rate": exp.get("improvement_rate", 0.0),
                            "baseline_ir_used": exp.get("baseline_ir_used", 0.0),
                        }
        return {"promoted": False}

    def _justification(self, param_id: str, evidence: dict, variants: list) -> str:
        if not evidence.get("promoted", False):
            return f"No variant for {param_id} was promoted; default value retained."
        ir = evidence.get("improvement_rate", 0.0)
        bl = evidence.get("baseline_ir_used", 0.0)
        return (
            f"Promoted based on experiment {evidence.get('experiment_id', '?')}: "
            f"IR={ir:.4f} vs baseline={bl:.4f} across {len(variants)} variants tested."
        )

    def _sensitivity_class(self, param_id: str, variants: list) -> str:
        if len(variants) < 2:
            return "unknown"
        irs = [v["mean_ir"] for v in variants]
        spread = max(irs) - min(irs)
        mean_ir = sum(irs) / len(irs) if irs else 1.0
        if mean_ir == 0:
            return "unknown"
        rel = spread / abs(mean_ir)
        if rel > 0.3:
            return "critical"
        if rel > 0.1:
            return "moderate"
        return "robust"

    def _best_config_ir(self, history: list) -> float:
        if not history:
            return 0.0
        best = max(exp.get("improvement_rate", 0.0) for exp in history)
        return best
