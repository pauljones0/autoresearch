"""Sensitivity analysis for meta-optimized parameters."""

from meta.schemas import MetaBanditState, SensitivityReport


class MetaSensitivityAnalyzer:
    """Analyses sensitivity of IR to perturbations of each tuned dimension."""

    def analyze(
        self,
        meta_state: MetaBanditState,
        experiment_history: list,
        n_perturbations: int = 3,
    ) -> SensitivityReport:
        """Compute sensitivity per dimension via +/-10% perturbation analysis.

        Sensitivity = |IR_{+10%} - IR_{-10%}| / (0.2 * IR_promoted).
        Classification:
            > 0.3  -> critical
            0.1-0.3 -> moderate
            < 0.1  -> robust
        """
        per_dimension = {}
        critical = []
        robust = []

        for param_id, dim in meta_state.dimensions.items():
            promoted_value = dim.current_best
            promoted_ir = self._get_promoted_ir(param_id, experiment_history)

            if promoted_ir == 0.0 or promoted_value is None:
                per_dimension[param_id] = {
                    "promoted_value": promoted_value,
                    "promoted_ir": promoted_ir,
                    "sensitivity": 0.0,
                    "classification": "unknown",
                    "perturbations": [],
                }
                continue

            perturbations = self._compute_perturbation_irs(
                param_id, promoted_value, promoted_ir, experiment_history, n_perturbations
            )

            ir_plus = perturbations.get("ir_plus_10", promoted_ir)
            ir_minus = perturbations.get("ir_minus_10", promoted_ir)
            sensitivity = abs(ir_plus - ir_minus) / (0.2 * abs(promoted_ir))

            if sensitivity > 0.3:
                classification = "critical"
                critical.append(param_id)
            elif sensitivity >= 0.1:
                classification = "moderate"
            else:
                classification = "robust"
                robust.append(param_id)

            per_dimension[param_id] = {
                "promoted_value": promoted_value,
                "promoted_ir": promoted_ir,
                "sensitivity": round(sensitivity, 4),
                "classification": classification,
                "ir_plus_10": ir_plus,
                "ir_minus_10": ir_minus,
                "perturbations": perturbations.get("details", []),
            }

        return SensitivityReport(
            per_dimension=per_dimension,
            critical_dimensions=critical,
            robust_dimensions=robust,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_promoted_ir(self, param_id: str, history: list) -> float:
        """Get the IR of the promoted variant."""
        for exp in reversed(history):
            if exp.get("compared_to_baseline") == "better":
                for diff in exp.get("config_diff", []):
                    if diff.get("param_id") == param_id:
                        return exp.get("improvement_rate", 0.0)
        # Fallback: average IR across experiments touching this param
        irs = []
        for exp in history:
            for diff in exp.get("config_diff", []):
                if diff.get("param_id") == param_id:
                    irs.append(exp.get("improvement_rate", 0.0))
        return sum(irs) / len(irs) if irs else 0.0

    def _compute_perturbation_irs(
        self,
        param_id: str,
        promoted_value,
        promoted_ir: float,
        history: list,
        n_perturbations: int,
    ) -> dict:
        """Estimate IR at +10% and -10% of promoted value from history data.

        Uses nearby experiment results to interpolate. When no nearby data
        exists, assumes IR equals the promoted IR (sensitivity = 0).
        """
        if not isinstance(promoted_value, (int, float)):
            return {"ir_plus_10": promoted_ir, "ir_minus_10": promoted_ir, "details": []}

        target_plus = promoted_value * 1.1
        target_minus = promoted_value * 0.9

        # Collect (value, ir) pairs from experiments
        pairs = []
        for exp in history:
            for diff in exp.get("config_diff", []):
                if diff.get("param_id") == param_id:
                    val = diff.get("new_value")
                    if isinstance(val, (int, float)):
                        pairs.append((val, exp.get("improvement_rate", 0.0)))

        ir_plus = self._interpolate(pairs, target_plus, promoted_value, promoted_ir)
        ir_minus = self._interpolate(pairs, target_minus, promoted_value, promoted_ir)

        details = []
        for val, ir in pairs[:n_perturbations]:
            details.append({"value": val, "ir": ir})

        return {"ir_plus_10": ir_plus, "ir_minus_10": ir_minus, "details": details}

    def _interpolate(
        self, pairs: list, target: float, promoted_value: float, promoted_ir: float
    ) -> float:
        """Simple nearest-neighbour interpolation."""
        if not pairs:
            return promoted_ir
        # Sort by distance to target
        sorted_pairs = sorted(pairs, key=lambda p: abs(p[0] - target))
        nearest_val, nearest_ir = sorted_pairs[0]
        if abs(nearest_val - promoted_value) < 1e-12:
            # Exact match to promoted; use second nearest if available
            if len(sorted_pairs) > 1:
                return sorted_pairs[1][1]
            return promoted_ir
        return nearest_ir
