"""
Allocation efficiency comparator: bandit vs fixed-fraction allocation.
"""

from bandit.schemas import AllocationComparisonReport, load_jsonl


class AllocationEfficiencyComparator:
    """Compares actual bandit allocation efficiency vs fixed-fraction baseline."""

    def compare(
        self,
        treatment_log: str,
        control_log: str,
    ) -> AllocationComparisonReport:
        """Compare treatment (bandit) vs control (fixed-fraction) allocation.

        Analyzes:
        - Per-arm allocation fractions
        - Per-arm efficiency (delta per attempt)
        - Allocation gaps (under-allocated arms with good performance)
        - Allocation mistakes (over-allocated arms with poor performance)
        """
        treatment_entries = load_jsonl(treatment_log)
        control_entries = load_jsonl(control_log)

        t_alloc = self._compute_allocation(treatment_entries)
        c_alloc = self._compute_allocation(control_entries)

        per_arm_eff = self._compute_per_arm_efficiency(treatment_entries, control_entries)
        gaps = self._find_allocation_gaps(t_alloc, per_arm_eff)
        mistakes = self._find_allocation_mistakes(t_alloc, per_arm_eff)

        return AllocationComparisonReport(
            treatment_allocation=t_alloc,
            control_allocation=c_alloc,
            per_arm_efficiency=per_arm_eff,
            allocation_gaps=gaps,
            allocation_mistakes=mistakes,
        )

    def _compute_allocation(self, entries: list) -> dict:
        """Compute per-arm allocation fractions."""
        counts = {}
        total = 0
        for e in entries:
            arm_id = e.get("arm_selected", "")
            if arm_id:
                counts[arm_id] = counts.get(arm_id, 0) + 1
                total += 1

        if total == 0:
            return {}

        return {arm_id: count / total for arm_id, count in counts.items()}

    def _compute_per_arm_efficiency(
        self,
        treatment_entries: list,
        control_entries: list,
    ) -> dict:
        """Compute per-arm efficiency for both treatment and control."""
        result = {}

        for label, entries in [("treatment", treatment_entries),
                                ("control", control_entries)]:
            arm_data = {}
            for e in entries:
                arm_id = e.get("arm_selected", "")
                delta = e.get("delta")
                if arm_id:
                    arm_data.setdefault(arm_id, {"attempts": 0, "deltas": []})
                    arm_data[arm_id]["attempts"] += 1
                    if delta is not None:
                        arm_data[arm_id]["deltas"].append(delta)

            for arm_id, data in arm_data.items():
                result.setdefault(arm_id, {})
                deltas = data["deltas"]
                attempts = data["attempts"]
                result[arm_id][f"{label}_attempts"] = attempts
                result[arm_id][f"{label}_mean_delta"] = (
                    sum(deltas) / len(deltas) if deltas else 0.0)
                result[arm_id][f"{label}_efficiency"] = (
                    sum(d for d in deltas if d < 0) / attempts
                    if attempts > 0 else 0.0)

        return result

    def _find_allocation_gaps(
        self,
        allocation: dict,
        efficiency: dict,
    ) -> list:
        """Find arms that are under-allocated despite good efficiency."""
        gaps = []
        if not allocation or not efficiency:
            return gaps

        avg_alloc = sum(allocation.values()) / len(allocation) if allocation else 0.0

        for arm_id, eff in efficiency.items():
            alloc_frac = allocation.get(arm_id, 0.0)
            t_eff = eff.get("treatment_efficiency", 0.0)
            # Under-allocated and efficient (negative delta is good)
            if alloc_frac < avg_alloc * 0.5 and t_eff < -0.001:
                gaps.append(
                    f"Arm {arm_id}: allocated {alloc_frac:.1%} "
                    f"(below avg {avg_alloc:.1%}) but efficiency={t_eff:.4f}")

        return gaps

    def _find_allocation_mistakes(
        self,
        allocation: dict,
        efficiency: dict,
    ) -> list:
        """Find arms that are over-allocated despite poor efficiency."""
        mistakes = []
        if not allocation or not efficiency:
            return mistakes

        avg_alloc = sum(allocation.values()) / len(allocation) if allocation else 0.0

        for arm_id, eff in efficiency.items():
            alloc_frac = allocation.get(arm_id, 0.0)
            t_eff = eff.get("treatment_efficiency", 0.0)
            # Over-allocated and inefficient (positive or zero efficiency)
            if alloc_frac > avg_alloc * 1.5 and t_eff >= 0:
                mistakes.append(
                    f"Arm {arm_id}: allocated {alloc_frac:.1%} "
                    f"(above avg {avg_alloc:.1%}) but efficiency={t_eff:.4f}")

        return mistakes
