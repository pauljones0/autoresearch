"""
A/B statistical analyzer: Mann-Whitney U, effect size, per-arm contributions.
"""

import math

from bandit.schemas import ABAnalysisReport, load_jsonl


class ABStatisticalAnalyzer:
    """Performs statistical analysis on A/B test results."""

    def analyze(
        self,
        treatment_paths: list,
        control_paths: list,
    ) -> ABAnalysisReport:
        """Analyze treatment vs control from log files.

        Computes:
        - Mann-Whitney U test for significance
        - Effect size (rank-biserial correlation)
        - Per-arm contribution analysis
        - Annealing stepping stone count
        - Waste analysis (rejected iterations)
        """
        treatment_data = self._load_all(treatment_paths)
        control_data = self._load_all(control_paths)

        treatment_deltas = [e["delta"] for e in treatment_data
                           if e.get("delta") is not None]
        control_deltas = [e["delta"] for e in control_data
                         if e.get("delta") is not None]

        # Mann-Whitney U test
        u_stat, p_value = self._mann_whitney_u(treatment_deltas, control_deltas)

        # Effect size (rank-biserial correlation)
        n1 = len(treatment_deltas)
        n2 = len(control_deltas)
        if n1 > 0 and n2 > 0:
            effect_size = 1.0 - (2.0 * u_stat) / (n1 * n2)
        else:
            effect_size = 0.0

        # Medians
        t_med = _median(treatment_deltas)
        c_med = _median(control_deltas)

        # Per-arm contribution
        per_arm = self._per_arm_contributions(treatment_data)

        # Annealing stepping stones
        stepping_stones = self._count_stepping_stones(treatment_data)

        # Waste analysis
        waste_t = self._waste_rate(treatment_data)
        waste_c = self._waste_rate(control_data)

        # Verdict
        significant = p_value < 0.05
        if significant and t_med < c_med:
            verdict = "bandit_significantly_better"
        elif significant and t_med > c_med:
            verdict = "control_significantly_better"
        elif significant:
            verdict = "significant_no_practical_difference"
        else:
            verdict = "no_significant_difference"

        return ABAnalysisReport(
            n_seeds=max(len(treatment_paths), len(control_paths)),
            treatment_median_improvement=t_med,
            control_median_improvement=c_med,
            u_statistic=u_stat,
            p_value=p_value,
            significant=significant,
            effect_size=effect_size,
            per_arm_contributions=per_arm,
            annealing_stepping_stones=stepping_stones,
            waste_rate_treatment=waste_t,
            waste_rate_control=waste_c,
            verdict=verdict,
        )

    def _load_all(self, paths: list) -> list:
        """Load all entries from multiple JSONL paths."""
        all_entries = []
        for path in paths:
            all_entries.extend(load_jsonl(path))
        return all_entries

    def _mann_whitney_u(
        self, x: list, y: list
    ) -> tuple:
        """Compute Mann-Whitney U statistic and approximate p-value.

        Uses normal approximation for large samples.
        Returns (U, p_value).
        """
        if not x or not y:
            return 0.0, 1.0

        n1, n2 = len(x), len(y)
        # Combine and rank
        combined = [(v, 0) for v in x] + [(v, 1) for v in y]
        combined.sort(key=lambda t: t[0])

        # Assign ranks with tie handling
        ranks = [0.0] * len(combined)
        i = 0
        while i < len(combined):
            j = i
            while j < len(combined) and combined[j][0] == combined[i][0]:
                j += 1
            avg_rank = (i + j + 1) / 2.0  # 1-based
            for k in range(i, j):
                ranks[k] = avg_rank
            i = j

        # Sum ranks for group 0 (treatment)
        r1 = sum(ranks[k] for k in range(len(combined))
                 if combined[k][1] == 0)

        u1 = r1 - n1 * (n1 + 1) / 2.0
        u2 = n1 * n2 - u1
        u_stat = min(u1, u2)

        # Normal approximation for p-value
        mu = n1 * n2 / 2.0
        sigma = math.sqrt(n1 * n2 * (n1 + n2 + 1) / 12.0)
        if sigma == 0:
            return u_stat, 1.0

        z = (u_stat - mu) / sigma
        # Two-tailed p-value using approximation
        p_value = 2.0 * _norm_cdf(z)

        return u_stat, min(p_value, 1.0)

    def _per_arm_contributions(self, entries: list) -> dict:
        """Compute per-arm delta contributions."""
        arm_deltas = {}
        for e in entries:
            arm_id = e.get("arm_selected", "")
            delta = e.get("delta")
            if arm_id and delta is not None:
                arm_deltas.setdefault(arm_id, []).append(delta)

        result = {}
        for arm_id, deltas in arm_deltas.items():
            result[arm_id] = {
                "count": len(deltas),
                "mean_delta": sum(deltas) / len(deltas) if deltas else 0.0,
                "median_delta": _median(deltas),
                "total_delta": sum(deltas),
            }
        return result

    def _count_stepping_stones(self, entries: list) -> int:
        """Count annealing stepping stones (accepted regressions
        followed by improvements)."""
        count = 0
        prev_was_annealing_accept = False
        for e in entries:
            accepted_by = e.get("accepted_by", "")
            delta = e.get("delta")
            if prev_was_annealing_accept and delta is not None and delta <= 0:
                count += 1
            prev_was_annealing_accept = (accepted_by == "annealing")
        return count

    def _waste_rate(self, entries: list) -> float:
        """Compute waste rate (fraction of rejected iterations)."""
        if not entries:
            return 0.0
        rejected = sum(1 for e in entries if not e.get("accepted", False))
        return rejected / len(entries)


def _median(values: list) -> float:
    """Compute median."""
    if not values:
        return 0.0
    s = sorted(values)
    n = len(s)
    if n % 2 == 1:
        return s[n // 2]
    return (s[n // 2 - 1] + s[n // 2]) / 2.0


def _norm_cdf(z: float) -> float:
    """Approximate standard normal CDF using error function approximation."""
    # Abramowitz and Stegun approximation
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
