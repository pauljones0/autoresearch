"""
Acceptance log analysis for bandit annealing decisions.
"""

import json
from bandit.schemas import AcceptanceAnalysisReport, load_jsonl


class AcceptanceAnalyzer:
    """Analyzes acceptance decision logs to compute statistics."""

    def analyze(
        self,
        log_path: str,
        window_size: int = 50,
    ) -> AcceptanceAnalysisReport:
        """Analyze acceptance decisions from a JSONL log file.

        Computes acceptance rates, per-arm rates, stepping stone rate,
        dead end rate, temperature trajectory, and surrogate modulation impact.

        Args:
            log_path: Path to JSONL log of AcceptanceDecision records.
            window_size: Rolling window size for trajectory analysis.

        Returns:
            AcceptanceAnalysisReport with computed statistics.
        """
        entries = load_jsonl(log_path)

        if not entries:
            return AcceptanceAnalysisReport()

        total = len(entries)
        greedy_count = 0
        annealing_accepted_count = 0
        rejected_count = 0
        stepping_stones = 0
        dead_ends = 0

        per_arm_annealing = {}  # arm_id -> [accepted_count, total_annealing]
        temperature_trajectory = {}  # arm_id -> list of T_effective
        surrogate_impact = {"with_surrogate": 0, "without_surrogate": 0,
                            "accepted_with": 0, "accepted_without": 0}

        for i, entry in enumerate(entries):
            accepted_by = entry.get("accepted_by", "rejected")
            arm_id = entry.get("arm_id", "unknown")
            accepted = entry.get("accepted", False)
            T_eff = entry.get("T_effective", 0.0)
            surrogate_pred = entry.get("surrogate_predicted_delta")

            if accepted_by == "improvement":
                greedy_count += 1
            elif accepted_by == "annealing":
                annealing_accepted_count += 1
            else:
                rejected_count += 1

            # Per-arm annealing stats (only for non-improvement decisions)
            if accepted_by != "improvement":
                if arm_id not in per_arm_annealing:
                    per_arm_annealing[arm_id] = [0, 0]
                per_arm_annealing[arm_id][1] += 1
                if accepted_by == "annealing":
                    per_arm_annealing[arm_id][0] += 1

            # Temperature trajectory
            if arm_id not in temperature_trajectory:
                temperature_trajectory[arm_id] = []
            temperature_trajectory[arm_id].append(T_eff)

            # Surrogate impact
            if surrogate_pred is not None:
                surrogate_impact["with_surrogate"] += 1
                if accepted:
                    surrogate_impact["accepted_with"] += 1
            else:
                surrogate_impact["without_surrogate"] += 1
                if accepted:
                    surrogate_impact["accepted_without"] += 1

            # Stepping stone: accepted via annealing, then next is improvement
            if accepted_by == "annealing" and i + 1 < total:
                next_entry = entries[i + 1]
                if next_entry.get("accepted_by") == "improvement":
                    stepping_stones += 1

            # Dead end: accepted via annealing, then next N are all rejections
            if accepted_by == "annealing":
                is_dead_end = True
                lookahead = min(3, total - i - 1)
                if lookahead == 0:
                    is_dead_end = False
                for j in range(1, lookahead + 1):
                    if entries[i + j].get("accepted", False):
                        is_dead_end = False
                        break
                if is_dead_end and lookahead > 0:
                    dead_ends += 1

        total_accepted = greedy_count + annealing_accepted_count
        acceptance_rate = total_accepted / total if total > 0 else 0.0
        greedy_rate = greedy_count / total if total > 0 else 0.0
        annealing_rate = annealing_accepted_count / total if total > 0 else 0.0

        annealing_total = annealing_accepted_count + rejected_count
        stepping_stone_rate = stepping_stones / annealing_accepted_count if annealing_accepted_count > 0 else 0.0
        dead_end_rate = dead_ends / annealing_accepted_count if annealing_accepted_count > 0 else 0.0

        per_arm_rates = {}
        for arm_id, (acc, tot) in per_arm_annealing.items():
            per_arm_rates[arm_id] = acc / tot if tot > 0 else 0.0

        # Summarize temperature trajectory (last window_size values per arm)
        temp_summary = {}
        for arm_id, temps in temperature_trajectory.items():
            recent = temps[-window_size:]
            temp_summary[arm_id] = {
                "mean": sum(recent) / len(recent) if recent else 0.0,
                "last": recent[-1] if recent else 0.0,
                "count": len(temps),
            }

        return AcceptanceAnalysisReport(
            total_decisions=total,
            acceptance_rate=acceptance_rate,
            greedy_rate=greedy_rate,
            annealing_rate=annealing_rate,
            per_arm_annealing_rates=per_arm_rates,
            stepping_stone_rate=stepping_stone_rate,
            dead_end_rate=dead_end_rate,
            temperature_trajectory=temp_summary,
            surrogate_modulation_impact=surrogate_impact,
        )
