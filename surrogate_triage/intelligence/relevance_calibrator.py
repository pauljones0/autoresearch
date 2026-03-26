"""
RelevanceCalibrator: calibrates the diagnostics boost coefficient so it
does not overwhelm the keyword score. Uses retrospective analysis of
historical paper evaluations.
"""

import json
import math
import os


class RelevanceCalibrator:
    """Calibrates the diagnostics-to-relevance boost coefficient."""

    DEFAULT_BOOST_COEFFICIENT = 0.5

    def __init__(self, calibration_path: str = ""):
        self._calibration_path = calibration_path
        self.boost_coefficient = self.DEFAULT_BOOST_COEFFICIENT
        if calibration_path:
            self._load_calibration(calibration_path)

    def _load_calibration(self, path: str):
        """Load previously stored calibration."""
        try:
            with open(path) as f:
                data = json.load(f)
            self.boost_coefficient = data.get("boost_coefficient", self.DEFAULT_BOOST_COEFFICIENT)
        except (FileNotFoundError, json.JSONDecodeError):
            pass

    def calibrate(self, paper_evaluations: list, diagnostics_history: list) -> float:
        """Find the optimal boost coefficient from historical data.

        Args:
            paper_evaluations: list of dicts with keys:
                paper_id, keyword_score, diagnostics_boost, outcome (float, positive = good)
            diagnostics_history: list of diagnostics report dicts (for context, used to
                re-compute boosts if needed)

        Returns:
            Optimal boost coefficient (float).
        """
        if not paper_evaluations:
            return self.boost_coefficient

        # Grid search over candidate coefficients
        candidates = [i * 0.1 for i in range(1, 11)]  # 0.1 to 1.0
        best_coeff = self.boost_coefficient
        best_ndcg = -1.0

        for coeff in candidates:
            baseline_rankings = []
            boosted_rankings = []
            outcomes = []

            for ev in paper_evaluations:
                kw = ev.get("keyword_score", 0.0)
                diag = ev.get("diagnostics_boost", 0.0)
                outcome = ev.get("outcome", 0.0)
                baseline_rankings.append(kw)
                boosted_rankings.append(kw + coeff * diag)
                outcomes.append(outcome)

            improvement = self.compute_ranking_improvement(
                baseline_rankings, boosted_rankings, outcomes
            )
            if improvement > best_ndcg:
                best_ndcg = improvement
                best_coeff = coeff

        self.boost_coefficient = best_coeff
        self._save_calibration()
        return best_coeff

    def compute_ranking_improvement(
        self, baseline_rankings: list, boosted_rankings: list, outcomes: list
    ) -> float:
        """Compute improvement in ranking quality (NDCG-like metric).

        Returns the difference: ndcg(boosted) - ndcg(baseline).
        Positive means boosted ranking is better.
        """
        if not outcomes:
            return 0.0

        ndcg_base = self._ndcg(baseline_rankings, outcomes)
        ndcg_boost = self._ndcg(boosted_rankings, outcomes)
        return ndcg_boost - ndcg_base

    @staticmethod
    def _ndcg(scores: list, relevances: list) -> float:
        """Compute Normalized Discounted Cumulative Gain.

        Ranks items by scores descending, then evaluates using relevances.
        """
        if not scores or not relevances:
            return 0.0

        n = len(scores)
        # Get ranking by scores (descending)
        indexed = list(range(n))
        indexed.sort(key=lambda i: scores[i], reverse=True)

        dcg = 0.0
        for rank, idx in enumerate(indexed):
            rel = max(relevances[idx], 0.0)
            dcg += rel / math.log2(rank + 2)  # rank+2 because rank is 0-indexed

        # Ideal DCG: sort by true relevance
        ideal_rels = sorted(relevances, reverse=True)
        idcg = 0.0
        for rank, rel in enumerate(ideal_rels):
            rel = max(rel, 0.0)
            idcg += rel / math.log2(rank + 2)

        if idcg == 0.0:
            return 0.0
        return dcg / idcg

    def _save_calibration(self):
        """Persist calibration to file."""
        if not self._calibration_path:
            return
        try:
            parent = os.path.dirname(self._calibration_path)
            if parent:
                os.makedirs(parent, exist_ok=True)
            with open(self._calibration_path, "w") as f:
                json.dump({"boost_coefficient": self.boost_coefficient}, f, indent=2)
        except OSError:
            pass
