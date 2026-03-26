"""
Phase 5 — DualCalibrationManager: maintain separate calibration thresholds
for internal vs paper-sourced candidates.

Paper-sourced techniques have higher implementation variance (synthetic diff
generation adds noise), so they get a looser threshold.
"""

import json
import os


class DualCalibrationManager:
    """Separate calibration for internal vs paper-sourced surrogate predictions.

    Default percentile thresholds:
        - internal: top 20% (only best internally-generated candidates evaluated)
        - paper: top 40% (looser, compensating for noisier predictions)
    """

    def __init__(
        self,
        internal_percentile: float = 0.80,
        paper_percentile: float = 0.60,
        save_path: str = None,
    ):
        self.internal_percentile = internal_percentile
        self.paper_percentile = paper_percentile
        self.internal_threshold: float = 0.0
        self.paper_threshold: float = 0.0
        self.save_path = save_path
        self._calibrated = False

        if save_path and os.path.exists(save_path):
            self._load(save_path)

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def calibrate(
        self,
        predictions: list,
        actuals: list,
        sources: list,
    ) -> dict:
        """Calibrate thresholds from historical predictions, actuals, and sources.

        Args:
            predictions: Surrogate predicted deltas.
            actuals: Actual measured deltas.
            sources: "internal" or "paper" for each entry.

        Returns:
            dict with internal_threshold, paper_threshold.
        """
        internal_vals = []
        paper_vals = []

        for pred, actual, src in zip(predictions, actuals, sources):
            if src == "internal":
                internal_vals.append(pred)
            else:
                paper_vals.append(pred)

        self.internal_threshold = self._percentile(
            internal_vals, self.internal_percentile
        ) if internal_vals else 0.0

        self.paper_threshold = self._percentile(
            paper_vals, self.paper_percentile
        ) if paper_vals else 0.0

        self._calibrated = True

        if self.save_path:
            self._save(self.save_path)

        return {
            "internal_threshold": self.internal_threshold,
            "paper_threshold": self.paper_threshold,
        }

    def get_threshold(self, source: str) -> float:
        """Return the threshold for the given source type."""
        if source == "internal":
            return self.internal_threshold
        return self.paper_threshold

    def should_evaluate(self, predicted_delta: float, source: str) -> bool:
        """Whether a candidate's predicted delta exceeds the calibrated threshold.

        Negative deltas are improvements (lower bpb), so we check
        predicted_delta <= -threshold (i.e., the improvement is large enough).
        If not yet calibrated, accept all candidates.
        """
        if not self._calibrated:
            return True
        threshold = self.get_threshold(source)
        # predicted_delta is negative for improvements; threshold is the
        # cutoff value at the desired percentile.  Candidates whose
        # predicted improvement is at least as good as threshold pass.
        return predicted_delta <= threshold

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _percentile(values: list, pct: float) -> float:
        """Compute the pct-th percentile of values (lower = better improvement)."""
        if not values:
            return 0.0
        sorted_vals = sorted(values)
        # We want the value below which `1 - pct` of values fall,
        # i.e., the top (1-pct) cutoff.  For "top 20%" we want the
        # value at the 20th-percentile (most negative).
        idx = int((1.0 - pct) * (len(sorted_vals) - 1))
        idx = max(0, min(idx, len(sorted_vals) - 1))
        return sorted_vals[idx]

    def _save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(
                {
                    "internal_threshold": self.internal_threshold,
                    "paper_threshold": self.paper_threshold,
                    "internal_percentile": self.internal_percentile,
                    "paper_percentile": self.paper_percentile,
                },
                f,
            )

    def _load(self, path: str) -> None:
        try:
            with open(path) as f:
                data = json.load(f)
            self.internal_threshold = data.get("internal_threshold", 0.0)
            self.paper_threshold = data.get("paper_threshold", 0.0)
            self._calibrated = True
        except (json.JSONDecodeError, OSError):
            pass
