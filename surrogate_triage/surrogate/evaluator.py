"""
Phase 2 — SurrogateEvaluator: evaluate surrogate model accuracy
using rank correlation, MAE, and worst-case analysis.
"""

import math


def _rank(values: list[float]) -> list[float]:
    """Assign ranks to values (1-based, average ties)."""
    n = len(values)
    indexed = sorted(range(n), key=lambda i: values[i])
    ranks = [0.0] * n

    i = 0
    while i < n:
        # Find all tied values
        j = i + 1
        while j < n and values[indexed[j]] == values[indexed[i]]:
            j += 1
        # Average rank for tied group
        avg_rank = (i + j + 1) / 2.0  # 1-based average
        for k in range(i, j):
            ranks[indexed[k]] = avg_rank
        i = j

    return ranks


def _pearson(x: list[float], y: list[float]) -> float:
    """Pearson correlation coefficient."""
    n = len(x)
    if n < 2:
        return 0.0

    mx = sum(x) / n
    my = sum(y) / n

    num = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y))
    dx = math.sqrt(sum((xi - mx) ** 2 for xi in x))
    dy = math.sqrt(sum((yi - my) ** 2 for yi in y))

    if dx < 1e-12 or dy < 1e-12:
        return 0.0
    return num / (dx * dy)


def _spearman(predictions: list[float], actuals: list[float]) -> float:
    """Spearman rank correlation (implemented from scratch)."""
    if len(predictions) < 2:
        return 0.0
    pred_ranks = _rank(predictions)
    actual_ranks = _rank(actuals)
    return _pearson(pred_ranks, actual_ranks)


class SurrogateEvaluator:
    """Evaluate surrogate model predictions against actual outcomes."""

    def evaluate(
        self,
        predictions: list[float],
        actuals: list[float],
        sources: list[str] | None = None,
    ) -> dict:
        """Compute all evaluation metrics.

        Args:
            predictions: Surrogate-predicted delta values.
            actuals: Actual observed delta values.
            sources: Optional list of "internal" or "paper" labels per example.

        Returns:
            Dict with spearman_r, mae, worst_case_rate, and per-source metrics.
        """
        n = len(predictions)
        if n == 0:
            return {
                "n": 0,
                "spearman_r": 0.0,
                "mae": 0.0,
                "worst_case_rate": 0.0,
            }

        if len(actuals) != n:
            raise ValueError(
                f"predictions ({n}) and actuals ({len(actuals)}) must have the same length"
            )

        # Spearman rank correlation
        spearman_r = _spearman(predictions, actuals)

        # Mean absolute error
        mae = sum(abs(p - a) for p, a in zip(predictions, actuals)) / n

        # Worst-case analysis: how often does a bad modification rank in top-5?
        # "bad" = actual_delta > 0 (regression)
        worst_case_rate = self._worst_case_rate(predictions, actuals, top_k=5)

        result = {
            "n": n,
            "spearman_r": spearman_r,
            "mae": mae,
            "worst_case_rate": worst_case_rate,
        }

        # Per-source accuracy
        if sources and len(sources) == n:
            for source_type in ("internal", "paper"):
                idxs = [i for i, s in enumerate(sources) if s == source_type]
                if idxs:
                    src_preds = [predictions[i] for i in idxs]
                    src_acts = [actuals[i] for i in idxs]
                    result[f"{source_type}_spearman_r"] = _spearman(src_preds, src_acts)
                    result[f"{source_type}_mae"] = (
                        sum(abs(p - a) for p, a in zip(src_preds, src_acts)) / len(idxs)
                    )
                    result[f"{source_type}_n"] = len(idxs)

        return result

    def calibration_data(
        self,
        predictions: list[float],
        actuals: list[float],
    ) -> list[tuple[float, float]]:
        """Return (predicted, actual) pairs for calibration plotting.

        Args:
            predictions: Surrogate-predicted delta values.
            actuals: Actual observed delta values.

        Returns:
            List of (predicted, actual) tuples sorted by predicted value.
        """
        pairs = list(zip(predictions, actuals))
        pairs.sort(key=lambda pa: pa[0])
        return pairs

    @staticmethod
    def _worst_case_rate(
        predictions: list[float],
        actuals: list[float],
        top_k: int = 5,
    ) -> float:
        """Fraction of top-k predicted candidates that are actually bad.

        A 'bad' modification is one with actual_delta > 0 (regression).
        """
        n = len(predictions)
        if n == 0:
            return 0.0

        k = min(top_k, n)
        # Sort by predicted delta ascending (most negative = best predicted improvement)
        indexed = sorted(range(n), key=lambda i: predictions[i])
        top_k_indices = indexed[:k]

        bad_count = sum(1 for i in top_k_indices if actuals[i] > 0)
        return bad_count / k
