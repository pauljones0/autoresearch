"""
EvolutionConvergenceDetector: determines when evolutionary refinement
should stop based on improvement stagnation and generation limits.
"""

from ..schemas import GenerationResult


class EvolutionConvergenceDetector:
    """Detect when evolutionary refinement has converged.

    Stop conditions:
      1. No improvement in 2 consecutive generations.
      2. Maximum generations reached (default 10).
      3. All mutations in a generation fail verification.
      4. Improvement < 1% for 3 consecutive generations.
    """

    def __init__(self, max_generations: int = 10, min_improvement_pct: float = 0.01):
        """Initialize the convergence detector.

        Args:
            max_generations: Maximum number of generations before forced stop.
            min_improvement_pct: Minimum improvement fraction to count as progress.
        """
        self._max_generations = max_generations
        self._min_improvement_pct = min_improvement_pct

    def should_stop(
        self, generation_results: list[GenerationResult]
    ) -> tuple[bool, str]:
        """Determine whether evolution should stop.

        Args:
            generation_results: List of GenerationResult from all generations so far.

        Returns:
            Tuple of (should_stop, reason).
        """
        if not generation_results:
            return False, ""

        n = len(generation_results)

        # Normalize: accept dicts or GenerationResult objects
        results = [_normalize(r) for r in generation_results]

        # Condition 2: Max generations reached
        if n >= self._max_generations:
            return True, f"Maximum generations ({self._max_generations}) reached"

        # Condition 3: All mutations failed in latest generation
        latest = results[-1]
        if latest.mutations_tested > 0 and latest.mutations_passed == 0:
            return True, "All mutations failed verification in latest generation"

        # Condition 1: No improvement in 2 consecutive generations
        if n >= 2:
            no_improvement_count = 0
            for r in reversed(results):
                if r.improvement_over_parent <= 0:
                    no_improvement_count += 1
                else:
                    break
            if no_improvement_count >= 2:
                return True, "No improvement in 2 consecutive generations"

        # Condition 4: Improvement < 1% for 3 consecutive generations
        if n >= 3:
            marginal_count = 0
            for r in reversed(results):
                # improvement_over_parent is absolute speedup delta
                # compute relative improvement
                parent_speedup = r.best_speedup - r.improvement_over_parent
                if parent_speedup > 0:
                    relative_improvement = r.improvement_over_parent / parent_speedup
                else:
                    relative_improvement = r.improvement_over_parent

                if 0 < relative_improvement < self._min_improvement_pct:
                    marginal_count += 1
                elif relative_improvement <= 0:
                    marginal_count += 1
                else:
                    break

            if marginal_count >= 3:
                return True, "Improvement < 1% for 3 consecutive generations"

        return False, ""


def _normalize(r) -> GenerationResult:
    """Convert a dict to GenerationResult if needed."""
    if isinstance(r, GenerationResult):
        return r
    if isinstance(r, dict):
        gr = GenerationResult()
        for k, v in r.items():
            if hasattr(gr, k):
                setattr(gr, k, v)
        return gr
    return r
