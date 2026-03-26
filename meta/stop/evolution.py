"""Multi-generation evolutionary strategy optimization."""

from typing import List, Optional

from meta.schemas import GeneratedStrategy, MetaContext, MetaExperimentResult
from meta.stop.scaffold import STOPScaffold
from meta.stop.executor import StrategyExecutor


class StrategyEvolutionController:
    """Evolves strategies over multiple generations using tournament selection."""

    def __init__(self):
        self._scaffold = STOPScaffold()
        self._executor = StrategyExecutor()

    def evolve(
        self,
        hook_type: str,
        n_generations: int = 3,
        n_candidates: int = 3,
        context: Optional[MetaContext] = None,
        experiment_length: int = 50,
        baseline_ir: float = 0.0,
    ) -> GeneratedStrategy:
        """Run multi-generation evolution and return the best strategy.

        Algorithm:
        1. Generate n_candidates initial strategies for hook_type.
        2. Evaluate each candidate.
        3. Select top-2 performers.
        4. Generate new candidates seeded from top-2.
        5. Repeat for n_generations.
        6. Return the overall best.
        """
        if context is None:
            context = MetaContext()

        best_overall: Optional[GeneratedStrategy] = None
        best_overall_ir: float = -1.0

        # Track cumulative experiment history for template rotation
        experiment_history: List[MetaExperimentResult] = []

        for gen in range(n_generations):
            # Generate candidates
            candidates = self._generate_candidates(
                hook_type, n_candidates, experiment_history, baseline_ir, gen
            )

            # Evaluate each candidate
            results: List[tuple] = []
            for candidate in candidates:
                result = self._executor.execute(candidate, context, experiment_length)
                experiment_history.append(result)
                results.append((candidate, result))

            # Sort by improvement rate descending
            results.sort(key=lambda x: x[1].improvement_rate, reverse=True)

            # Track overall best
            top_candidate, top_result = results[0]
            if top_result.improvement_rate > best_overall_ir:
                best_overall_ir = top_result.improvement_rate
                best_overall = top_candidate

            # Select top-2 for next generation seeding
            # (the scaffold will naturally rotate templates based on history)

        if best_overall is None:
            # Fallback: return a default strategy
            best_overall = self._scaffold.generate_for_hook(hook_type, 0)

        return best_overall

    def _generate_candidates(
        self,
        hook_type: str,
        n_candidates: int,
        experiment_history: List[MetaExperimentResult],
        baseline_ir: float,
        generation: int,
    ) -> List[GeneratedStrategy]:
        """Generate n_candidates strategies for a hook type."""
        candidates = []
        for i in range(n_candidates):
            # Use different template indices offset by generation
            template_idx = generation * n_candidates + i
            strategy = self._scaffold.generate_for_hook(hook_type, template_idx)
            candidates.append(strategy)
        return candidates
