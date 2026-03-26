"""
EvolutionarySelectionController: runs one evolutionary generation.
Verifies and benchmarks mutations, applies elitism and crossover.
"""

import re
import uuid

from ..schemas import MutatedKernel, GenerationResult


class EvolutionarySelectionController:
    """Run one evolutionary generation on a set of kernel mutations.

    Applies:
      - Verification + benchmarking of each mutation.
      - Elitism: best mutation AND parent survive to next generation.
      - Crossover: if 2+ mutations pass targeting different aspects, combine them.
    """

    def __init__(self, verifier=None, benchmarker=None):
        """Initialize the selector.

        Args:
            verifier: Correctness verifier with .verify(source, ref) -> dict.
            benchmarker: Benchmarker with .benchmark(source, ref) -> dict.
        """
        self._verifier = verifier
        self._benchmarker = benchmarker

    def run_generation(
        self,
        parent: dict,
        mutations: list[MutatedKernel],
        reference_group_id: str,
        generation_number: int = 0,
    ) -> GenerationResult:
        """Run one evolutionary generation.

        Args:
            parent: Dict with 'kernel_id', 'kernel_source', 'speedup'.
            mutations: List of MutatedKernel candidates.
            reference_group_id: Group ID for benchmark reference.
            generation_number: Current generation index.

        Returns:
            GenerationResult summarizing the generation.
        """
        parent_id = parent.get("kernel_id", "")
        parent_speedup = parent.get("speedup", 1.0)

        # Verify and benchmark each mutation
        scored: list[tuple[MutatedKernel, float]] = []
        for mut in mutations:
            passed = self._verify(mut)
            if not passed:
                continue
            speedup = self._benchmark(mut)
            scored.append((mut, speedup))

        mutations_passed = len(scored)

        if not scored:
            return GenerationResult(
                generation=generation_number,
                parent_id=parent_id,
                mutations_tested=len(mutations),
                mutations_passed=0,
                best_mutation_id="",
                best_speedup=parent_speedup,
                improvement_over_parent=0.0,
            )

        # Sort by speedup descending
        scored.sort(key=lambda x: x[1], reverse=True)
        best_mut, best_speedup = scored[0]

        # Elitism: parent always survives. Best mutation also survives.
        improvement = best_speedup - parent_speedup

        # Crossover: if 2+ mutations pass targeting different aspects, combine
        crossover_attempted = False
        crossover_result: dict = {}
        if len(scored) >= 2:
            crossover_source = self._attempt_crossover(scored)
            if crossover_source is not None:
                crossover_attempted = True
                # Verify and benchmark the crossover
                cross_mut = MutatedKernel(
                    mutation_id=f"crossover_{uuid.uuid4().hex[:6]}",
                    parent_id=parent_id,
                    mutation_type="crossover",
                    mutation_description="Crossover of top mutations",
                    kernel_source=crossover_source,
                )
                cross_passed = self._verify(cross_mut)
                if cross_passed:
                    cross_speedup = self._benchmark(cross_mut)
                    crossover_result = {
                        "kernel_id": cross_mut.mutation_id,
                        "speedup": cross_speedup,
                        "passed": True,
                    }
                    # If crossover beats best mutation, use it
                    if cross_speedup > best_speedup:
                        best_mut = cross_mut
                        best_speedup = cross_speedup
                        improvement = best_speedup - parent_speedup
                else:
                    crossover_result = {"passed": False}

        return GenerationResult(
            generation=generation_number,
            parent_id=parent_id,
            mutations_tested=len(mutations),
            mutations_passed=mutations_passed,
            best_mutation_id=best_mut.mutation_id,
            best_speedup=best_speedup,
            improvement_over_parent=improvement,
            crossover_attempted=crossover_attempted,
            crossover_result=crossover_result,
        )

    def _verify(self, mutation: MutatedKernel) -> bool:
        """Verify a mutation's correctness."""
        if self._verifier is None:
            return True
        try:
            result = self._verifier.verify(mutation.kernel_source, "")
            if isinstance(result, dict):
                return result.get("passed", False)
            return getattr(result, "passed", False)
        except Exception:
            return False

    def _benchmark(self, mutation: MutatedKernel) -> float:
        """Benchmark a mutation and return speedup ratio."""
        if self._benchmarker is None:
            return 1.0
        try:
            result = self._benchmarker.benchmark(mutation.kernel_source, "")
            if isinstance(result, dict):
                return result.get("overall_speedup", 1.0)
            return getattr(result, "overall_speedup", 1.0)
        except Exception:
            return 1.0

    def _attempt_crossover(
        self, scored: list[tuple[MutatedKernel, float]]
    ) -> str | None:
        """Attempt crossover of top two mutations if they target different aspects.

        Returns combined source or None if crossover is not applicable.
        """
        if len(scored) < 2:
            return None

        mut_a, _ = scored[0]
        mut_b, _ = scored[1]

        # Only crossover if mutation types are different
        if mut_a.mutation_type == mut_b.mutation_type:
            return None

        # Simple crossover: apply mut_b's change on top of mut_a's source
        # by detecting what changed between parent and mut_b
        source_a = mut_a.kernel_source
        source_b = mut_b.kernel_source

        # Heuristic: if one changed block_size and other changed num_warps,
        # apply both changes to the parent
        if _targets_block_size(mut_a.mutation_type) and _targets_num_warps(mut_b.mutation_type):
            return _apply_num_warps_from(source_a, source_b)
        elif _targets_num_warps(mut_a.mutation_type) and _targets_block_size(mut_b.mutation_type):
            return _apply_block_size_from(source_a, source_b)

        return None


def _targets_block_size(mut_type: str) -> bool:
    return mut_type in ("block_size_double", "block_size_half")


def _targets_num_warps(mut_type: str) -> bool:
    return mut_type in ("num_warps_increase", "num_warps_decrease")


def _apply_num_warps_from(target_source: str, donor_source: str) -> str | None:
    """Copy num_warps value from donor into target source."""
    pattern = re.compile(r'num_warps\s*=\s*(\d+)')
    donor_match = pattern.search(donor_source)
    if not donor_match:
        return None
    new_val = donor_match.group(1)
    result = pattern.sub(f"num_warps={new_val}", target_source)
    return result if result != target_source else None


def _apply_block_size_from(target_source: str, donor_source: str) -> str | None:
    """Copy BLOCK_SIZE value from donor into target source."""
    pattern = re.compile(r'(BLOCK_SIZE\s*(?::\s*tl\.constexpr\s*)?=\s*)(\d+)')
    donor_match = pattern.search(donor_source)
    if not donor_match:
        return None
    new_val = donor_match.group(2)
    result = pattern.sub(lambda m: f"{m.group(1)}{new_val}", target_source)
    return result if result != target_source else None
