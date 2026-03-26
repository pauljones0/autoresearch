"""
DiscoveryLoopOrchestrator: runs a full autonomous kernel discovery cycle.
Rank opportunities -> generate variants -> verify -> benchmark -> select winner.
"""

from ..schemas import (
    DiscoveryCycleResult,
    KernelOpportunity,
    OperationProfile,
    FuseableGroup,
)
from gpu_kernels.discovery.opportunity_ranker import KernelOpportunityRanker
from gpu_kernels.discovery.autonomous_generator import AutonomousKernelGenerator


class DiscoveryLoopOrchestrator:
    """Full autonomous kernel discovery cycle.

    Steps:
      1. Rank opportunities from diagnostics and profiling data.
      2. Generate kernel variants for the top opportunity.
      3. Verify correctness of each variant.
      4. Benchmark passing variants.
      5. Select the winner and queue for integration.
    """

    def __init__(
        self,
        verifier=None,
        benchmarker=None,
        max_opportunities: int = 5,
    ):
        """Initialize the orchestrator.

        Args:
            verifier: Optional correctness verifier (must have .verify(kernel_source, reference_path) -> dict).
            benchmarker: Optional benchmarker (must have .benchmark(kernel_source, reference_path) -> dict).
            max_opportunities: Maximum number of opportunities to consider per cycle.
        """
        self._ranker = KernelOpportunityRanker()
        self._generator = AutonomousKernelGenerator()
        self._verifier = verifier
        self._benchmarker = benchmarker
        self._max_opportunities = max_opportunities

    def run_cycle(
        self,
        diagnostics_report: dict,
        kernel_config: dict,
        base_source: str,
        fuseable_groups: list | None = None,
        kernel_profiling: list | None = None,
        constraints: list[dict] | None = None,
    ) -> DiscoveryCycleResult:
        """Run one autonomous discovery cycle.

        Args:
            diagnostics_report: DiagnosticsReport as dict.
            kernel_config: Current kernel_config.json as dict.
            base_source: Training script source code.
            fuseable_groups: Pre-detected fuseable groups (list of FuseableGroup or dicts).
            kernel_profiling: Per-operation profiling data (list of OperationProfile or dicts).
            constraints: Optional negative constraints.

        Returns:
            DiscoveryCycleResult summarizing the cycle.
        """
        if fuseable_groups is None:
            fuseable_groups = []
        if kernel_profiling is None:
            kernel_profiling = []

        # Step 1: Rank opportunities
        opportunities = self._ranker.rank(
            fuseable_groups=fuseable_groups,
            kernel_profiling=kernel_profiling,
            kernel_config=kernel_config,
            constraints=constraints or [],
        )

        if not opportunities:
            return DiscoveryCycleResult(
                target_group_id="",
                variants_generated=0,
            )

        # Take the top opportunity
        top = opportunities[0]

        # Step 2: Generate kernel variants
        variants = self._generator.generate(
            opportunity=top,
            base_source=base_source,
        )

        if not variants:
            return DiscoveryCycleResult(
                target_group_id=top.group_id,
                variants_generated=0,
            )

        # Step 3: Verify correctness (pre-screen)
        passed_variants = []
        for variant in variants:
            if self._verifier is not None:
                try:
                    result = self._verifier.verify(
                        variant["kernel_source"],
                        variant.get("reference_path", ""),
                    )
                    if isinstance(result, dict) and result.get("passed", False):
                        passed_variants.append(variant)
                    elif hasattr(result, "passed") and result.passed:
                        passed_variants.append(variant)
                except Exception:
                    continue
            else:
                # No verifier: all syntactically valid variants pass pre-screen
                passed_variants.append(variant)

        # Step 4: Benchmark passing variants
        benchmarked = []
        for variant in passed_variants:
            if self._benchmarker is not None:
                try:
                    bench = self._benchmarker.benchmark(
                        variant["kernel_source"],
                        variant.get("reference_path", ""),
                    )
                    speedup = (
                        bench.get("overall_speedup", 0.0)
                        if isinstance(bench, dict)
                        else getattr(bench, "overall_speedup", 0.0)
                    )
                    benchmarked.append((variant, speedup))
                except Exception:
                    continue
            else:
                # No benchmarker: assign estimated speedup of 1.0
                benchmarked.append((variant, 1.0))

        # Step 5: Select winner
        if not benchmarked:
            return DiscoveryCycleResult(
                target_group_id=top.group_id,
                variants_generated=len(variants),
                variants_passed_prescreen=len(passed_variants),
                variants_passed_full_verification=0,
            )

        benchmarked.sort(key=lambda x: x[1], reverse=True)
        winner, winner_speedup = benchmarked[0]

        # Only queue if speedup > 1.0 (actually faster)
        queued = winner_speedup > 1.0

        return DiscoveryCycleResult(
            target_group_id=top.group_id,
            variants_generated=len(variants),
            variants_passed_prescreen=len(passed_variants),
            variants_passed_full_verification=len(benchmarked),
            winner_kernel_id=winner["kernel_id"],
            winner_speedup=winner_speedup,
            queued=queued,
        )
