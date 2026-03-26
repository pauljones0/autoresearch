"""
Baseline run orchestrator — run independent baseline campaigns.
"""

import time
from meta.schemas import BaselineResult


class BaselineRunOrchestrator:
    """Run independent baseline campaigns for calibration."""

    def __init__(self, work_dir: str = "."):
        self.work_dir = work_dir

    def run_baselines(self, n_runs: int = 3, n_iterations: int = 100,
                      seeds: list = None) -> list:
        if seeds is None:
            seeds = [42, 123, 456]

        results = []
        for seed in seeds[:n_runs]:
            result = self._run_single(seed, n_iterations)
            results.append(result)
        return results

    def _run_single(self, seed: int, n_iterations: int) -> BaselineResult:
        """Run a single baseline campaign.

        In production this calls AdaptiveBanditPipeline.run_iteration()
        n_iterations times with checkpoint/restore. Here we provide the
        structure for integration.
        """
        raise NotImplementedError("Baseline implementation requires calling AdaptiveBanditPipeline.run_iteration()")
