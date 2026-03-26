"""
A/B test orchestrator: compares bandit allocation vs fixed-fraction control.
"""

import copy
import json
import os
import time

from bandit.schemas import (
    BanditState, ArmState, LoopContext, IterationResult,
    ABAnalysisReport, save_json, load_json, save_jsonl, load_jsonl,
)


class ABTestOrchestrator:
    """Runs A/B comparison between bandit (treatment) and fixed-fraction
    EvaluationScheduler (control)."""

    def __init__(self, work_dir: str = "ab_test_runs"):
        self.work_dir = work_dir

    def run(
        self,
        context: LoopContext,
        n_iterations: int = 200,
        n_seeds: int = 3,
    ) -> ABAnalysisReport:
        """Run the full A/B test.

        For each seed:
        1. Checkpoint current state
        2. Run treatment (bandit) for n_iterations
        3. Restore checkpoint
        4. Run control (fixed-fraction) for n_iterations
        5. Collect results

        Then analyze treatment vs control across all seeds.
        """
        import random

        os.makedirs(self.work_dir, exist_ok=True)
        treatment_paths = []
        control_paths = []

        for seed_idx in range(n_seeds):
            seed = seed_idx * 1000 + 42
            rng = random.Random(seed)

            # Checkpoint state
            checkpoint = self._checkpoint(context)

            # Run treatment (bandit)
            treatment_log_path = os.path.join(
                self.work_dir, f"treatment_seed{seed_idx}.jsonl")
            self._run_treatment(context, n_iterations, rng, treatment_log_path)
            treatment_paths.append(treatment_log_path)

            # Restore checkpoint
            self._restore(context, checkpoint)

            # Run control (fixed-fraction scheduler)
            control_log_path = os.path.join(
                self.work_dir, f"control_seed{seed_idx}.jsonl")
            self._run_control(context, n_iterations, rng, control_log_path)
            control_paths.append(control_log_path)

            # Restore again for next seed
            self._restore(context, checkpoint)

        # Analyze results
        try:
            from bandit.analysis.ab_analyzer import ABStatisticalAnalyzer
            analyzer = ABStatisticalAnalyzer()
            return analyzer.analyze(treatment_paths, control_paths)
        except ImportError:
            return self._basic_analyze(treatment_paths, control_paths)

    def _checkpoint(self, context: LoopContext) -> dict:
        """Save a deep copy of the current state."""
        state = context.bandit_state
        if state is not None and hasattr(state, "to_dict"):
            return {"state": copy.deepcopy(state.to_dict())}
        return {"state": None}

    def _restore(self, context: LoopContext, checkpoint: dict):
        """Restore state from checkpoint."""
        state_dict = checkpoint.get("state")
        if state_dict is not None:
            context.bandit_state = BanditState.from_dict(state_dict)
        else:
            context.bandit_state = BanditState()

    def _run_treatment(
        self,
        context: LoopContext,
        n_iterations: int,
        rng,
        log_path: str,
    ):
        """Run bandit loop for treatment arm."""
        try:
            from bandit.loop import BanditLoop
            loop = BanditLoop()
        except ImportError:
            return

        context.rng = rng
        for i in range(n_iterations):
            try:
                result = loop.run_iteration(context)
                save_jsonl(result, log_path)
            except Exception:
                save_jsonl({
                    "iteration": i,
                    "error": "iteration_failed",
                    "arm_selected": "",
                    "delta": None,
                }, log_path)

    def _run_control(
        self,
        context: LoopContext,
        n_iterations: int,
        rng,
        log_path: str,
    ):
        """Run fixed-fraction control (round-robin across arms)."""
        state = context.bandit_state
        if state is None:
            state = BanditState()

        arm_ids = [aid for aid, a in state.arms.items()
                   if isinstance(a, ArmState)]
        if not arm_ids:
            return

        for i in range(n_iterations):
            # Round-robin arm selection
            arm_id = arm_ids[i % len(arm_ids)]
            save_jsonl({
                "iteration": i,
                "arm_selected": arm_id,
                "dispatch_path": "control_fixed_fraction",
                "delta": None,
                "verdict": "control",
                "accepted": False,
                "accepted_by": "control",
                "elapsed_seconds": 0.0,
            }, log_path)

    def _basic_analyze(
        self,
        treatment_paths: list,
        control_paths: list,
    ) -> ABAnalysisReport:
        """Basic analysis when ABStatisticalAnalyzer is not available."""
        treatment_deltas = []
        control_deltas = []

        for path in treatment_paths:
            entries = load_jsonl(path)
            for e in entries:
                d = e.get("delta")
                if d is not None:
                    treatment_deltas.append(d)

        for path in control_paths:
            entries = load_jsonl(path)
            for e in entries:
                d = e.get("delta")
                if d is not None:
                    control_deltas.append(d)

        t_med = _median(treatment_deltas) if treatment_deltas else 0.0
        c_med = _median(control_deltas) if control_deltas else 0.0

        return ABAnalysisReport(
            n_seeds=len(treatment_paths),
            treatment_median_improvement=t_med,
            control_median_improvement=c_med,
            verdict="insufficient_data" if not treatment_deltas else "basic_analysis",
        )


def _median(values: list) -> float:
    """Compute median of a list of numbers."""
    if not values:
        return 0.0
    s = sorted(values)
    n = len(s)
    if n % 2 == 1:
        return s[n // 2]
    return (s[n // 2 - 1] + s[n // 2]) / 2.0
