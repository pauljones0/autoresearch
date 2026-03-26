"""Execute generated strategies by monkey-patching hooks into pipelines."""

import time
import traceback

from meta.schemas import (
    GeneratedStrategy,
    MetaContext,
    MetaExperimentResult,
    SafetyCheckResult,
)
from meta.stop.safety_checker import StrategySafetyChecker


class StrategyExecutor:
    """Compiles and executes a generated strategy via monkey-patching."""

    def __init__(self):
        self._safety_checker = StrategySafetyChecker()

    def execute(
        self,
        strategy: GeneratedStrategy,
        context: MetaContext,
        experiment_length: int = 50,
    ) -> MetaExperimentResult:
        """Execute a strategy and return experimental results.

        Steps:
        1. Safety-check the strategy code.
        2. Compile the code into a callable hook.
        3. Monkey-patch the hook into the target pipeline.
        4. Run iterations, collecting deltas.
        5. Restore the original hook.
        """
        experiment_id = f"exec_{strategy.strategy_id}_{int(time.time())}"

        # Step 1: safety check
        safety_result = self._safety_checker.check(strategy)
        if not safety_result.safe:
            return MetaExperimentResult(
                experiment_id=experiment_id,
                config_diff=[],
                n_iterations=0,
                improvement_rate=0.0,
                compared_to_baseline="inconclusive",
                raw_deltas=[],
            )

        # Step 2: compile to callable
        hook_fn = self._compile_hook(strategy)
        if hook_fn is None:
            return MetaExperimentResult(
                experiment_id=experiment_id,
                config_diff=[],
                n_iterations=0,
                improvement_rate=0.0,
                compared_to_baseline="inconclusive",
                raw_deltas=[],
            )

        # Step 3: identify target and patch
        target_pipeline = self._get_target_pipeline(strategy.hook_type, context)
        original_hook = None
        hook_attr = strategy.hook_type

        if target_pipeline is not None and hasattr(target_pipeline, hook_attr):
            original_hook = getattr(target_pipeline, hook_attr)
            setattr(target_pipeline, hook_attr, hook_fn)

        # Step 4: run iterations
        deltas = []
        improvements = 0
        try:
            for i in range(experiment_length):
                delta = self._run_single_iteration(
                    target_pipeline, hook_fn, strategy.hook_type, i
                )
                deltas.append(delta)
                if delta > 0:
                    improvements += 1
        except Exception:
            traceback.print_exc()
        finally:
            # Step 5: restore original
            if target_pipeline is not None and original_hook is not None:
                setattr(target_pipeline, hook_attr, original_hook)

        n = len(deltas)
        ir = improvements / n if n > 0 else 0.0

        return MetaExperimentResult(
            experiment_id=experiment_id,
            config_diff=[{"param_id": strategy.hook_type, "old_value": "default", "new_value": strategy.strategy_id}],
            n_iterations=n,
            improvement_rate=ir,
            compared_to_baseline="inconclusive",
            raw_deltas=deltas,
        )

    def _compile_hook(self, strategy: GeneratedStrategy):
        """Compile strategy code and extract the hook function."""
        namespace = {}
        try:
            code_obj = compile(strategy.code, f"<strategy:{strategy.strategy_id}>", "exec")
            exec(code_obj, namespace)  # noqa: S102 - intentional; guarded by safety checker
        except Exception:
            traceback.print_exc()
            return None

        # Find the function matching the hook type
        hook_fn = namespace.get(strategy.hook_type)
        if hook_fn is None:
            # Try finding any callable
            for name, obj in namespace.items():
                if callable(obj) and not name.startswith("_"):
                    hook_fn = obj
                    break
        return hook_fn

    @staticmethod
    def _get_target_pipeline(hook_type: str, context: MetaContext):
        """Return the pipeline object that owns the given hook type."""
        # All hook types currently target the bandit pipeline
        if context.bandit_pipeline is not None:
            return context.bandit_pipeline
        if context.model_scientist_pipeline is not None:
            return context.model_scientist_pipeline
        return None

    @staticmethod
    def _run_single_iteration(pipeline, hook_fn, hook_type: str, iteration: int) -> float:
        """Run one iteration with the patched hook.

        If no pipeline is available, call the hook with synthetic data
        to measure that it runs without error and return a simulated delta.
        """
        if pipeline is not None and hasattr(pipeline, "run_iteration"):
            try:
                result = pipeline.run_iteration()
                if isinstance(result, (int, float)):
                    return float(result)
                if hasattr(result, "delta"):
                    return float(result.delta)
            except Exception:
                pass
            return 0.0

        # Fallback: synthetic call to verify the hook works
        try:
            if hook_type == "selection_hook":
                hook_fn(candidates=["a", "b", "c"], scores=[0.1, 0.5, 0.3], iteration=iteration)
            elif hook_type == "acceptance_hook":
                hook_fn(delta=0.01, iteration=iteration)
            elif hook_type == "prompt_hook":
                hook_fn(base_prompt="test prompt", context={})
            elif hook_type == "scheduling_hook":
                hook_fn(dimensions=["d1", "d2"], scores=[0.5, 0.3], iteration=iteration)
        except Exception:
            return 0.0
        return 0.0
