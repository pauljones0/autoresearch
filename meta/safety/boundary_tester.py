"""
Boundary violation tester — adversarial tests for all 5 safety boundaries.
"""

import os
from meta.schemas import BoundaryViolationError, TestResult
from meta.safety.sandbox import MetaSandboxEnforcer
from meta.safety.recursion_guard import RecursionDepthGuard
from meta.safety.compute_budget import MetaComputeBudgetEnforcer
from meta.safety.eval_guard import EvaluationMetricGuard


class BoundaryViolationTester:
    """Adversarial tests for all 5 meta-loop safety boundaries."""

    def __init__(self, work_dir: str = "."):
        self.work_dir = work_dir

    def run_all_tests(self) -> list:
        results = []
        for test_fn in [self._test_train_py_write, self._test_eval_path_write,
                         self._test_recursion_depth, self._test_budget_exhaustion,
                         self._test_eval_hash_change]:
            try:
                results.append(test_fn())
            except Exception as e:
                results.append(TestResult(test_fn.__name__, False, "exception", str(e)))
        return results

    def _test_train_py_write(self) -> TestResult:
        sandbox = MetaSandboxEnforcer(self.work_dir)
        try:
            sandbox.check_write(os.path.join(self.work_dir, "train.py"))
            return TestResult("train_py_write", False, "no_error", "Should have raised")
        except BoundaryViolationError:
            return TestResult("train_py_write", True, "BoundaryViolationError", "Blocked")

    def _test_eval_path_write(self) -> TestResult:
        sandbox = MetaSandboxEnforcer(self.work_dir)
        try:
            sandbox.check_write(os.path.join(self.work_dir, "eval_config.json"))
            return TestResult("eval_path_write", False, "no_error", "Should have raised")
        except BoundaryViolationError:
            return TestResult("eval_path_write", True, "BoundaryViolationError", "Blocked")

    def _test_recursion_depth(self) -> TestResult:
        guard = RecursionDepthGuard()
        old_val = os.environ.get("META_RECURSION_DEPTH", "")
        try:
            os.environ["META_RECURSION_DEPTH"] = "1"
            guard.check_depth()
            return TestResult("recursion_depth", False, "no_error", "Should have raised")
        except BoundaryViolationError:
            return TestResult("recursion_depth", True, "BoundaryViolationError", "Blocked")
        finally:
            if old_val:
                os.environ["META_RECURSION_DEPTH"] = old_val
            else:
                os.environ.pop("META_RECURSION_DEPTH", None)

    def _test_budget_exhaustion(self) -> TestResult:
        enforcer = MetaComputeBudgetEnforcer(budget_fraction=0.2)
        can_run = enforcer.can_run_meta_experiment(100, 20)
        if can_run:
            return TestResult("budget_exhaustion", False, "allowed", "Should have rejected")
        return TestResult("budget_exhaustion", True, "rejected", "Correctly rejected")

    def _test_eval_hash_change(self) -> TestResult:
        guard = EvaluationMetricGuard(train_path=os.path.join(self.work_dir, "train.py"))
        guard.initialize()
        guard._stored_hash = "fake_hash_that_wont_match"
        try:
            guard.verify_evaluation_unchanged()
            return TestResult("eval_hash_change", False, "no_error", "Should have raised")
        except BoundaryViolationError:
            return TestResult("eval_hash_change", True, "BoundaryViolationError", "Detected")
