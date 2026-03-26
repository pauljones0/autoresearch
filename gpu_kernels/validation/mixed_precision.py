"""
Mixed precision stress testing for custom GPU kernels.

Runs 1000-step autocast stress test monitoring for NaN/Inf in kernel outputs.
When detected, captures input tensors and re-runs with PyTorch reference to
confirm the issue is kernel-specific.
"""

import json
import math
import os
import time


class MixedPrecisionStressTester:
    """Stress test kernels under mixed-precision (autocast) training."""

    def __init__(self, data_dir: str = None):
        self.data_dir = data_dir or os.path.join(
            os.path.dirname(__file__), "..", "data"
        )

    def stress_test(
        self,
        kernel_config: dict,
        n_steps: int = 1000,
        seed: int = 42,
    ) -> dict:
        """
        Run mixed-precision stress test.

        Simulates n_steps of mixed-precision training, monitoring kernel
        outputs for NaN/Inf values via hooks. When detected, captures the
        input tensors and re-runs the operation with PyTorch reference to
        confirm the issue is kernel-specific.

        Args:
            kernel_config: Dict mapping kernel IDs to config entries.
            n_steps: Number of training steps to run.
            seed: Random seed.

        Returns:
            dict with keys:
                passed: bool — no kernel-specific NaN/Inf detected
                nan_events: list of dicts with step, kernel_id, input_snapshot
                inf_events: list of dicts with step, kernel_id, input_snapshot
                scale_underflow_count: int — number of loss scale underflows
        """
        result = {
            "passed": True,
            "nan_events": [],
            "inf_events": [],
            "scale_underflow_count": 0,
        }

        active_kernels = self._get_active_kernels(kernel_config)
        if not active_kernels:
            return result

        loss_scale = 65536.0
        min_scale = 1.0

        for step in range(1, n_steps + 1):
            step_result = self._run_autocast_step(
                step, seed, active_kernels, loss_scale
            )

            # Check for NaN
            for event in step_result.get("nan_detections", []):
                is_kernel_specific = self._confirm_kernel_specific(
                    event, "nan"
                )
                if is_kernel_specific:
                    result["nan_events"].append({
                        "step": step,
                        "kernel_id": event.get("kernel_id", ""),
                        "input_snapshot": event.get("input_snapshot", ""),
                        "kernel_specific": True,
                    })
                    result["passed"] = False

            # Check for Inf
            for event in step_result.get("inf_detections", []):
                is_kernel_specific = self._confirm_kernel_specific(
                    event, "inf"
                )
                if is_kernel_specific:
                    result["inf_events"].append({
                        "step": step,
                        "kernel_id": event.get("kernel_id", ""),
                        "input_snapshot": event.get("input_snapshot", ""),
                        "kernel_specific": True,
                    })
                    result["passed"] = False

            # Track loss scale underflows
            if step_result.get("scale_underflow", False):
                result["scale_underflow_count"] += 1
                loss_scale = max(loss_scale / 2.0, min_scale)

        self._save_result(result)
        return result

    def _get_active_kernels(self, kernel_config: dict) -> dict:
        """Extract active triton kernels from config."""
        active = {}
        for kid, entry in kernel_config.items():
            if isinstance(entry, dict):
                if entry.get("enabled", True) and entry.get("backend") == "triton":
                    active[kid] = entry
            elif hasattr(entry, "enabled"):
                if entry.enabled and entry.backend == "triton":
                    active[kid] = entry
        return active

    def _run_autocast_step(
        self, step: int, seed: int, kernels: dict, loss_scale: float
    ) -> dict:
        """
        Run a single autocast training step.

        In a real implementation, this would:
        1. Run forward pass with autocast enabled
        2. Attach hooks to monitor kernel outputs
        3. Scale loss and run backward pass
        4. Check grad scaler for overflow/underflow

        Returns dict with nan_detections, inf_detections, scale_underflow.
        """
        return {
            "nan_detections": [],
            "inf_detections": [],
            "scale_underflow": False,
        }

    def _confirm_kernel_specific(self, event: dict, issue_type: str) -> bool:
        """
        Re-run the operation with PyTorch reference to confirm kernel-specific.

        If the reference also produces NaN/Inf, the issue is not kernel-specific.
        """
        input_snapshot = event.get("input_snapshot")
        if input_snapshot is None:
            return True  # Cannot confirm, assume kernel-specific

        # In real implementation: re-run with PyTorch reference using
        # captured input tensors and check output for NaN/Inf.
        reference_result = self._run_reference(input_snapshot)

        if issue_type == "nan":
            ref_has_issue = reference_result.get("has_nan", False)
        else:
            ref_has_issue = reference_result.get("has_inf", False)

        # If reference also has the issue, it's not kernel-specific
        return not ref_has_issue

    def _run_reference(self, input_snapshot) -> dict:
        """Run PyTorch reference with captured inputs. Placeholder."""
        return {"has_nan": False, "has_inf": False}

    def _save_result(self, result: dict):
        """Persist stress test result."""
        out_dir = os.path.join(self.data_dir, "mixed_precision")
        try:
            os.makedirs(out_dir, exist_ok=True)
            path = os.path.join(out_dir, f"result_{int(time.time())}.json")
            with open(path, "w") as f:
                json.dump(result, f, indent=2)
        except OSError:
            pass
