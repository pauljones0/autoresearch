"""
Fallback integrity verification after kernel disable.

After a kernel is disabled and PyTorch fallback is activated, verifies that
the fallback path is functional and produces expected loss trajectories.
"""

import json
import os
import time


class FallbackIntegrityVerifier:
    """Verify PyTorch fallback functions correctly after kernel disable."""

    def __init__(self, data_dir: str = None):
        self.data_dir = data_dir or os.path.join(
            os.path.dirname(__file__), "..", "data"
        )

    def verify_fallback(
        self,
        kernel_id: str,
        kernel_config: dict,
        n_steps: int = 10,
        loss_tolerance: float = 0.1,
    ) -> dict:
        """
        Verify that PyTorch fallback is functional after kernel disable.

        Runs n_steps with all kernels except kernel_id using PyTorch backend,
        checks for errors, and compares loss trajectory against baseline.

        Args:
            kernel_id: The kernel that was disabled.
            kernel_config: Full kernel config dict.
            n_steps: Number of steps to run for verification.
            loss_tolerance: Max relative loss deviation from baseline.

        Returns:
            dict with keys:
                fallback_functional: bool — no errors during fallback run
                loss_within_baseline: bool — loss trajectory is acceptable
                errors: list of error strings encountered
        """
        result = {
            "fallback_functional": True,
            "loss_within_baseline": True,
            "errors": [],
        }

        # Build fallback config — disabled kernel uses pytorch backend
        fallback_config = {}
        for kid, entry in kernel_config.items():
            if isinstance(entry, dict):
                fc = dict(entry)
            elif hasattr(entry, "to_dict"):
                fc = entry.to_dict()
            else:
                fc = {"enabled": False}
            if kid == kernel_id:
                fc["enabled"] = False
                fc["backend"] = "pytorch"
            fallback_config[kid] = fc

        # Run fallback steps
        losses = []
        for step in range(1, n_steps + 1):
            try:
                step_result = self._run_fallback_step(step, fallback_config)
                losses.append(step_result.get("loss", 0.0))
            except Exception as e:
                result["fallback_functional"] = False
                result["errors"].append(f"Step {step}: {str(e)}")

        if result["errors"]:
            result["fallback_functional"] = False

        # Compare loss trajectory against baseline
        baseline_losses = self._load_baseline_losses(kernel_id)
        if baseline_losses and losses:
            loss_ok = self._compare_loss_trajectory(
                losses, baseline_losses, loss_tolerance
            )
            result["loss_within_baseline"] = loss_ok

        self._save_result(kernel_id, result)
        return result

    def _run_fallback_step(self, step: int, config: dict) -> dict:
        """
        Run a single training step with fallback config.

        In real implementation, initializes model with config and runs
        one forward+backward pass.
        """
        return {"loss": 0.0}

    def _load_baseline_losses(self, kernel_id: str) -> list:
        """Load baseline loss trajectory for comparison."""
        path = os.path.join(self.data_dir, "baselines", f"{kernel_id}_losses.json")
        if os.path.exists(path):
            try:
                with open(path) as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                pass
        return []

    def _compare_loss_trajectory(
        self, actual: list, baseline: list, tolerance: float
    ) -> bool:
        """Check if actual losses are within tolerance of baseline."""
        n = min(len(actual), len(baseline))
        for i in range(n):
            if baseline[i] != 0:
                rel_diff = abs(actual[i] - baseline[i]) / abs(baseline[i])
                if rel_diff > tolerance:
                    return False
        return True

    def _save_result(self, kernel_id: str, result: dict):
        """Persist verification result."""
        out_dir = os.path.join(self.data_dir, "fallback_verification")
        try:
            os.makedirs(out_dir, exist_ok=True)
            path = os.path.join(
                out_dir, f"{kernel_id}_{int(time.time())}.json"
            )
            with open(path, "w") as f:
                json.dump(result, f, indent=2)
        except OSError:
            pass
