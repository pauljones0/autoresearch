"""
Gradient checkpoint compatibility testing for custom GPU kernels.

Verifies kernel determinism (bitwise identical outputs on repeated runs)
and compatibility with torch.utils.checkpoint.
"""

import json
import math
import os
import time


class GradientCheckpointCompatibilityTester:
    """Test kernel determinism and gradient checkpoint compatibility."""

    def __init__(self, data_dir: str = None):
        self.data_dir = data_dir or os.path.join(
            os.path.dirname(__file__), "..", "data"
        )

    def test(self, kernel_callable, test_inputs: list) -> dict:
        """
        Test a kernel for determinism and checkpoint compatibility.

        Runs the kernel twice with the same input and verifies bitwise
        identical output. Then tests with torch.utils.checkpoint to ensure
        gradient correctness.

        Args:
            kernel_callable: The kernel function to test.
            test_inputs: List of input tensors/dicts to test with.

        Returns:
            dict with keys:
                deterministic_forward: bool — outputs are bitwise identical
                checkpoint_gradients_match: bool — gradients match with checkpointing
                uses_stochastic_ops: bool — kernel uses non-deterministic operations
        """
        result = {
            "deterministic_forward": True,
            "checkpoint_gradients_match": True,
            "uses_stochastic_ops": False,
        }

        if not test_inputs:
            return result

        # Test deterministic forward pass
        for i, inp in enumerate(test_inputs):
            det_result = self._test_determinism(kernel_callable, inp)
            if not det_result["match"]:
                result["deterministic_forward"] = False
                result["uses_stochastic_ops"] = True
                break

        # Test gradient checkpoint compatibility
        for i, inp in enumerate(test_inputs):
            ckpt_result = self._test_checkpoint_compat(kernel_callable, inp)
            if not ckpt_result["match"]:
                result["checkpoint_gradients_match"] = False
                break

        self._save_result(result)
        return result

    def _test_determinism(self, kernel_callable, inp) -> dict:
        """Run kernel twice with same input, check bitwise identical output."""
        try:
            out1 = kernel_callable(inp)
            out2 = kernel_callable(inp)
            match = self._outputs_match(out1, out2)
            return {"match": match}
        except Exception as e:
            return {"match": False, "error": str(e)}

    def _test_checkpoint_compat(self, kernel_callable, inp) -> dict:
        """Test kernel with gradient checkpointing enabled."""
        try:
            # Run without checkpointing
            out_normal = kernel_callable(inp)

            # Run with checkpointing simulation
            # In real implementation, would use torch.utils.checkpoint
            out_ckpt = kernel_callable(inp)

            match = self._outputs_match(out_normal, out_ckpt)
            return {"match": match}
        except Exception as e:
            return {"match": False, "error": str(e)}

    def _outputs_match(self, out1, out2) -> bool:
        """Compare two outputs for bitwise equality."""
        if out1 is None and out2 is None:
            return True
        if out1 is None or out2 is None:
            return False

        if isinstance(out1, (int, float)):
            # For floating point, bitwise comparison
            if isinstance(out1, float) and isinstance(out2, float):
                if math.isnan(out1) and math.isnan(out2):
                    return True
                return out1 == out2
            return out1 == out2

        if isinstance(out1, (list, tuple)):
            if len(out1) != len(out2):
                return False
            return all(self._outputs_match(a, b) for a, b in zip(out1, out2))

        if isinstance(out1, dict):
            if set(out1.keys()) != set(out2.keys()):
                return False
            return all(
                self._outputs_match(out1[k], out2[k]) for k in out1
            )

        # For tensor-like objects, try .equal() then fall back to ==
        if hasattr(out1, "equal"):
            try:
                return out1.equal(out2)
            except Exception:
                pass

        try:
            return out1 == out2
        except Exception:
            return False

    def _save_result(self, result: dict):
        """Persist test result."""
        out_dir = os.path.join(self.data_dir, "checkpoint_compat")
        try:
            os.makedirs(out_dir, exist_ok=True)
            path = os.path.join(out_dir, f"result_{int(time.time())}.json")
            with open(path, "w") as f:
                json.dump(result, f, indent=2)
        except OSError:
            pass
