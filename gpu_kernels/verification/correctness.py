"""
Kernel correctness verification against reference implementations.
"""

import os
import time
import glob
import traceback

import torch

from gpu_kernels.schemas import CorrectnessResult, ToleranceBounds


class KernelCorrectnessVerifier:
    """Verify that a Triton kernel produces outputs matching a reference implementation."""

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def verify(
        self,
        kernel_callable,
        reference_callable,
        test_inputs_dir: str,
        tolerances: ToleranceBounds = None,
        dtypes: list = None,
    ) -> CorrectnessResult:
        """Run correctness verification across dtypes and test configurations.

        Args:
            kernel_callable: The kernel function to test.
            reference_callable: The reference PyTorch function.
            test_inputs_dir: Directory containing .pt test input files.
            tolerances: Per-dtype tolerance bounds.
            dtypes: List of dtype names to test (e.g. ["fp32", "bf16"]).

        Returns:
            CorrectnessResult with pass/fail details.
        """
        if not torch.cuda.is_available():
            return CorrectnessResult(passed=False, failed_configs=[{"reason": "CUDA not available"}])

        if tolerances is None:
            tolerances = ToleranceBounds()
        if dtypes is None:
            dtypes = ["fp32", "bf16"]

        dtype_map = {
            "fp32": torch.float32,
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
        }
        tolerance_map = {
            "fp32": tolerances.fp32,
            "fp16": tolerances.fp16,
            "bf16": tolerances.bf16,
        }

        # Load test input files
        input_files = sorted(glob.glob(os.path.join(test_inputs_dir, "*.pt")))
        if not input_files:
            return CorrectnessResult(
                passed=False,
                failed_configs=[{"reason": f"No .pt files found in {test_inputs_dir}"}],
            )

        t_start = time.time()
        tested = 0
        passed = 0
        failed = []

        for input_file in input_files:
            test_inputs = torch.load(input_file, map_location=self.device, weights_only=True)
            config_name = os.path.basename(input_file)

            for dtype_name in dtypes:
                if dtype_name not in dtype_map:
                    continue
                torch_dtype = dtype_map[dtype_name]
                tols = tolerance_map.get(dtype_name, {"atol": 1e-5, "rtol": 1e-5})
                tested += 1

                try:
                    casted = self._cast_inputs(test_inputs, torch_dtype)
                    ref_out = reference_callable(*casted)
                    kern_out = kernel_callable(*casted)

                    if not self._compare(ref_out, kern_out, tols):
                        failure = self._build_failure(
                            config_name, dtype_name, ref_out, kern_out, tols
                        )
                        failed.append(failure)
                    else:
                        passed += 1
                except Exception:
                    failed.append({
                        "config": config_name,
                        "dtype": dtype_name,
                        "error": traceback.format_exc(),
                    })

        total_time = time.time() - t_start
        return CorrectnessResult(
            passed=(len(failed) == 0),
            tested_configs=tested,
            passed_configs=passed,
            failed_configs=failed,
            total_time_seconds=total_time,
        )

    def _cast_inputs(self, inputs, dtype):
        """Cast input tensors to the target dtype."""
        if isinstance(inputs, torch.Tensor):
            return [inputs.to(dtype=dtype)]
        if isinstance(inputs, (list, tuple)):
            return [t.to(dtype=dtype) if isinstance(t, torch.Tensor) and t.is_floating_point() else t for t in inputs]
        if isinstance(inputs, dict):
            return [v.to(dtype=dtype) if isinstance(v, torch.Tensor) and v.is_floating_point() else v for v in inputs.values()]
        return [inputs]

    def _compare(self, ref, kern, tols):
        """Compare reference and kernel outputs."""
        if isinstance(ref, torch.Tensor) and isinstance(kern, torch.Tensor):
            return torch.allclose(
                ref.float(), kern.float(),
                atol=tols["atol"], rtol=tols["rtol"],
            )
        if isinstance(ref, (list, tuple)) and isinstance(kern, (list, tuple)):
            return all(self._compare(r, k, tols) for r, k in zip(ref, kern))
        return False

    def _build_failure(self, config_name, dtype_name, ref_out, kern_out, tols):
        """Build a failure detail dict with max error info."""
        failure = {"config": config_name, "dtype": dtype_name}
        if isinstance(ref_out, torch.Tensor) and isinstance(kern_out, torch.Tensor):
            diff = (ref_out.float() - kern_out.float()).abs()
            max_err = diff.max().item()
            max_pos = torch.argmax(diff).item()
            failure["max_abs_error"] = max_err
            failure["max_error_position"] = max_pos
            failure["atol"] = tols["atol"]
            failure["rtol"] = tols["rtol"]
        return failure
