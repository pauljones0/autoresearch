"""
Elementwise verification runner: runs full verify+benchmark pipeline on kernel variants.
"""

import time
import importlib.util
from typing import List, Tuple, Optional, Callable, Any

import torch

import sys, os
from ..schemas import GeneratedKernel


class ElementwiseVerificationRunner:
    """Run correctness verification and benchmarking on elementwise kernel variants.

    Verifies each variant against a reference callable, then benchmarks to find
    the fastest correct variant.
    """

    def __init__(
        self,
        atol: float = 1e-2,
        rtol: float = 1e-2,
        n_warmup: int = 10,
        n_benchmark: int = 100,
        n_correctness_trials: int = 5,
    ):
        self.atol = atol
        self.rtol = rtol
        self.n_warmup = n_warmup
        self.n_benchmark = n_benchmark
        self.n_correctness_trials = n_correctness_trials

    def _load_kernel_callable(self, kernel: GeneratedKernel) -> Optional[Callable]:
        """Dynamically load the fused_op callable from a kernel file."""
        try:
            spec = importlib.util.spec_from_file_location(
                kernel.kernel_id, kernel.kernel_path
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return getattr(module, 'fused_op', None)
        except Exception:
            return None

    def _verify_correctness(
        self,
        kernel_callable: Callable,
        reference_callable: Callable,
        test_inputs: List[torch.Tensor],
    ) -> dict:
        """Verify kernel correctness against reference for all test inputs.

        Returns:
            Dict with keys: passed, max_abs_error, max_rel_error, failures
        """
        max_abs_error = 0.0
        max_rel_error = 0.0
        failures = []

        for i, inp in enumerate(test_inputs):
            for trial in range(self.n_correctness_trials):
                try:
                    ref_out = reference_callable(inp)
                    kernel_out = kernel_callable(inp)

                    if ref_out.shape != kernel_out.shape:
                        failures.append({
                            "input_idx": i,
                            "trial": trial,
                            "error": f"Shape mismatch: ref={ref_out.shape}, kernel={kernel_out.shape}",
                        })
                        continue

                    abs_err = (ref_out.float() - kernel_out.float()).abs()
                    cur_max_abs = abs_err.max().item()
                    max_abs_error = max(max_abs_error, cur_max_abs)

                    ref_abs = ref_out.float().abs().clamp(min=1e-8)
                    rel_err = (abs_err / ref_abs).max().item()
                    max_rel_error = max(max_rel_error, rel_err)

                    if not torch.allclose(
                        ref_out.float(), kernel_out.float(),
                        atol=self.atol, rtol=self.rtol
                    ):
                        failures.append({
                            "input_idx": i,
                            "trial": trial,
                            "max_abs_error": cur_max_abs,
                            "max_rel_error": rel_err,
                        })

                except Exception as e:
                    failures.append({
                        "input_idx": i,
                        "trial": trial,
                        "error": str(e),
                    })

        return {
            "passed": len(failures) == 0,
            "max_abs_error": max_abs_error,
            "max_rel_error": max_rel_error,
            "failures": failures,
            "n_tests": len(test_inputs) * self.n_correctness_trials,
        }

    def _benchmark_kernel(
        self,
        kernel_callable: Callable,
        test_inputs: List[torch.Tensor],
    ) -> dict:
        """Benchmark kernel latency using CUDA events.

        Returns:
            Dict with keys: median_us, mean_us, min_us, max_us, cv
        """
        # Use the largest test input for benchmarking
        inp = max(test_inputs, key=lambda t: t.numel())

        # Warmup
        for _ in range(self.n_warmup):
            kernel_callable(inp)
        torch.cuda.synchronize()

        # Benchmark with CUDA events
        timings_ms = []
        for _ in range(self.n_benchmark):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            kernel_callable(inp)
            end.record()
            torch.cuda.synchronize()
            timings_ms.append(start.elapsed_time(end))

        timings_us = [t * 1000 for t in timings_ms]
        timings_us.sort()
        n = len(timings_us)
        median = timings_us[n // 2]
        mean = sum(timings_us) / n
        variance = sum((t - mean) ** 2 for t in timings_us) / n
        std = variance ** 0.5
        cv = std / mean if mean > 0 else 0.0

        return {
            "median_us": median,
            "mean_us": mean,
            "min_us": timings_us[0],
            "max_us": timings_us[-1],
            "cv": cv,
        }

    def run(
        self,
        variants: List[GeneratedKernel],
        reference_callable: Callable,
        test_inputs: List[torch.Tensor],
    ) -> Tuple[Optional[GeneratedKernel], List[dict]]:
        """Run full verification + benchmark pipeline on all variants.

        Args:
            variants: List of GeneratedKernel variants to test.
            reference_callable: PyTorch reference implementation.
            test_inputs: List of test input tensors.

        Returns:
            Tuple of (winner, results):
                - winner: The fastest correct GeneratedKernel, or None if all fail.
                - results: List of per-variant result dicts.
        """
        results = []
        best_kernel = None
        best_median_us = float('inf')

        # Benchmark reference
        ref_benchmark = self._benchmark_kernel(reference_callable, test_inputs)

        for variant in variants:
            result = {
                "kernel_id": variant.kernel_id,
                "variant_index": variant.variant_index,
                "block_size": variant.block_size,
                "num_warps": variant.num_warps,
                "correctness": None,
                "benchmark": None,
                "speedup": 0.0,
                "status": "pending",
            }

            # Load kernel
            kernel_fn = self._load_kernel_callable(variant)
            if kernel_fn is None:
                result["status"] = "load_failed"
                results.append(result)
                continue

            # Verify correctness
            correctness = self._verify_correctness(
                kernel_fn, reference_callable, test_inputs
            )
            result["correctness"] = correctness

            if not correctness["passed"]:
                result["status"] = "correctness_failed"
                results.append(result)
                continue

            # Benchmark
            benchmark = self._benchmark_kernel(kernel_fn, test_inputs)
            result["benchmark"] = benchmark
            result["speedup"] = ref_benchmark["median_us"] / benchmark["median_us"] if benchmark["median_us"] > 0 else 0.0
            result["status"] = "passed"

            results.append(result)

            # Track best
            if benchmark["median_us"] < best_median_us:
                best_median_us = benchmark["median_us"]
                best_kernel = variant

        return best_kernel, results
