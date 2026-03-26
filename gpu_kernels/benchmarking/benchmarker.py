"""
Kernel benchmarking with CUDA event timing.
"""

import math
import statistics

import torch

from gpu_kernels.schemas import BenchmarkResult, BenchmarkConfigResult


class KernelBenchmarker:
    """Benchmark a Triton kernel against a reference implementation."""

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def benchmark(
        self,
        kernel_callable,
        reference_callable,
        input_configs: list,
        n_warmup: int = 50,
        n_timed: int = 200,
        n_runs: int = 5,
        kernel_id: str = "",
        group_id: str = "",
    ) -> BenchmarkResult:
        """Benchmark kernel vs reference across input configurations.

        Args:
            kernel_callable: The kernel function to benchmark.
            reference_callable: The reference PyTorch function.
            input_configs: List of dicts, each with "name" and "inputs" (list of tensors).
            n_warmup: Warmup iterations before timing.
            n_timed: Timed iterations per run.
            n_runs: Number of independent runs to collect.
            kernel_id: Kernel identifier for the result.
            group_id: Associated group identifier.

        Returns:
            BenchmarkResult with per-config and overall speedup.
        """
        if not torch.cuda.is_available():
            return BenchmarkResult(kernel_id=kernel_id, reference_group_id=group_id)

        config_results = []
        speedup_ratios = []

        peak_mem_kernel = 0
        peak_mem_ref = 0

        for cfg in input_configs:
            config_name = cfg.get("name", "unnamed")
            inputs = cfg["inputs"]

            ref_times = self._time_callable(reference_callable, inputs, n_warmup, n_timed, n_runs)
            kern_times = self._time_callable(kernel_callable, inputs, n_warmup, n_timed, n_runs)

            ref_median = statistics.median(ref_times)
            kern_median = statistics.median(kern_times)

            ref_cv = statistics.stdev(ref_times) / statistics.mean(ref_times) if len(ref_times) > 1 and statistics.mean(ref_times) > 0 else 0.0
            kern_cv = statistics.stdev(kern_times) / statistics.mean(kern_times) if len(kern_times) > 1 and statistics.mean(kern_times) > 0 else 0.0

            speedup = ref_median / kern_median if kern_median > 0 else 0.0

            # Reject noisy results
            if kern_cv > 0.05 or ref_cv > 0.05:
                speedup = 0.0  # mark as unreliable

            config_results.append(BenchmarkConfigResult(
                config_name=config_name,
                reference_median_us=ref_median,
                kernel_median_us=kern_median,
                speedup_ratio=speedup,
                cv_reference=ref_cv,
                cv_kernel=kern_cv,
            ))

            if speedup > 0:
                speedup_ratios.append(speedup)

            # Track peak memory
            torch.cuda.reset_peak_memory_stats()
            kernel_callable(*inputs)
            torch.cuda.synchronize()
            peak_mem_kernel = max(peak_mem_kernel, torch.cuda.max_memory_allocated())

            torch.cuda.reset_peak_memory_stats()
            reference_callable(*inputs)
            torch.cuda.synchronize()
            peak_mem_ref = max(peak_mem_ref, torch.cuda.max_memory_allocated())

        # Geometric mean of valid speedups
        overall_speedup = 0.0
        if speedup_ratios:
            log_sum = sum(math.log(s) for s in speedup_ratios)
            overall_speedup = math.exp(log_sum / len(speedup_ratios))

        mem_savings = 1.0 - (peak_mem_kernel / peak_mem_ref) if peak_mem_ref > 0 else 0.0

        return BenchmarkResult(
            kernel_id=kernel_id,
            reference_group_id=group_id,
            input_configs_tested=len(input_configs),
            results=[r.to_dict() for r in config_results],
            overall_speedup=overall_speedup,
            peak_memory_kernel_bytes=peak_mem_kernel,
            peak_memory_reference_bytes=peak_mem_ref,
            memory_savings_ratio=mem_savings,
        )

    def _time_callable(self, fn, inputs, n_warmup, n_timed, n_runs):
        """Time a callable using CUDA events. Returns list of median times (us) per run."""
        run_medians = []
        for _ in range(n_runs):
            # Warmup
            for _ in range(n_warmup):
                fn(*inputs)
            torch.cuda.synchronize()

            # Timed
            times = []
            for _ in range(n_timed):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                fn(*inputs)
                end.record()
                torch.cuda.synchronize()
                elapsed_ms = start.elapsed_time(end)
                times.append(elapsed_ms * 1000.0)  # convert ms -> us

            run_medians.append(statistics.median(times))
        return run_medians
