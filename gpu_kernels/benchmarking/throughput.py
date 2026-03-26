"""
Training throughput impact estimation from kernel benchmark results.
"""

from gpu_kernels.schemas import BenchmarkResult, ThroughputEstimate


class TrainingThroughputEstimator:
    """Estimate end-to-end training throughput improvement from a kernel optimization."""

    def estimate_impact(
        self,
        benchmark_result: BenchmarkResult,
        operation_gpu_time_us: float,
        total_step_time_us: float,
        baseline_tok_per_sec: float,
        kernel_id: str = "",
        target_operation: str = "",
    ) -> ThroughputEstimate:
        """Estimate training throughput impact of a kernel optimization.

        Args:
            benchmark_result: BenchmarkResult from KernelBenchmarker.
            operation_gpu_time_us: GPU time of the target operation in the baseline (us).
            total_step_time_us: Total training step time in the baseline (us).
            baseline_tok_per_sec: Baseline training throughput in tokens/second.
            kernel_id: Kernel identifier.
            target_operation: Name of the target operation.

        Returns:
            ThroughputEstimate with projected improvements.
        """
        if total_step_time_us <= 0 or baseline_tok_per_sec <= 0:
            return ThroughputEstimate(kernel_id=kernel_id, target_operation=target_operation)

        speedup = benchmark_result.overall_speedup
        if speedup <= 0:
            return ThroughputEstimate(kernel_id=kernel_id, target_operation=target_operation)

        # Fraction of step time this operation occupies
        op_fraction = operation_gpu_time_us / total_step_time_us

        # New operation time after speedup
        new_op_time_us = operation_gpu_time_us / speedup
        time_saved_us = operation_gpu_time_us - new_op_time_us

        # New total step time (Amdahl's law)
        new_step_time_us = total_step_time_us - time_saved_us

        # Throughput improvement
        step_speedup = total_step_time_us / new_step_time_us if new_step_time_us > 0 else 1.0
        new_tok_per_sec = baseline_tok_per_sec * step_speedup
        tok_delta = new_tok_per_sec - baseline_tok_per_sec

        # Training time savings as percentage
        savings_pct = (1.0 - new_step_time_us / total_step_time_us) * 100.0

        # Confidence based on speedup magnitude and op fraction
        if op_fraction < 0.01:
            confidence = "low"
        elif speedup > 2.0 or op_fraction > 0.2:
            confidence = "high"
        else:
            confidence = "medium"

        return ThroughputEstimate(
            kernel_id=kernel_id,
            target_operation=target_operation,
            operation_fraction_of_step=op_fraction,
            estimated_step_time_reduction_us=time_saved_us,
            estimated_tok_per_sec_delta=tok_delta,
            estimated_training_time_savings_percent=savings_pct,
            confidence=confidence,
        )
