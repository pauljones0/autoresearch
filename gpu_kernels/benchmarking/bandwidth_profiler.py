"""
Memory bandwidth profiling for generated kernels.
"""

import torch

from gpu_kernels.schemas import BandwidthProfile


class MemoryBandwidthProfiler:
    """Profile memory bandwidth utilization of a kernel."""

    def __init__(self, peak_bandwidth_gbps: float = 0.0):
        """Initialize profiler.

        Args:
            peak_bandwidth_gbps: Theoretical peak memory bandwidth in GB/s.
                If 0, will attempt to auto-detect from device properties.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.peak_bandwidth_gbps = peak_bandwidth_gbps
        if self.peak_bandwidth_gbps == 0.0 and torch.cuda.is_available():
            self.peak_bandwidth_gbps = self._detect_peak_bandwidth()

    def profile(
        self,
        kernel_callable,
        input_config: dict,
        n_warmup: int = 20,
        n_timed: int = 100,
    ) -> BandwidthProfile:
        """Profile memory bandwidth for a kernel invocation.

        Args:
            kernel_callable: The kernel function.
            input_config: Dict with "inputs" (list of tensors) and optionally
                "bytes_read" and "bytes_written" overrides. If not provided,
                bytes are estimated from input/output tensor sizes.
            n_warmup: Warmup iterations.
            n_timed: Timed iterations for median time.

        Returns:
            BandwidthProfile with bandwidth measurements and classification.
        """
        if not torch.cuda.is_available():
            return BandwidthProfile()

        inputs = input_config["inputs"]

        # Estimate bytes read from input tensors
        bytes_read = input_config.get("bytes_read", 0)
        if bytes_read == 0:
            for t in inputs:
                if isinstance(t, torch.Tensor):
                    bytes_read += t.nelement() * t.element_size()

        # Run once to get output shape for bytes_written estimate
        out = kernel_callable(*inputs)
        bytes_written = input_config.get("bytes_written", 0)
        if bytes_written == 0:
            if isinstance(out, torch.Tensor):
                bytes_written = out.nelement() * out.element_size()
            elif isinstance(out, (list, tuple)):
                for o in out:
                    if isinstance(o, torch.Tensor):
                        bytes_written += o.nelement() * o.element_size()

        # Warmup
        for _ in range(n_warmup):
            kernel_callable(*inputs)
        torch.cuda.synchronize()

        # Time the kernel
        times_us = []
        for _ in range(n_timed):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            kernel_callable(*inputs)
            end.record()
            torch.cuda.synchronize()
            times_us.append(start.elapsed_time(end) * 1000.0)

        times_us.sort()
        median_time_us = times_us[len(times_us) // 2]

        # Compute bandwidth
        total_bytes = bytes_read + bytes_written
        execution_time_s = median_time_us / 1e6
        achieved_gbps = (total_bytes / 1e9) / execution_time_s if execution_time_s > 0 else 0.0

        efficiency = 0.0
        if self.peak_bandwidth_gbps > 0:
            efficiency = achieved_gbps / self.peak_bandwidth_gbps

        return BandwidthProfile(
            bytes_read=bytes_read,
            bytes_written=bytes_written,
            execution_time_us=median_time_us,
            achieved_bandwidth_gbps=achieved_gbps,
            bandwidth_efficiency=efficiency,
            is_memory_bound=efficiency > 0.6,
            is_compute_bound=efficiency < 0.3,
        )

    def _detect_peak_bandwidth(self) -> float:
        """Estimate peak memory bandwidth from CUDA device properties."""
        props = torch.cuda.get_device_properties(0)
        # memory_clock_rate is in kHz, memory_bus_width in bits
        clock_khz = props.memory_clock_rate  # kHz
        bus_width_bits = props.memory_bus_width  # bits
        # Bandwidth = clock * bus_width * 2 (DDR) / 8 (bits->bytes) / 1e6 (kHz*bits -> GB/s)
        peak_gbps = (clock_khz * 1e3) * (bus_width_bits / 8) * 2 / 1e9
        return peak_gbps
