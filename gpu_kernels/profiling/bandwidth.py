"""
BandwidthUtilizationCalculator: computes memory bandwidth utilization
for each profiled operation and flags memory-bound bottlenecks.
"""

from ..schemas import OperationProfile, HardwareProfile


class BandwidthUtilizationCalculator:
    """Computes bandwidth utilization for profiled ops and flags bottlenecks.

    An op is flagged when:
      - bandwidth_utilization < 0.3 (less than 30% of peak), AND
      - gpu_time_us > 100 (non-trivial runtime)
    """

    LOW_UTILIZATION_THRESHOLD = 0.3
    MIN_GPU_TIME_US = 100.0

    def calculate(
        self,
        profiles: list[OperationProfile],
        hardware: HardwareProfile,
    ) -> list[OperationProfile]:
        """Fill in bandwidth_utilization for each profile.

        Args:
            profiles: List of OperationProfile from the profiler.
            hardware: HardwareProfile with peak_bandwidth_gbps.

        Returns:
            The same list with bandwidth_utilization and is_fuseable updated.
        """
        peak_bw_bytes_per_sec = hardware.peak_bandwidth_gbps * 1e9  # GB/s -> B/s

        if peak_bw_bytes_per_sec <= 0:
            return profiles

        for op in profiles:
            total_bytes = op.memory_read_bytes + op.memory_write_bytes
            gpu_time_s = op.gpu_time_us * 1e-6

            if gpu_time_s > 0 and total_bytes > 0:
                achieved_bw = total_bytes / gpu_time_s
                op.bandwidth_utilization = achieved_bw / peak_bw_bytes_per_sec
            else:
                op.bandwidth_utilization = 0.0

            # Flag as fuseable candidate if low utilization AND significant time
            if (
                op.bandwidth_utilization < self.LOW_UTILIZATION_THRESHOLD
                and op.gpu_time_us > self.MIN_GPU_TIME_US
            ):
                op.is_fuseable = True

        return profiles
