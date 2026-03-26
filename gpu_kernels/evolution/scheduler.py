"""
EvolutionaryRefinementScheduler: selects which active kernel to refine next
based on bandwidth utilization, step fraction, and recency of last refinement.
"""

import time

from ..schemas import KernelConfigEntry


class EvolutionaryRefinementScheduler:
    """Select which active kernel should be refined next.

    Criteria (higher priority first):
      - bandwidth_utilization < 0.6 (room for improvement)
      - High fraction of step time (bigger impact)
      - Not recently refined (avoid thrashing)
    """

    def __init__(
        self,
        min_bandwidth_threshold: float = 0.6,
        recency_cooldown_seconds: float = 3600.0,
    ):
        """Initialize the scheduler.

        Args:
            min_bandwidth_threshold: Only consider kernels below this utilization.
            recency_cooldown_seconds: Minimum seconds since last refinement.
        """
        self._min_bw = min_bandwidth_threshold
        self._cooldown = recency_cooldown_seconds

    def select(
        self,
        kernel_config: dict,
        diagnostics_report: dict,
    ) -> str | None:
        """Select the best kernel group to refine.

        Args:
            kernel_config: Current kernel_config.json as dict (group_id -> entry).
            diagnostics_report: DiagnosticsReport as dict, used for step timing.

        Returns:
            group_id to refine, or None if no suitable candidate.
        """
        now = time.time()
        total_step_time = diagnostics_report.get("training_seconds", 0.0)
        # Treat step timing as the total measured time for ranking

        candidates: list[tuple[str, float]] = []

        for group_id, entry in kernel_config.items():
            if isinstance(entry, dict):
                backend = entry.get("backend", "pytorch")
                enabled = entry.get("enabled", True)
                bw_util = entry.get("bandwidth_utilization", 1.0)
                step_fraction = entry.get("step_fraction", 0.0)
                last_refined = entry.get("last_refined_at", 0.0)
                speedup = entry.get("speedup", 1.0)
            elif isinstance(entry, KernelConfigEntry):
                backend = entry.backend
                enabled = entry.enabled
                bw_util = getattr(entry, "bandwidth_utilization", 1.0)
                step_fraction = getattr(entry, "step_fraction", 0.0)
                last_refined = getattr(entry, "last_refined_at", 0.0)
                speedup = entry.speedup
            else:
                continue

            # Only consider active Triton kernels
            if backend != "triton" or not enabled:
                continue

            # Skip if bandwidth utilization is already good
            if bw_util >= self._min_bw:
                continue

            # Skip if recently refined
            if now - last_refined < self._cooldown:
                continue

            # Score: higher is more urgent to refine
            # Prioritize: low bandwidth utilization + high step fraction
            improvement_room = self._min_bw - bw_util
            score = improvement_room * (1.0 + step_fraction * 10.0)

            candidates.append((group_id, score))

        if not candidates:
            return None

        # Sort by score descending, return the best candidate
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[0][0]
