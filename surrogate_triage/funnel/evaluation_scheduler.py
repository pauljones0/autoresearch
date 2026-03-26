"""
Phase 3.3 — EvaluationScheduler: decide whether the next iteration should
evaluate a paper candidate or an internal modification.
"""

import logging

logger = logging.getLogger(__name__)

_DEFAULT_PAPER_FRACTION = 0.30


class EvaluationScheduler:
    """Schedule paper vs internal evaluations with configurable split."""

    def __init__(self, paper_fraction: float = _DEFAULT_PAPER_FRACTION):
        """
        Args:
            paper_fraction: Target fraction of iterations allocated to paper
                candidates (default 0.30 = 30%).
        """
        self.paper_fraction = paper_fraction
        self._paper_count = 0
        self._internal_count = 0

    @property
    def total_count(self) -> int:
        return self._paper_count + self._internal_count

    @property
    def current_paper_fraction(self) -> float:
        if self.total_count == 0:
            return 0.0
        return self._paper_count / self.total_count

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def should_evaluate_paper(
        self,
        queue_size: int,
        iteration: int,
        safety_guard=None,
    ) -> bool:
        """Decide whether the next evaluation should be a paper candidate.

        Args:
            queue_size: Number of pending paper candidates in the queue.
            iteration: Current iteration number (1-based).
            safety_guard: Optional SafetyGuard instance for compute budget check.

        Returns:
            True if a paper candidate should be evaluated next.
        """
        # No paper candidates available — always internal
        if queue_size <= 0:
            return False

        # Check compute budget if safety_guard available
        if safety_guard is not None:
            try:
                allowed, reason = safety_guard.can_run_scale_test(0.0)
                if not allowed:
                    logger.info(
                        "Skipping paper evaluation due to compute budget: %s", reason
                    )
                    return False
            except Exception as exc:
                logger.warning("Safety guard check failed: %s", exc)

        # Maintain target paper fraction
        if self.total_count == 0:
            # First iteration: use paper if available and fraction > 0
            return self.paper_fraction > 0.0

        # If paper fraction is below target, prefer paper
        return self.current_paper_fraction < self.paper_fraction

    def get_next_candidate(self, queue_manager):
        """Pop the next candidate from the queue.

        Args:
            queue_manager: QueueManager instance.

        Returns:
            QueueEntry or None if no candidates available.
        """
        entry = queue_manager.pop_next()
        return entry

    def record_paper_evaluation(self):
        """Record that a paper evaluation occurred."""
        self._paper_count += 1

    def record_internal_evaluation(self):
        """Record that an internal evaluation occurred."""
        self._internal_count += 1

    def stats(self) -> dict:
        """Return scheduler statistics."""
        return {
            "paper_count": self._paper_count,
            "internal_count": self._internal_count,
            "total_count": self.total_count,
            "current_paper_fraction": round(self.current_paper_fraction, 3),
            "target_paper_fraction": self.paper_fraction,
        }
