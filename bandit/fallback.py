"""
Graceful degradation handler for bandit pipeline failures.
"""

import copy
import json
import os

from bandit.schemas import BanditState, ArmState, FallbackDecision


class GracefulDegradationHandler:
    """Recovers from various failure modes in the bandit pipeline."""

    def handle_failure(
        self,
        error: Exception,
        state: BanditState,
        iteration: int,
    ) -> FallbackDecision:
        """Determine recovery action based on failure type.

        Handles:
        - State corruption: reset to safe defaults
        - Log corruption: truncate to last valid entry
        - Dispatch target unavailable: disable affected arms
        - Nonsensical selections: fallback to scheduler
        """
        error_str = str(error).lower()
        error_type = type(error).__name__

        # State corruption
        if self._is_state_corruption(error_str, error_type, state):
            return self._recover_state(state, iteration)

        # Log corruption
        if self._is_log_corruption(error_str, error_type):
            return FallbackDecision(
                action="recover_log",
                detail=f"Log corruption at iteration {iteration}: {error_str[:200]}. "
                       f"Truncate log to last valid entry and continue.",
            )

        # Dispatch target unavailable
        if self._is_dispatch_unavailable(error_str, error_type):
            arm_id = self._extract_arm_from_error(error_str)
            return FallbackDecision(
                action="disable_arms",
                detail=f"Dispatch target unavailable for arm {arm_id} "
                       f"at iteration {iteration}. "
                       f"Disabling arm until next health check.",
            )

        # Nonsensical selection (empty arm, unknown arm, etc.)
        if self._is_nonsensical_selection(error_str, error_type):
            return FallbackDecision(
                action="fallback_to_scheduler",
                detail=f"Nonsensical selection at iteration {iteration}: "
                       f"{error_str[:200]}. Falling back to EvaluationScheduler.",
            )

        # Default: fallback to scheduler
        return FallbackDecision(
            action="fallback_to_scheduler",
            detail=f"Unhandled error at iteration {iteration}: "
                   f"{error_type}: {error_str[:200]}. "
                   f"Falling back to EvaluationScheduler.",
        )

    def _is_state_corruption(
        self, error_str: str, error_type: str, state: BanditState
    ) -> bool:
        """Detect state corruption from error context."""
        corruption_signals = [
            "alpha", "beta", "negative", "nan", "inf",
            "keyerror", "attributeerror", "typeerror",
        ]
        if any(s in error_str for s in corruption_signals):
            return True
        # Also check if state itself is invalid
        if state is not None:
            issues = state.validate()
            if issues:
                return True
        return False

    def _is_log_corruption(self, error_str: str, error_type: str) -> bool:
        """Detect log file corruption."""
        return any(s in error_str for s in [
            "jsondecodeerror", "json", "log", "truncat",
            "expecting value", "unterminated",
        ])

    def _is_dispatch_unavailable(self, error_str: str, error_type: str) -> bool:
        """Detect dispatch target unavailability."""
        return any(s in error_str for s in [
            "dispatch", "target", "unavailable", "connection",
            "timeout", "unreachable", "importerror", "modulenotfounderror",
        ])

    def _is_nonsensical_selection(self, error_str: str, error_type: str) -> bool:
        """Detect nonsensical arm selections."""
        return any(s in error_str for s in [
            "selection", "empty", "unknown arm", "no arms",
            "invalid arm", "sample",
        ])

    def _extract_arm_from_error(self, error_str: str) -> str:
        """Try to extract arm ID from error message."""
        # Look for common arm IDs in the error string
        known = [
            "architecture", "optimizer", "hyperparameter", "activation",
            "initialization", "regularization", "scheduling",
            "kernel_discovery", "kernel_evolution",
        ]
        for arm_id in known:
            if arm_id in error_str:
                return arm_id
        return "unknown"

    def _recover_state(
        self, state: BanditState, iteration: int
    ) -> FallbackDecision:
        """Generate recovery decision for corrupted state."""
        issues = state.validate() if state is not None else ["state is None"]
        return FallbackDecision(
            action="recover_state",
            detail=f"State corruption at iteration {iteration}. "
                   f"Issues: {'; '.join(issues[:5])}. "
                   f"Reset affected arms to Beta(1,1) priors.",
        )
