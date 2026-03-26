"""
Posterior update engine — updates arm state after dispatch results.
"""

import copy
import time

from bandit.schemas import BanditState, ArmState, DispatchResult, save_json
from bandit.boosting import DiagnosticsArmBooster
from bandit.delayed_corrections import DelayedCorrectionReceiver


class PosteriorUpdateEngine:
    """Updates bandit posterior (alpha/beta) based on dispatch results."""

    def __init__(self, state_path: str = "strategy_state.json"):
        self.state_path = state_path
        self.booster = DiagnosticsArmBooster()
        self.correction_receiver = DelayedCorrectionReceiver()

    def update(self, state: BanditState, dispatch_result: DispatchResult,
               log_writer=None) -> BanditState:
        """Update alpha/beta based on verdict, recompute temperature, decay boosts.

        Transactional: if save fails, don't update in-memory state.

        Args:
            state: Current bandit state.
            dispatch_result: Result from dispatching the arm.
            log_writer: Optional log writer for recording updates.

        Returns:
            Updated BanditState.
        """
        # Work on a copy for transactional safety
        new_state = copy.deepcopy(state)

        arm_id = dispatch_result.arm_id
        if arm_id not in new_state.arms:
            return state

        arm = new_state.arms[arm_id]
        if not isinstance(arm, ArmState):
            return state

        # Update alpha/beta based on verdict
        verdict = dispatch_result.verdict.lower() if dispatch_result.verdict else ""
        if dispatch_result.success and verdict in ("accepted", "improvement", "accept"):
            arm.alpha += 1.0
            arm.total_successes += 1
            arm.consecutive_failures = 0
        else:
            arm.beta += 1.0
            arm.consecutive_failures += 1

        arm.total_attempts += 1
        arm.last_selected = time.time()

        # Recompute temperature using cooling schedule
        arm.temperature = self._compute_temperature(arm, new_state)

        # Decay all boosts by 50%
        new_state = self.booster.decay_all_boosts(new_state)

        # Apply delayed corrections
        self._apply_delayed_corrections(new_state)

        # Increment global iteration
        new_state.global_iteration += 1
        new_state.metadata["last_updated"] = time.time()

        # Try to save state (transactional)
        try:
            save_json(new_state, self.state_path)
        except Exception:
            # Save failed — return original state
            return state

        # Log the update
        if log_writer is not None:
            log_entry = {
                "type": "posterior_update",
                "iteration": new_state.global_iteration,
                "arm_id": arm_id,
                "verdict": dispatch_result.verdict,
                "success": dispatch_result.success,
                "delta": dispatch_result.delta,
                "alpha": arm.alpha,
                "beta": arm.beta,
                "temperature": arm.temperature,
                "timestamp": time.time(),
            }
            if hasattr(log_writer, "write"):
                log_writer.write(log_entry)
            elif hasattr(log_writer, "append"):
                log_writer.append(log_entry)
            elif callable(log_writer):
                log_writer(log_entry)

        return new_state

    def _compute_temperature(self, arm: ArmState, state: BanditState) -> float:
        """Compute temperature using exponential cooling with reheat awareness."""
        # Base cooling: T = T_base / (1 + total_attempts * 0.1)
        t = state.T_base / (1.0 + arm.total_attempts * 0.1)

        # Enforce minimum
        t = max(t, state.min_temperature)

        return t

    def _apply_delayed_corrections(self, state: BanditState):
        """Apply any queued delayed corrections."""
        remaining = []
        for correction in state.delayed_corrections:
            arm_id = correction.get("arm_id", "")
            if arm_id in state.arms and isinstance(state.arms[arm_id], ArmState):
                beta_add = correction.get("beta_add", 0.0)
                state.arms[arm_id].beta += beta_add
            else:
                # Keep if arm not found yet
                remaining.append(correction)
        state.delayed_corrections = remaining
