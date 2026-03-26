"""
Adaptive temperature reheating for stalled arms.
"""

import time
from bandit.schemas import BanditState, ArmState, ReheatEvent


class AdaptiveReheatEngine:
    """Reheats temperature for arms stuck in consecutive failures."""

    def check_and_reheat(
        self,
        state: BanditState,
        log_writer=None,
    ) -> list:
        """Check all arms for reheat eligibility and apply reheats.

        Reheat condition: consecutive_failures >= K_reheat_threshold
        Budget: max 3 reheats per 100 iterations per arm.
        Reheat formula: temperature *= reheat_factor * (1 + 0.1 * reheat_count), cap at 5x.
        If budget exhausted: reduce exploration floor instead.

        Args:
            state: Current bandit state.
            log_writer: Optional object with .write(entry) for logging.

        Returns:
            List of ReheatEvent records.
        """
        events = []
        budget_per_100 = 3

        for arm_id, arm in state.arms.items():
            if not isinstance(arm, ArmState):
                continue

            if arm.consecutive_failures < state.K_reheat_threshold:
                continue

            # Compute budget: 3 reheats per 100 iterations
            max_reheats = max(1, (state.global_iteration // 100 + 1) * budget_per_100)
            budget_remaining = max(0, max_reheats - arm.reheat_count)

            if budget_remaining > 0:
                # Apply reheat
                temp_before = arm.temperature
                multiplier = state.reheat_factor * (1 + 0.1 * arm.reheat_count)
                # Cap at 5x the current temperature
                multiplier = min(multiplier, 5.0)
                arm.temperature = arm.temperature * multiplier
                arm.reheat_count += 1
                arm.consecutive_failures = 0
                arm.last_reheat = time.time()

                event = ReheatEvent(
                    arm_id=arm_id,
                    temperature_before=temp_before,
                    temperature_after=arm.temperature,
                    consecutive_failures_at_trigger=state.K_reheat_threshold,
                    reheat_count=arm.reheat_count,
                    budget_remaining=budget_remaining - 1,
                )
                events.append(event)

                if log_writer is not None:
                    log_writer.write(event.to_dict())
            else:
                # Budget exhausted: reduce exploration floor instead
                state.exploration_floor = max(0.01, state.exploration_floor * 0.9)
                arm.consecutive_failures = 0

        return events
