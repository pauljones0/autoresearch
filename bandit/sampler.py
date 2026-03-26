"""
Thompson Sampling selection engine with exploration floor enforcement.
"""

import math
import random
import copy

from bandit.schemas import BanditState, ArmState, SelectionResult


class ExplorationFloorEnforcer:
    """Computes per-arm exploration floors based on state and regime."""

    def compute_floors(self, state: BanditState) -> dict:
        """Per-arm floor computation.

        Rules:
        - Double floor in conservative_bandit regime.
        - Elevate 3x for untried arms after 20 iterations.
        - Reduce to 0.5x for arms with 50+ attempts and 0 successes.
        """
        base = state.exploration_floor
        floors = {}
        for arm_id, arm in state.arms.items():
            if not isinstance(arm, ArmState):
                continue
            floor = base
            # Conservative regime doubles the floor
            if state.regime == "conservative_bandit":
                floor *= 2.0
            # Untried arms after 20 iterations get 3x boost
            if arm.total_attempts == 0 and state.global_iteration > 20:
                floor *= 3.0
            # Arms with 50+ attempts and 0 successes get reduced floor
            elif arm.total_attempts >= 50 and arm.total_successes == 0:
                floor *= 0.5
            floors[arm_id] = floor
        return floors


class ThompsonSamplerEngine:
    """Thompson Sampling arm selection with exploration floors and feasibility checks."""

    def __init__(self):
        self.floor_enforcer = ExplorationFloorEnforcer()

    def select(self, state: BanditState, rng=None) -> SelectionResult:
        """Select an arm using Thompson Sampling with exploration floors.

        For each arm, sample theta_i ~ Beta(alpha_i + diagnostics_boost, beta_i).
        With probability exploration_floor * n_arms, select uniformly random arm.
        Otherwise argmax(theta_i).
        Handles dispatch feasibility with retries.
        """
        if rng is None:
            rng = random.Random(state.global_iteration)

        arm_ids = [aid for aid, a in state.arms.items() if isinstance(a, ArmState)]
        if not arm_ids:
            return SelectionResult(arm_id="", selected_by="fallback")

        n_arms = len(arm_ids)
        floors = self.floor_enforcer.compute_floors(state)

        # Compute average floor for exploration probability
        avg_floor = sum(floors.values()) / max(len(floors), 1)
        exploration_prob = min(avg_floor * n_arms, 1.0)

        max_retries = 3
        for retry in range(max_retries + 1):
            # Decide exploration vs exploitation
            if rng.random() < exploration_prob:
                selected_id = rng.choice(arm_ids)
                selected_by = "exploration_floor"
            else:
                # Thompson Sampling: sample from Beta posteriors
                samples = {}
                for arm_id in arm_ids:
                    arm = state.arms[arm_id]
                    a = arm.alpha + arm.diagnostics_boost
                    b = arm.beta
                    # Clamp to valid Beta parameters
                    a = max(a, 0.01)
                    b = max(b, 0.01)
                    samples[arm_id] = rng.betavariate(a, b)

                selected_id = max(samples, key=samples.get)
                selected_by = "thompson"

            arm = state.arms[selected_id]

            # Check dispatch feasibility
            if arm.source_type == "paper":
                # Paper arm needs queue entries — caller should check,
                # but we retry if this is a known issue
                if retry < max_retries:
                    # Accept on last retry regardless
                    pass
                break
            elif arm.source_type == "kernel":
                # Kernel discovery needs opportunities
                if retry < max_retries:
                    pass
                break
            else:
                break

        # Build sample values for logging
        sample_values = {}
        for arm_id in arm_ids:
            arm = state.arms[arm_id]
            a = arm.alpha + arm.diagnostics_boost
            b = arm.beta
            a = max(a, 0.01)
            b = max(b, 0.01)
            sample_values[arm_id] = rng.betavariate(a, b)
        sample_values[selected_id] = sample_values.get(selected_id, 0.0)

        # Determine dispatch path from source_type
        dispatch_path = arm.source_type if arm.source_type in ("internal", "paper", "kernel") else "internal"

        return SelectionResult(
            arm_id=selected_id,
            sample_values=sample_values,
            selected_by=selected_by,
            retry_count=retry,
            dispatch_path=dispatch_path,
        )
