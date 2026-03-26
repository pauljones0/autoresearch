"""
Diagnostics-driven arm boosting for Thompson Sampling.
"""

import copy

from bandit.schemas import BanditState, ArmState


# Rule table: diagnostic signal -> (arm_id pattern, boost amount)
BOOST_RULES = [
    # Attention entropy issues boost architecture arm
    ("attention_entropy_high", "architecture", 1.5),
    ("attention_entropy_low", "architecture", 0.5),
    # Gradient norm issues boost optimizer arm
    ("gradient_norm_exploding", "optimizer", 2.0),
    ("gradient_norm_vanishing", "optimizer", 1.5),
    ("gradient_norm_unstable", "optimizer", 1.0),
    # Dead neurons boost activation and initialization arms
    ("dead_neurons_high", "activation", 1.5),
    ("dead_neurons_high", "initialization", 1.0),
    # Loss decomposition signals
    ("loss_bias_high", "regularization", 1.0),
    ("loss_variance_high", "hyperparameter", 1.0),
    ("loss_plateau", "scheduling", 1.5),
    ("loss_plateau", "optimizer", 0.5),
    # Kernel profiling signals
    ("kernel_slow", "architecture", 0.5),
    ("kernel_memory_bound", "architecture", 1.0),
    ("kernel_compute_bound", "optimizer", 0.5),
]


class DiagnosticsArmBooster:
    """Computes and applies diagnostic-driven boosts to arm priors."""

    def compute_boosts(self, diagnostics_report, taxonomy, state: BanditState) -> dict:
        """Compute per-arm boosts from diagnostic signals.

        Args:
            diagnostics_report: Object or dict with diagnostic signals.
            taxonomy: Arm taxonomy (used for arm_id lookup).
            state: Current bandit state.

        Returns:
            dict mapping arm_id -> boost amount.
        """
        boosts = {arm_id: 0.0 for arm_id in state.arms}

        # Extract signals from diagnostics report
        if diagnostics_report is None:
            return boosts

        signals = set()
        if isinstance(diagnostics_report, dict):
            signals = set(diagnostics_report.get("signals", []))
        elif hasattr(diagnostics_report, "signals"):
            signals = set(diagnostics_report.signals)

        # Apply rule table
        for signal, arm_pattern, boost_amount in BOOST_RULES:
            if signal in signals:
                for arm_id in state.arms:
                    if arm_pattern in arm_id:
                        boosts[arm_id] += boost_amount

        return boosts

    def apply_boosts(self, state: BanditState, boosts: dict) -> BanditState:
        """Apply boosts to a copy of state for sampling (ephemeral).

        Does not modify the original state.
        """
        new_state = copy.deepcopy(state)
        for arm_id, boost in boosts.items():
            if arm_id in new_state.arms and isinstance(new_state.arms[arm_id], ArmState):
                new_state.arms[arm_id].diagnostics_boost += boost
        return new_state

    def decay_all_boosts(self, state: BanditState) -> BanditState:
        """Decay all diagnostics boosts by 50% each iteration."""
        new_state = copy.deepcopy(state)
        for arm_id, arm in new_state.arms.items():
            if isinstance(arm, ArmState):
                arm.diagnostics_boost *= 0.5
                # Zero out tiny values
                if abs(arm.diagnostics_boost) < 0.001:
                    arm.diagnostics_boost = 0.0
        return new_state
