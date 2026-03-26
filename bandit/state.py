"""
Bandit state validation utilities.
"""

from bandit.schemas import BanditState

_VALID_REGIMES = {"no_bandit", "conservative_bandit", "full_bandit"}


def validate_state(state: BanditState) -> list[str]:
    """Validate a BanditState object. Returns a list of issue strings (empty = valid).

    Performs all checks from BanditState.validate() plus additional
    structural and semantic checks.
    """
    issues = state.validate()

    # Regime check
    if state.regime not in _VALID_REGIMES:
        issues.append(f"Unknown regime: {state.regime!r}")

    # Temperature bounds
    if state.T_base <= 0:
        issues.append(f"T_base={state.T_base} must be > 0")
    if state.min_temperature < 0:
        issues.append(f"min_temperature={state.min_temperature} must be >= 0")
    if state.min_temperature > state.T_base:
        issues.append(
            f"min_temperature={state.min_temperature} > T_base={state.T_base}")

    # Reheat factor
    if state.reheat_factor < 1:
        issues.append(f"reheat_factor={state.reheat_factor} must be >= 1")

    # Exploration floor
    if not (0 <= state.exploration_floor <= 1):
        issues.append(
            f"exploration_floor={state.exploration_floor} not in [0, 1]")

    # Paper preference ratio
    if not (0 <= state.paper_preference_ratio <= 1):
        issues.append(
            f"paper_preference_ratio={state.paper_preference_ratio} not in [0, 1]")

    # K_reheat_threshold
    if state.K_reheat_threshold < 1:
        issues.append(
            f"K_reheat_threshold={state.K_reheat_threshold} must be >= 1")

    # Metadata
    meta = state.metadata
    if not isinstance(meta, dict):
        issues.append("metadata must be a dict")
    else:
        if "schema_version" not in meta:
            issues.append("metadata missing schema_version")

    # Per-arm consistency
    from bandit.schemas import ArmState
    for arm_id, arm in state.arms.items():
        if not isinstance(arm, ArmState):
            continue
        if arm.total_attempts < 0:
            issues.append(f"Arm {arm_id}: total_attempts={arm.total_attempts} < 0")
        if arm.total_successes < 0:
            issues.append(f"Arm {arm_id}: total_successes={arm.total_successes} < 0")
        if arm.total_successes > arm.total_attempts:
            issues.append(
                f"Arm {arm_id}: total_successes={arm.total_successes} > "
                f"total_attempts={arm.total_attempts}")
        if arm.consecutive_failures < 0:
            issues.append(
                f"Arm {arm_id}: consecutive_failures={arm.consecutive_failures} < 0")

    return issues
