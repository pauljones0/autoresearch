import random
import pytest
from bandit.pipeline import AdaptiveBanditPipeline
from bandit.schemas import BanditState, ArmState, SelectionResult
from bandit.sampler import ThompsonSamplerEngine
from bandit.acceptance import AnnealingAcceptanceEngine
from bandit.ceiling_bridge import CeilingMonitorBanditBridge


def test_bandit_instantiation(tmp_data_dir):
    pipeline = AdaptiveBanditPipeline(work_dir=tmp_data_dir)
    pipeline.initialize()
    assert pipeline.state is not None


# ---------------------------------------------------------------------------
# Thompson Sampling tests
# ---------------------------------------------------------------------------

def _make_state_with_arms():
    """Create a BanditState with 3 arms having distinct posteriors."""
    state = BanditState(regime="full_bandit", exploration_floor=0.0)
    state.arms = {
        "strong": ArmState(alpha=50.0, beta=2.0, source_type="internal"),
        "weak": ArmState(alpha=2.0, beta=50.0, source_type="internal"),
        "medium": ArmState(alpha=10.0, beta=10.0, source_type="internal"),
    }
    return state


def test_thompson_selects_highest_sampled():
    """Thompson sampling should predominantly select the arm with the best posterior."""
    sampler = ThompsonSamplerEngine()
    state = _make_state_with_arms()

    counts = {"strong": 0, "weak": 0, "medium": 0}
    n = 200
    for i in range(n):
        rng = random.Random(i)
        result = sampler.select(state, rng=rng)
        counts[result.arm_id] += 1

    # The strong arm (alpha=50, beta=2) should be selected most often
    assert counts["strong"] > counts["weak"]
    assert counts["strong"] > counts["medium"]


def test_thompson_sample_values_always_populated():
    """sample_values should be populated regardless of exploration vs exploitation."""
    sampler = ThompsonSamplerEngine()
    state = _make_state_with_arms()
    # Force high exploration to test the exploration path
    state.exploration_floor = 1.0

    rng = random.Random(42)
    result = sampler.select(state, rng=rng)
    assert result.sample_values, "sample_values should not be empty"
    assert len(result.sample_values) == 3


def test_thompson_dispatch_path_from_source_type():
    """dispatch_path should reflect the arm's source_type."""
    sampler = ThompsonSamplerEngine()
    state = BanditState(regime="full_bandit", exploration_floor=0.0)
    state.arms = {
        "paper_arm": ArmState(alpha=100.0, beta=1.0, source_type="paper"),
    }
    result = sampler.select(state, rng=random.Random(0))
    assert result.dispatch_path == "paper"


# ---------------------------------------------------------------------------
# Acceptance engine tests
# ---------------------------------------------------------------------------

def test_acceptance_always_accepts_improvement():
    """Negative delta (improvement) should always be accepted."""
    engine = AnnealingAcceptanceEngine()
    state = BanditState(T_base=0.025, min_temperature=0.001)
    arm = ArmState(temperature=0.02)

    decision = engine.decide(-0.05, arm, state, rng=random.Random(0))
    assert decision.accepted is True
    assert decision.accepted_by == "improvement"


def test_acceptance_higher_temp_more_risk():
    """Higher T_base should give higher acceptance probability for regressions."""
    engine = AnnealingAcceptanceEngine()
    arm = ArmState(temperature=0.02)

    # Low temperature
    state_low = BanditState(T_base=0.001, min_temperature=0.0001)
    decision_low = engine.decide(0.05, arm, state_low, rng=random.Random(99))

    # High temperature
    state_high = BanditState(T_base=0.5, min_temperature=0.001)
    decision_high = engine.decide(0.05, arm, state_high, rng=random.Random(99))

    assert decision_high.probability > decision_low.probability


def test_acceptance_rejects_regression_at_zero_temp():
    """Zero effective temperature should never accept regressions."""
    engine = AnnealingAcceptanceEngine()
    state = BanditState(T_base=0.0, min_temperature=0.0)
    arm = ArmState(temperature=0.0)

    decision = engine.decide(0.05, arm, state, rng=random.Random(0))
    assert decision.accepted is False


# ---------------------------------------------------------------------------
# Ceiling bridge tests
# ---------------------------------------------------------------------------

def test_ceiling_bridge_boosts_diagnostics_not_alpha():
    """ceiling_bridge should boost diagnostics_boost, not alpha."""
    bridge = CeilingMonitorBanditBridge()
    state = BanditState()
    state.arms = {
        "paper_arm": ArmState(alpha=5.0, diagnostics_boost=0.0, source_type="paper"),
        "kernel_arm": ArmState(alpha=5.0, diagnostics_boost=0.0, source_type="kernel"),
    }

    report = {"paper_fraction_trend": 0.3, "kernel_fraction_trend": 0.2}
    result = bridge.apply_ceiling_signal(state, report)

    # alpha should be unchanged for both
    assert result.arms["paper_arm"].alpha == 5.0
    assert result.arms["kernel_arm"].alpha == 5.0
    # diagnostics_boost should have increased
    assert result.arms["paper_arm"].diagnostics_boost > 0.0
    assert result.arms["kernel_arm"].diagnostics_boost > 0.0
