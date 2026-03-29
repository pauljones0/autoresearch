import random
import pytest
from meta.pipeline import MetaAutoresearchPipeline
from meta.schemas import MetaBanditState, DimensionState, MetaExperimentResult
from meta.bandit.meta_bandit import MetaBandit
from meta.bandit.meta_updater import MetaPosteriorUpdater
from meta.bandit.discretizer import MetaVariantDiscretizer
from meta.convergence.divergence import DivergenceWatcher


def test_meta_instantiation(tmp_data_dir):
    pipeline = MetaAutoresearchPipeline(work_dir=tmp_data_dir)
    pipeline.initialize()
    assert pipeline.state is not None


# ---------------------------------------------------------------------------
# Meta-bandit Thompson sampling tests
# ---------------------------------------------------------------------------

def _make_meta_state():
    """Create a MetaBanditState with one dimension where variant 'high' dominates."""
    state = MetaBanditState()
    state.dimensions["test_param"] = DimensionState(
        param_id="test_param",
        variants=[0.01, 0.025, 0.05],
        variant_posteriors={
            "0.01": {"alpha": 2.0, "beta": 10.0},    # weak
            "0.025": {"alpha": 50.0, "beta": 2.0},   # strong
            "0.05": {"alpha": 2.0, "beta": 5.0},      # medium
        },
        current_best=0.025,
    )
    return state


def test_meta_bandit_selects_highest_sampled():
    """Meta-bandit should predominantly select the variant with the best posterior."""
    bandit = MetaBandit()
    state = _make_meta_state()

    counts = {0.01: 0, 0.025: 0, 0.05: 0}
    n = 200
    for i in range(n):
        config = bandit.select(state, rng=random.Random(i))
        val = config["test_param"]
        counts[val] += 1

    assert counts[0.025] > counts[0.01]
    assert counts[0.025] > counts[0.05]


# ---------------------------------------------------------------------------
# Three-zone scoring tests
# ---------------------------------------------------------------------------

def test_posterior_updater_three_zones():
    """MetaPosteriorUpdater should classify results into better/worse/inconclusive."""
    updater = MetaPosteriorUpdater()
    state = _make_meta_state()

    # Result that is clearly better than baseline
    result = MetaExperimentResult(
        experiment_id="exp_1",
        config_diff={"test_param": 0.025},
        improvement_rate=0.15,
        raw_deltas=[-0.01, -0.02, 0.005],
    )
    updated = updater.update(state, result, baseline_ir=0.05, baseline_std=0.02)
    assert result.compared_to_baseline == "better"

    # Result that is clearly worse
    result2 = MetaExperimentResult(
        experiment_id="exp_2",
        config_diff={"test_param": 0.01},
        improvement_rate=0.01,
        raw_deltas=[0.01, 0.02, 0.03],
    )
    updater.update(updated, result2, baseline_ir=0.10, baseline_std=0.02)
    assert result2.compared_to_baseline == "worse"


# ---------------------------------------------------------------------------
# Discretizer tests
# ---------------------------------------------------------------------------

def test_discretizer_bool():
    """Bool parameters should produce [True, False]."""
    from meta.schemas import MetaParameter
    disc = MetaVariantDiscretizer()
    param = MetaParameter(param_id="enable_x", type="bool", default_value=True)
    variants = disc.discretize(param)
    assert set(variants) == {True, False}


def test_discretizer_int():
    """Int parameters should produce valid discrete variants."""
    from meta.schemas import MetaParameter
    disc = MetaVariantDiscretizer()
    param = MetaParameter(
        param_id="batch_size", type="int", default_value=128,
        valid_range={"min": 32, "max": 512},
    )
    variants = disc.discretize(param)
    assert len(variants) >= 2
    assert all(isinstance(v, int) for v in variants)
    assert 128 in variants  # default should be included


def test_discretizer_float():
    """Float parameters should produce valid discrete variants."""
    from meta.schemas import MetaParameter
    disc = MetaVariantDiscretizer()
    param = MetaParameter(
        param_id="learning_rate", type="float", default_value=0.01,
        valid_range={"min": 0.001, "max": 0.1},
    )
    variants = disc.discretize(param)
    assert len(variants) >= 2


# ---------------------------------------------------------------------------
# DivergenceWatcher tests
# ---------------------------------------------------------------------------

def test_divergence_watcher_returns_instructions_not_mutates():
    """DivergenceWatcher.check() should return alert without mutating state."""
    watcher = DivergenceWatcher(consecutive_threshold=2)
    state = _make_meta_state()

    # Store original posteriors
    original_alpha = state.dimensions["test_param"].variant_posteriors["0.025"]["alpha"]

    # Create windows that drop below baseline
    windows = [{"mean_ir": 0.01}, {"mean_ir": 0.01}, {"mean_ir": 0.01}]
    alert = watcher.check(state, windows, baseline_ir={"mean_ir": 0.10, "std_ir": 0.01})

    assert alert is not None
    assert alert.triggered is True
    # State should NOT have been mutated
    assert state.dimensions["test_param"].variant_posteriors["0.025"]["alpha"] == original_alpha
    # Alert recommendation should contain mutation instructions
    assert "soft_reset" in alert.recommendation
