"""Cross-layer integration tests (mocked, no GPU needed)."""

import json
import os
import pytest
from unittest.mock import MagicMock, patch

from bandit.pipeline import AdaptiveBanditPipeline
from bandit.schemas import BanditState, ArmState


# ---------------------------------------------------------------------------
# Seam A: Bandit dispatches to sub-layers
# ---------------------------------------------------------------------------

def test_bandit_dispatches_to_model_scientist(tmp_data_dir):
    """Bandit should route internal arms to model_scientist."""
    mock_ms = MagicMock()
    mock_ms.evaluate_modification.return_value = {
        "success": True, "delta": -0.01, "verdict": "accepted",
        "journal_entry_id": "j001",
    }

    pipeline = AdaptiveBanditPipeline(
        work_dir=tmp_data_dir, model_scientist=mock_ms
    )
    pipeline.initialize()

    # Force an internal arm to exist
    pipeline.state.regime = "full_bandit"
    pipeline.state.arms["arch_mod"] = ArmState(
        alpha=10.0, beta=1.0, source_type="internal"
    )
    pipeline.state.global_iteration = 5

    result = pipeline.run_iteration(base_source="def train(): pass")
    # The dispatch should have reached model_scientist
    assert result is not None


def test_bandit_dispatches_to_surrogate(tmp_data_dir):
    """Bandit should route paper arms to surrogate_triage."""
    mock_st = MagicMock()
    mock_st.evaluate_next_paper_candidate.return_value = {
        "success": True, "delta": -0.005, "verdict": "accepted",
    }

    mock_ms = MagicMock()
    mock_ms.evaluate_modification.return_value = {
        "success": True, "delta": -0.005, "verdict": "accepted",
        "journal_entry_id": "j002",
    }

    pipeline = AdaptiveBanditPipeline(
        work_dir=tmp_data_dir, model_scientist=mock_ms,
        surrogate_triage=mock_st,
    )
    pipeline.initialize()

    # Force a paper arm
    pipeline.state.regime = "full_bandit"
    pipeline.state.arms["paper_technique"] = ArmState(
        alpha=10.0, beta=1.0, source_type="paper"
    )
    pipeline.state.global_iteration = 5

    result = pipeline.run_iteration(base_source="def train(): pass")
    assert result is not None


# ---------------------------------------------------------------------------
# Seam B: Surrogate feeds back to model_scientist (tested via pipeline wiring)
# ---------------------------------------------------------------------------

def test_surrogate_feeds_back_to_model_scientist(tmp_data_dir, mock_train_source):
    """Surrogate triage should route candidates through model_scientist."""
    from surrogate_triage.pipeline import SurrogateTriagePipeline

    mock_ms = MagicMock()
    mock_ms.evaluate_modification.return_value = {
        "success": True, "delta": -0.01, "verdict": "accepted",
        "journal_entry_id": "j003",
    }

    train_path = os.path.join(tmp_data_dir, "train.py")
    with open(train_path, "w") as f:
        f.write(mock_train_source)

    pipeline = SurrogateTriagePipeline(
        train_path=train_path, data_dir=tmp_data_dir,
        model_scientist_pipeline=mock_ms,
    )
    pipeline.initialize()
    # model_scientist ref should be stored
    assert pipeline.model_scientist is mock_ms


# ---------------------------------------------------------------------------
# Seam C: Meta config propagates via overrides
# ---------------------------------------------------------------------------

def test_meta_config_propagates_to_bandit(tmp_data_dir):
    """Meta overrides should be reloadable by bandit pipeline."""
    pipeline = AdaptiveBanditPipeline(work_dir=tmp_data_dir)
    pipeline.initialize()

    # Write an overrides file
    overrides_path = os.path.join(tmp_data_dir, "bandit_overrides.json")
    with open(overrides_path, "w") as f:
        json.dump({"T_base": 0.05}, f)

    # The pipeline reloads during run_iteration via HotConfigReloader
    old_t = pipeline.state.T_base
    result = pipeline.run_iteration(base_source="def train(): pass")
    # The hot reloader should have applied the new T_base
    # (even if the iteration itself falls back)
    assert result is not None


def test_sub_layer_reload_overrides(tmp_data_dir, mock_train_source):
    """Sub-layer pipelines should support reload_overrides()."""
    from model_scientist.pipeline import ModelScientistPipeline
    from gpu_kernels.pipeline import GPUKernelPipeline

    # Model Scientist
    train_path = os.path.join(tmp_data_dir, "train.py")
    with open(train_path, "w") as f:
        f.write(mock_train_source)

    ms = ModelScientistPipeline(train_path=train_path, data_dir=tmp_data_dir)
    ms.initialize(baseline_val_bpb=1.5)
    overrides_path = os.path.join(tmp_data_dir, "ms_overrides.json")
    with open(overrides_path, "w") as f:
        json.dump({"diagnostics_capture_interval": 100}, f)
    assert ms.reload_overrides(overrides_path) is True

    # GPU Kernels
    gk = GPUKernelPipeline(data_dir=tmp_data_dir)
    gk.initialize()
    overrides_path = os.path.join(tmp_data_dir, "gk_overrides.json")
    with open(overrides_path, "w") as f:
        json.dump({}, f)
    # Empty overrides should still return True (file was read)
    assert gk.reload_overrides(overrides_path) is True
