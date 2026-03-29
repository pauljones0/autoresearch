"""
End-to-end test: runs one actual train.py execution.
Requires GPU and real training data — skipped by default.

Run with: pytest -m e2e
"""

import os
import pytest

pytestmark = pytest.mark.e2e


@pytest.mark.skipif(
    not os.environ.get("AUTORESEARCH_E2E"),
    reason="E2E tests require AUTORESEARCH_E2E=1 and GPU hardware"
)
def test_single_real_iteration():
    """Run one actual train.py execution and verify results flow through."""
    from bandit.pipeline import AdaptiveBanditPipeline
    from model_scientist.pipeline import ModelScientistPipeline

    data_dir = os.environ.get("AUTORESEARCH_DATA_DIR", "./data")
    train_path = os.environ.get("AUTORESEARCH_TRAIN_PATH", "./train.py")

    ms = ModelScientistPipeline(train_path=train_path, data_dir=data_dir)
    ms.initialize(baseline_val_bpb=1.5)

    bandit = AdaptiveBanditPipeline(work_dir=data_dir, model_scientist=ms)
    bandit.initialize()

    with open(train_path) as f:
        source = f.read()

    result = bandit.run_iteration(base_source=source)

    assert result is not None
    assert result.arm_selected != ""
    assert result.verdict != ""
