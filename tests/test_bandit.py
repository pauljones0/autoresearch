import pytest
from bandit.pipeline import AdaptiveBanditPipeline

def test_bandit_instantiation(tmp_data_dir):
    pipeline = AdaptiveBanditPipeline(work_dir=tmp_data_dir)
    pipeline.initialize()
    assert pipeline.state is not None
