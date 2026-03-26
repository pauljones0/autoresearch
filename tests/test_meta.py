import pytest
from meta.pipeline import MetaAutoresearchPipeline

def test_meta_instantiation(tmp_data_dir):
    pipeline = MetaAutoresearchPipeline(work_dir=tmp_data_dir)
    pipeline.initialize()
    assert pipeline.state is not None
