import pytest
from surrogate_triage.pipeline import SurrogateTriagePipeline
import os

def test_surrogate_triage_instantiation(tmp_data_dir, mock_train_source):
    train_path = os.path.join(tmp_data_dir, "train.py")
    with open(train_path, "w") as f:
        f.write(mock_train_source)
        
    pipeline = SurrogateTriagePipeline(train_path=train_path, data_dir=tmp_data_dir)
    pipeline.initialize()
    assert pipeline._initialized
