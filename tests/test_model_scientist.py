import pytest
from model_scientist.pipeline import ModelScientistPipeline
import os

def test_model_scientist_instantiation(tmp_data_dir, mock_train_source):
    train_path = os.path.join(tmp_data_dir, "train.py")
    with open(train_path, "w") as f:
        f.write(mock_train_source)
        
    pipeline = ModelScientistPipeline(train_path=train_path, data_dir=tmp_data_dir)
    pipeline.initialize(baseline_val_bpb=1.5)
    assert pipeline.loop._baseline_val_bpb == 1.5
