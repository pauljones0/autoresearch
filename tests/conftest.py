import os
import pytest
import subprocess

@pytest.fixture
def mock_train_source():
    return "def train():\n    pass\n"

@pytest.fixture
def tmp_data_dir(tmp_path):
    d = tmp_path / "data"
    d.mkdir()
    # Create an empty journal file to satisfy checks
    journal = d / "hypothesis_journal.jsonl"
    journal.touch()
    return str(d)

@pytest.fixture(autouse=True)
def mock_run_training(monkeypatch):
    class MockCompletedProcess:
        def __init__(self):
            self.returncode = 0
            self.stdout = '{"val_bpb": 0.99, "peak_vram_mb": 4000}'
            self.stderr = ''
    
    def fake_run(*args, **kwargs):
        return MockCompletedProcess()
    
    monkeypatch.setattr(subprocess, "run", fake_run)
    
    # Also patch anything else that might be used to run training if we know it
    # Just returning the mock function to be safe
    def mock_eval(*args, **kwargs):
        return {"val_bpb": 0.99, "peak_vram_mb": 4000}
    return mock_eval