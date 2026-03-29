import os
import json
import pytest
from model_scientist.pipeline import ModelScientistPipeline
from model_scientist.ablation.diff_parser import DiffParser
from model_scientist.metrics.registry import MetricRegistry
from model_scientist.journal.writer import JournalWriter


def test_model_scientist_instantiation(tmp_data_dir, mock_train_source):
    train_path = os.path.join(tmp_data_dir, "train.py")
    with open(train_path, "w") as f:
        f.write(mock_train_source)

    pipeline = ModelScientistPipeline(train_path=train_path, data_dir=tmp_data_dir)
    pipeline.initialize(baseline_val_bpb=1.5)
    assert pipeline.loop._baseline_val_bpb == 1.5


# ---------------------------------------------------------------------------
# DiffParser tests
# ---------------------------------------------------------------------------

def test_diff_parser_extracts_regions():
    """DiffParser should find changed regions between two sources."""
    parser = DiffParser()
    old_src = "def train():\n    lr = 0.01\n    epochs = 10\n"
    new_src = "def train():\n    lr = 0.001\n    epochs = 10\n"

    components = parser.parse(old_src, new_src)
    assert len(components) >= 1
    assert components[0].diff != ""


def test_diff_parser_empty_on_identical():
    """DiffParser should return empty list when sources are identical."""
    parser = DiffParser()
    src = "def train():\n    pass\n"
    components = parser.parse(src, src)
    assert components == []


def test_diff_parser_multiple_regions():
    """DiffParser should decompose changes in different AST nodes."""
    parser = DiffParser()
    old_src = (
        "# Hyperparameters\n"
        "lr = 0.01\n"
        "batch_size = 128\n"
        "\n"
        "# GPT Model\n"
        "class GPT(nn.Module):\n"
        "    def forward(self):\n"
        "        return 0\n"
    )
    new_src = (
        "# Hyperparameters\n"
        "lr = 0.001\n"
        "batch_size = 256\n"
        "\n"
        "# GPT Model\n"
        "class GPT(nn.Module):\n"
        "    def forward(self):\n"
        "        return 1\n"
    )
    components = parser.parse(old_src, new_src)
    # Should have at least 1 component (may group or split)
    assert len(components) >= 1


# ---------------------------------------------------------------------------
# MetricRegistry tests
# ---------------------------------------------------------------------------

def test_metric_registry_initializes_defaults(tmp_path):
    """MetricRegistry should initialize 5 default metrics on first load."""
    path = str(tmp_path / "metric_registry.json")
    registry = MetricRegistry(path=path)
    registry.load()

    active = registry.get_active()
    assert len(active) == 5

    names = {m.name for m in active}
    assert "gradient_norm_mean" in names
    assert "dead_neuron_fraction" in names


def test_metric_registry_roundtrip(tmp_path):
    """MetricRegistry should round-trip save/load."""
    path = str(tmp_path / "metric_registry.json")
    registry = MetricRegistry(path=path)
    registry.load()  # initializes defaults

    registry.save()

    registry2 = MetricRegistry(path=path)
    registry2.load()
    assert len(registry2.get_active()) == 5


# ---------------------------------------------------------------------------
# JournalWriter tests
# ---------------------------------------------------------------------------

def test_journal_writer_roundtrip(tmp_path):
    """JournalWriter should write entries that can be read back."""
    path = str(tmp_path / "journal.jsonl")
    writer = JournalWriter(path=path)

    entry = writer.log_experiment(
        hypothesis="test hypothesis",
        predicted_delta=-0.01,
        actual_delta=-0.005,
        modification_diff="diff here",
        verdict="accepted",
    )
    assert entry.id != ""

    # Read back
    with open(path) as f:
        lines = f.readlines()
    assert len(lines) == 1
    data = json.loads(lines[0])
    assert data["hypothesis"] == "test hypothesis"
    assert data["verdict"] == "accepted"


def test_journal_writer_update_entry(tmp_path):
    """update_entry should append a follow-up entry referencing the original."""
    path = str(tmp_path / "journal.jsonl")
    writer = JournalWriter(path=path)

    original = writer.log_experiment(
        hypothesis="original",
        predicted_delta=-0.01,
        actual_delta=-0.005,
        modification_diff="diff",
        verdict="accepted",
    )

    update = writer.update_entry(
        original.id,
        actual_delta=-0.008,
        verdict="re-evaluated",
    )

    # Should have 2 entries
    with open(path) as f:
        lines = f.readlines()
    assert len(lines) == 2

    update_data = json.loads(lines[1])
    assert original.id in update_data["tags"]
