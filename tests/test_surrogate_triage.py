import os
import pytest
from surrogate_triage.pipeline import SurrogateTriagePipeline
from surrogate_triage.schemas import ExtractionQualityRecord, PaperMetadata
from surrogate_triage.ingestion.paper_filter import PaperFilter
from surrogate_triage.intelligence.diagnostics_linker import DiagnosticsIngestionLinker


def test_surrogate_triage_instantiation(tmp_data_dir, mock_train_source):
    train_path = os.path.join(tmp_data_dir, "train.py")
    with open(train_path, "w") as f:
        f.write(mock_train_source)

    pipeline = SurrogateTriagePipeline(train_path=train_path, data_dir=tmp_data_dir)
    pipeline.initialize()
    assert pipeline._initialized


# ---------------------------------------------------------------------------
# ExtractionQualityRecord from_dict tests
# ---------------------------------------------------------------------------

def test_extraction_quality_record_from_dict():
    """from_dict should round-trip with to_dict."""
    original = ExtractionQualityRecord(
        paper_id="paper_001",
        technique_id="tech_001",
        surrogate_predicted_delta=-0.01,
        actual_delta=-0.005,
        prediction_error=0.005,
        extraction_confidence=0.8,
        quality_score=0.75,
    )
    data = original.to_dict()
    restored = ExtractionQualityRecord.from_dict(data)
    assert restored.paper_id == "paper_001"
    assert restored.technique_id == "tech_001"
    assert restored.quality_score == 0.75


# ---------------------------------------------------------------------------
# Paper filter normalization tests
# ---------------------------------------------------------------------------

def test_paper_filter_accepts_dicts():
    """filter_papers should accept dicts and normalize them to PaperMetadata."""
    f = PaperFilter()
    paper_dict = {
        "title": "Attention Is All You Need",
        "abstract": "We propose a new simple network architecture attention transformer",
        "arxiv_id": "1706.03762",
    }
    # Should not crash with dict input
    relevant, filtered = f.filter_papers([paper_dict])
    # All papers should be categorized
    assert len(relevant) + len(filtered) == 1


def test_paper_filter_keyword_scoring():
    """Papers with relevant keywords should score higher."""
    f = PaperFilter()
    good_paper = PaperMetadata(
        title="Novel Attention Mechanism for Transformer Training",
        abstract="attention mechanism transformer architecture training optimization gradient",
    )
    bad_paper = PaperMetadata(
        title="Cooking Recipes Database",
        abstract="food recipes cooking database management",
    )
    score_good = f.score(good_paper)
    score_bad = f.score(bad_paper)
    assert score_good > score_bad


# ---------------------------------------------------------------------------
# Diagnostics linker / BottleneckMapper wiring tests
# ---------------------------------------------------------------------------

def test_diagnostics_linker_populates_search_terms():
    """detect_bottlenecks should populate search_terms from BottleneckMapper."""
    linker = DiagnosticsIngestionLinker()
    diagnostics = {
        "attention_stats": [
            {"entropy": 0.1, "layer_idx": 0, "head_idx": 0},
        ],
    }
    bottlenecks = linker.detect_bottlenecks(diagnostics)
    assert len(bottlenecks) >= 1
    # search_terms should be populated, not empty
    bn = bottlenecks[0]
    assert bn["bottleneck_type"] == "attention_entropy_collapse"
    assert len(bn["search_terms"]) > 0
    assert bn["boost_weight"] > 0.0
