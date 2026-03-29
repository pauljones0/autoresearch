"""
Shared schemas and data structures for the Surrogate Triage Pipeline.
"""

from dataclasses import dataclass, field, asdict
from typing import Optional
import json
import time


# ---------------------------------------------------------------------------
# Phase 1.1: Paper Metadata
# ---------------------------------------------------------------------------

@dataclass
class PaperMetadata:
    """Metadata for an arXiv paper."""
    arxiv_id: str = ""
    title: str = ""
    abstract: str = ""
    authors: list = field(default_factory=list)
    categories: list = field(default_factory=list)
    pdf_url: str = ""
    published: str = ""  # ISO date string
    fetched_at: float = field(default_factory=time.time)
    relevance_score: float = 0.0
    keyword_score: float = 0.0
    diagnostics_boost: float = 0.0
    source_quality_boost: float = 0.0
    filtered_out: bool = False
    filter_reason: str = ""

    def to_dict(self):
        return asdict(self)

    def to_json_line(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: dict):
        entry = cls()
        for k, v in data.items():
            if hasattr(entry, k):
                setattr(entry, k, v)
        return entry


# ---------------------------------------------------------------------------
# Phase 1.2: Technique Extraction
# ---------------------------------------------------------------------------

@dataclass
class TechniqueDescription:
    """Structured technique description extracted from a paper."""
    technique_id: str = ""
    paper_id: str = ""  # arXiv ID
    name: str = ""
    modification_category: str = ""  # matches FailureExtractor taxonomy
    description: str = ""
    pseudo_code: str = ""
    reported_improvement: str = ""
    baseline: str = ""
    applicability_conditions: list = field(default_factory=list)
    extracted_at: float = field(default_factory=time.time)
    extraction_confidence: float = 0.0
    constraint_matches: list = field(default_factory=list)  # matched NegativeConstraints
    already_explored: bool = False
    journal_entry_id: str = ""  # if already explored
    deduplicated: bool = False
    duplicate_of: str = ""  # technique_id of canonical version

    def to_dict(self):
        return asdict(self)

    def to_json_line(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: dict):
        entry = cls()
        for k, v in data.items():
            if hasattr(entry, k):
                setattr(entry, k, v)
        return entry


# ---------------------------------------------------------------------------
# Phase 1.3: Synthetic Diffs
# ---------------------------------------------------------------------------

@dataclass
class SyntheticDiff:
    """A concrete code diff generated from a technique description."""
    diff_id: str = ""
    technique_id: str = ""
    paper_id: str = ""
    variant_index: int = 0  # 0..N for multiple variants per technique
    diff_text: str = ""  # unified diff format
    modification_category: str = ""
    applies_cleanly: bool = False
    passes_smoke_test: bool = False
    is_decomposable: bool = False  # parseable by DiffParser
    n_components: int = 0
    constraint_penalty: float = 0.0  # penalty from constraint pre-filter
    failure_feature_vector: list = field(default_factory=list)
    generated_at: float = field(default_factory=time.time)

    def to_dict(self):
        return asdict(self)

    def to_json_line(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: dict):
        entry = cls()
        for k, v in data.items():
            if hasattr(entry, k):
                setattr(entry, k, v)
        return entry


# ---------------------------------------------------------------------------
# Phase 2: Surrogate Model
# ---------------------------------------------------------------------------

@dataclass
class EnrichedFeatureVector:
    """Combined feature vector for surrogate input."""
    diff_id: str = ""
    code_embedding: list = field(default_factory=list)  # from DiffEmbedder
    failure_features: list = field(default_factory=list)  # 23-element from FailureExtractor
    metric_features: list = field(default_factory=list)  # from MetricRegistry active metrics
    combined_vector: list = field(default_factory=list)  # concatenated

    def to_dict(self):
        return asdict(self)


@dataclass
class SurrogateTrainingExample:
    """A single training example for the surrogate model."""
    journal_id: str = ""
    feature_vector: list = field(default_factory=list)
    actual_delta: float = 0.0
    source: str = ""  # "internal" or "paper"
    tags: list = field(default_factory=list)


@dataclass
class SurrogatePrediction:
    """Surrogate model prediction for a candidate diff."""
    diff_id: str = ""
    predicted_delta: float = 0.0
    confidence: float = 0.0
    constraint_penalty: float = 0.0
    adjusted_score: float = 0.0  # predicted_delta - constraint_penalty
    rank: int = 0


# ---------------------------------------------------------------------------
# Phase 3: Evaluation Queue
# ---------------------------------------------------------------------------

@dataclass
class QueueEntry:
    """A candidate in the evaluation queue."""
    queue_id: str = ""
    diff_id: str = ""
    technique_id: str = ""
    paper_id: str = ""
    paper_title: str = ""
    technique_name: str = ""
    surrogate_score: float = 0.0
    constraint_penalty: float = 0.0
    adjusted_score: float = 0.0
    diff_text: str = ""
    hypothesis: str = ""
    priority: float = 0.0
    status: str = "pending"  # "pending", "evaluating", "evaluated", "skipped"
    queued_at: float = field(default_factory=time.time)
    evaluated_at: float = 0.0
    verdict: str = ""
    actual_delta: float = 0.0

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict):
        entry = cls()
        for k, v in data.items():
            if hasattr(entry, k):
                setattr(entry, k, v)
        return entry


# ---------------------------------------------------------------------------
# Phase 3.3 / 4: Quality Tracking
# ---------------------------------------------------------------------------

@dataclass
class PaperSourceQuality:
    """Quality tracking per paper source dimension."""
    dimension: str = ""  # "author", "venue", "category", etc.
    value: str = ""  # the specific author/venue/category
    total_evaluated: int = 0
    total_accepted: int = 0
    success_rate: float = 0.0
    avg_delta: float = 0.0
    last_updated: float = field(default_factory=time.time)

    def to_dict(self):
        return asdict(self)


@dataclass
class ExtractionQualityRecord:
    """Quality tracking for technique extraction accuracy."""
    paper_id: str = ""
    technique_id: str = ""
    surrogate_predicted_delta: float = 0.0
    actual_delta: float = 0.0
    prediction_error: float = 0.0
    extraction_confidence: float = 0.0
    quality_score: float = 0.0  # derived from prediction error
    evaluated_at: float = field(default_factory=time.time)

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict):
        entry = cls()
        for k, v in data.items():
            if hasattr(entry, k):
                setattr(entry, k, v)
        return entry


# ---------------------------------------------------------------------------
# Phase 4: Diagnostics-Driven Ingestion
# ---------------------------------------------------------------------------

@dataclass
class BottleneckMapping:
    """Mapping from a diagnostic bottleneck type to paper search terms."""
    bottleneck_type: str = ""  # e.g., "attention_entropy_collapse"
    search_terms: list = field(default_factory=list)
    severity: float = 0.0  # from diagnostic metric deviation
    boost_weight: float = 0.0

    def to_dict(self):
        return asdict(self)


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def save_jsonl(entries: list, path: str, mode: str = 'a'):
    """Append entries to a JSONL file."""
    with open(path, mode) as f:
        for entry in entries:
            if hasattr(entry, 'to_dict'):
                f.write(json.dumps(entry.to_dict()) + '\n')
            elif isinstance(entry, dict):
                f.write(json.dumps(entry) + '\n')

def load_jsonl(path: str) -> list:
    """Load all entries from a JSONL file."""
    entries = []
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        entries.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
    except FileNotFoundError:
        pass
    return entries
