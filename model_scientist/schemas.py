"""
Shared schemas and data structures for the Model Scientist Pipeline.
All phases reference these canonical types.
"""

from dataclasses import dataclass, field, asdict
from typing import Optional
import json
import time


# ---------------------------------------------------------------------------
# Phase 1.1: Diagnostics Report Schema
# ---------------------------------------------------------------------------

@dataclass
class GradientStats:
    """Per-layer gradient statistics."""
    layer_idx: int
    norm: float
    mean: float
    std: float
    max_abs: float
    dead_fraction: float  # fraction of neurons with zero gradient


@dataclass
class ActivationStats:
    """Per-layer activation statistics."""
    layer_idx: int
    mean: float
    std: float
    max_abs: float
    dead_neuron_count: int
    dead_neuron_fraction: float


@dataclass
class AttentionStats:
    """Per-head attention statistics."""
    layer_idx: int
    head_idx: int
    entropy: float  # attention entropy (higher = more uniform)
    collapse_score: float  # 0=diverse, 1=fully collapsed to uniform
    max_attention_weight: float


@dataclass
class LossBucket:
    """Loss decomposition by token frequency bucket."""
    bucket_name: str  # "top_1k", "1k_10k", "10k_plus", "rare"
    token_count: int
    mean_loss: float
    std_loss: float


@dataclass
class LayerSimilarityEntry:
    """CKA similarity between two layers."""
    layer_i: int
    layer_j: int
    cka_score: float  # 0=dissimilar, 1=identical representations


@dataclass
class HeadCluster:
    """Cluster of attention heads by pattern type."""
    cluster_id: int
    pattern_type: str  # "positional", "syntactic", "rare_token", "mixed"
    head_indices: list  # list of (layer_idx, head_idx) tuples
    centroid_stats: dict


@dataclass
class ProbeResult:
    """Result from a linear probing classifier."""
    task_name: str  # "pos_tagging", "dep_depth", etc.
    layer_idx: int
    accuracy: float
    baseline_accuracy: float  # majority class baseline


@dataclass
class DiagnosticsReport:
    """Complete diagnostics report produced after every training run."""
    timestamp: float = field(default_factory=time.time)
    run_id: str = ""
    step: int = 0
    val_bpb: float = 0.0
    training_seconds: float = 0.0

    # Phase 1.1
    gradient_stats: list = field(default_factory=list)  # list of GradientStats
    activation_stats: list = field(default_factory=list)  # list of ActivationStats
    attention_stats: list = field(default_factory=list)  # list of AttentionStats
    loss_decomposition: list = field(default_factory=list)  # list of LossBucket

    # Phase 1.2
    layer_similarity_matrix: list = field(default_factory=list)  # list of LayerSimilarityEntry
    head_clusters: list = field(default_factory=list)  # list of HeadCluster
    probe_results: list = field(default_factory=list)  # list of ProbeResult

    def to_dict(self):
        return asdict(self)

    def to_json(self, path: str):
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_json(cls, path: str):
        with open(path) as f:
            data = json.load(f)
        report = cls()
        for k, v in data.items():
            if hasattr(report, k):
                setattr(report, k, v)
        return report


# ---------------------------------------------------------------------------
# Phase 1.3: Hypothesis Journal Schema
# ---------------------------------------------------------------------------

@dataclass
class JournalEntry:
    """Single entry in the hypothesis journal."""
    id: str = ""
    timestamp: float = field(default_factory=time.time)
    diagnostics_summary: dict = field(default_factory=dict)
    hypothesis: str = ""
    predicted_delta: float = 0.0  # predicted change in val_bpb (negative=improvement)
    actual_delta: float = 0.0  # actual change in val_bpb
    modification_diff: str = ""  # git diff of the modification
    verdict: str = ""  # "accepted", "rejected", "crashed"
    tags: list = field(default_factory=list)
    # Phase 2 extensions
    failure_pattern_ids: list = field(default_factory=list)
    constraints_applied: list = field(default_factory=list)
    # Phase 2.2-2.3 extensions
    scaling_data: dict = field(default_factory=dict)
    scale_gate_passed: Optional[bool] = None
    # Phase 3 extensions
    ablation_data: dict = field(default_factory=dict)
    components: list = field(default_factory=list)
    stripped_components: list = field(default_factory=list)
    final_val_bpb: Optional[float] = None

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
# Phase 2.1: Failure Pattern Schema
# ---------------------------------------------------------------------------

@dataclass
class FailureFeatures:
    """Structured features extracted from a rejected experiment."""
    journal_id: str = ""
    modification_category: str = ""  # "architecture", "optimizer", "hyperparameter", etc.
    diagnostics_snapshot: dict = field(default_factory=dict)
    predicted_delta: float = 0.0
    actual_delta: float = 0.0
    failure_mode: str = ""  # "regression", "instability", "no_change", "crash"


@dataclass
class FailurePattern:
    """A cluster of similar failures."""
    pattern_id: int = 0
    description: str = ""
    modification_type: str = ""
    instance_count: int = 0
    instance_ids: list = field(default_factory=list)
    centroid_features: dict = field(default_factory=dict)
    avg_actual_delta: float = 0.0


@dataclass
class NegativeConstraint:
    """Natural-language negative constraint derived from failure patterns."""
    constraint_id: int = 0
    pattern_id: int = 0
    text: str = ""  # e.g., "Attention head pruning when gradient norms < 0.01 fails 7/8 times"
    precision: float = 0.0  # fraction of flagged experiments that were actual failures
    recall: float = 0.0  # fraction of failures caught
    is_valid: bool = False  # passes validation criteria


# ---------------------------------------------------------------------------
# Phase 2.2: Scaling Schema
# ---------------------------------------------------------------------------

@dataclass
class ScaleConfig:
    """Model config at a specific scale."""
    scale_factor: float = 1.0  # 0.25, 0.5, 1.0
    depth: int = 0
    n_embd: int = 0
    n_head: int = 0
    n_kv_head: int = 0
    estimated_params: int = 0


@dataclass
class ScalingResult:
    """Result of training at a specific scale."""
    scale_factor: float = 1.0
    config: dict = field(default_factory=dict)
    val_bpb: float = 0.0
    delta_vs_baseline: float = 0.0
    training_seconds: float = 0.0
    converged: bool = True


@dataclass
class ScalingPrediction:
    """Power-law extrapolation from multi-scale results."""
    predicted_delta_1x: float = 0.0
    confidence_interval: tuple = (0.0, 0.0)
    power_law_exponent: float = 0.0
    r_squared: float = 0.0
    scaling_results: list = field(default_factory=list)  # list of ScalingResult


# ---------------------------------------------------------------------------
# Phase 3: Ablation Schema
# ---------------------------------------------------------------------------

@dataclass
class ModificationComponent:
    """A semantically independent component of a modification."""
    component_id: int = 0
    description: str = ""
    diff: str = ""  # isolated patch for this component
    category: str = ""  # "activation", "initialization", "width", etc.


@dataclass
class AblationResult:
    """Result of leave-one-out ablation for one component."""
    component_id: int = 0
    component_description: str = ""
    val_bpb_without: float = 0.0  # val_bpb with this component removed
    marginal_contribution: float = 0.0  # full_improvement - leave_one_out_improvement


@dataclass
class AblationReport:
    """Full ablation report for a modification."""
    modification_id: str = ""
    baseline_val_bpb: float = 0.0
    full_modification_val_bpb: float = 0.0
    full_improvement: float = 0.0
    components: list = field(default_factory=list)  # list of ModificationComponent
    ablation_results: list = field(default_factory=list)  # list of AblationResult
    stripped_components: list = field(default_factory=list)  # component_ids stripped
    final_val_bpb: float = 0.0  # after stripping


# ---------------------------------------------------------------------------
# Phase 4: Metric Evolution Schema
# ---------------------------------------------------------------------------

@dataclass
class MetricDefinition:
    """A diagnostic metric (hardcoded or critic-proposed)."""
    name: str = ""
    description: str = ""
    computation_method: str = ""  # Python code string or function name
    rationale: str = ""
    source: str = ""  # "hardcoded" or "critic"
    created_at: float = field(default_factory=time.time)
    status: str = "candidate"  # "active", "candidate", "retired"
    correlation_with_success: float = 0.0
    consecutive_low_correlation_cycles: int = 0


@dataclass
class MetricCorrelation:
    """Correlation between a metric and experiment outcomes."""
    metric_name: str = ""
    correlation_r: float = 0.0
    p_value: float = 0.0
    n_experiments: int = 0
    cycle: int = 0


# ---------------------------------------------------------------------------
# Utility: JSON serialization helpers
# ---------------------------------------------------------------------------

def save_jsonl(entries: list, path: str, mode: str = 'a'):
    """Append entries to a JSONL file."""
    with open(path, mode) as f:
        for entry in entries:
            if hasattr(entry, 'to_dict'):
                f.write(json.dumps(entry.to_dict()) + '\n')
            else:
                f.write(json.dumps(entry) + '\n')


def load_jsonl(path: str) -> list:
    """Load all entries from a JSONL file."""
    entries = []
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    entries.append(json.loads(line))
    except FileNotFoundError:
        pass
    return entries
