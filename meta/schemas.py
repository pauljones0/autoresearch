"""
Shared schemas for the Meta-Autoresearch pipeline.
"""

from dataclasses import dataclass, field, asdict
from typing import Optional, Any
import json
import time


# ---------------------------------------------------------------------------
# Meta-Parameter Definition
# ---------------------------------------------------------------------------

@dataclass
class MetaParameter:
    """Definition of a single meta-tunable harness parameter."""
    param_id: str = ""
    display_name: str = ""
    system: str = ""  # "bandit", "model_scientist", "surrogate_triage", "gpu_kernels"
    type: str = "float"  # "float", "int", "bool", "str"
    default_value: Any = None
    valid_range: dict = field(default_factory=dict)  # {min, max} or {enum: [...]}
    current_value: Any = None
    code_path: str = ""  # file:class.method
    impact_hypothesis: str = ""
    sensitivity_estimate: str = "unknown"  # "high", "medium", "low", "unknown"
    category: str = ""  # "temperature", "exploration", "evaluation", "scheduling", "prompt", "budget"

    def to_dict(self):
        return asdict(self)


# ---------------------------------------------------------------------------
# Meta-Bandit State
# ---------------------------------------------------------------------------

@dataclass
class DimensionState:
    """Per-dimension state in the meta-bandit."""
    param_id: str = ""
    variants: list = field(default_factory=list)
    variant_posteriors: dict = field(default_factory=dict)  # variant_str -> {alpha, beta}
    current_best: Any = None
    last_promoted: float = 0.0

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "DimensionState":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class MetaBanditState:
    """Complete meta-bandit state."""
    dimensions: dict = field(default_factory=dict)  # param_id -> DimensionState
    global_meta_iteration: int = 0
    meta_regime: str = "baseline"  # "baseline", "active", "maintenance"
    total_meta_experiments: int = 0
    budget_used: float = 0.0
    budget_fraction: float = 0.2
    budget_cycle_length: int = 500
    enable_auto_budget: bool = False
    current_config: dict = field(default_factory=dict)
    best_config: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=lambda: {
        "created_at": 0.0,
        "last_updated": 0.0,
        "schema_version": "1.0",
    })

    def to_dict(self):
        d = {}
        d["dimensions"] = {k: v.to_dict() if hasattr(v, "to_dict") else v
                           for k, v in self.dimensions.items()}
        for key in ["global_meta_iteration", "meta_regime", "total_meta_experiments",
                     "budget_used", "budget_fraction", "budget_cycle_length",
                     "enable_auto_budget", "current_config", "best_config", "metadata"]:
            d[key] = getattr(self, key)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "MetaBanditState":
        state = cls()
        dims = d.get("dimensions", {})
        state.dimensions = {k: DimensionState.from_dict(v) if isinstance(v, dict) else v
                            for k, v in dims.items()}
        for key in ["global_meta_iteration", "meta_regime", "total_meta_experiments",
                     "budget_used", "budget_fraction", "budget_cycle_length",
                     "enable_auto_budget", "current_config", "best_config", "metadata"]:
            if key in d:
                setattr(state, key, d[key])
        return state

    def validate(self) -> list:
        issues = []
        if self.meta_regime not in ("baseline", "active", "maintenance"):
            issues.append(f"Invalid meta_regime: {self.meta_regime}")
        if self.budget_fraction < 0 or self.budget_fraction > 1:
            issues.append(f"budget_fraction out of range: {self.budget_fraction}")
        if self.global_meta_iteration < 0:
            issues.append(f"global_meta_iteration < 0")
        for dim_id, dim in self.dimensions.items():
            if not isinstance(dim, DimensionState):
                issues.append(f"Dimension {dim_id} is not a DimensionState")
                continue
            for var_key, post in dim.variant_posteriors.items():
                if isinstance(post, dict):
                    if post.get("alpha", 1) < 1:
                        issues.append(f"Dim {dim_id} variant {var_key}: alpha < 1")
                    if post.get("beta", 1) < 1:
                        issues.append(f"Dim {dim_id} variant {var_key}: beta < 1")
        return issues


# ---------------------------------------------------------------------------
# Meta-Experiment
# ---------------------------------------------------------------------------

@dataclass
class ParamDiff:
    """A single parameter change."""
    param_id: str = ""
    old_value: Any = None
    new_value: Any = None

    def to_dict(self):
        return asdict(self)


@dataclass
class MetaExperimentResult:
    """Result of a meta-experiment."""
    experiment_id: str = ""
    config_diff: list = field(default_factory=list)
    n_iterations: int = 0
    improvement_rate: float = 0.0
    acceptance_rate: float = 0.0
    stepping_stone_rate: float = 0.0
    entropy: float = 0.0
    compared_to_baseline: str = "inconclusive"  # "better", "worse", "inconclusive"
    baseline_ir_used: float = 0.0
    raw_deltas: list = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)

    def to_dict(self):
        return asdict(self)


@dataclass
class MetaContext:
    """Context for meta-experiment execution."""
    bandit_pipeline: object = None
    model_scientist_pipeline: object = None
    surrogate_triage_pipeline: object = None
    gpu_kernel_pipeline: object = None
    work_dir: str = "."
    campaign_profile: dict = field(default_factory=dict)

    def to_dict(self):
        return {"work_dir": self.work_dir, "campaign_profile": self.campaign_profile}


# ---------------------------------------------------------------------------
# Baseline Measurement
# ---------------------------------------------------------------------------

@dataclass
class BaselineResult:
    """Result of a single baseline run."""
    seed: int = 0
    n_iterations: int = 0
    deltas: list = field(default_factory=list)
    verdicts: list = field(default_factory=list)
    improvement_rates: list = field(default_factory=list)
    mean_ir: float = 0.0
    std_ir: float = 0.0
    total_improvement: float = 0.0
    acceptance_rate: float = 0.0

    def to_dict(self):
        return asdict(self)


@dataclass
class AggregateIR:
    """Cross-run improvement rate statistics."""
    mean_ir: float = 0.0
    std_ir: float = 0.0
    ci_95_lower: float = 0.0
    ci_95_upper: float = 0.0
    median_ir: float = 0.0
    iqr: float = 0.0
    within_run_variance: float = 0.0
    between_run_variance: float = 0.0
    n_windows: int = 0
    n_runs: int = 0

    def to_dict(self):
        return asdict(self)


@dataclass
class MDESResult:
    """Minimum detectable effect size."""
    mdes_absolute: float = 0.0
    mdes_relative: float = 0.0
    meta_experiment_length: int = 0
    alpha: float = 0.05
    power: float = 0.8
    n_windows_per_experiment: int = 0

    def to_dict(self):
        return asdict(self)


@dataclass
class ExperimentLengthResult:
    """Optimal meta-experiment length."""
    optimal_K: int = 50
    n_experiments_per_cycle: int = 0
    mdes_at_optimal_K: float = 0.0
    total_meta_iterations_per_cycle: int = 0

    def to_dict(self):
        return asdict(self)


# ---------------------------------------------------------------------------
# Safety
# ---------------------------------------------------------------------------

class BoundaryViolationError(Exception):
    """Raised when the meta-loop attempts a forbidden action."""
    def __init__(self, violation_type: str, detail: str = ""):
        self.violation_type = violation_type
        self.detail = detail
        super().__init__(f"BoundaryViolation({violation_type}): {detail}")


@dataclass
class TestResult:
    """Result of a boundary violation test."""
    test_name: str = ""
    passed: bool = False
    error_type: str = ""
    detail: str = ""

    def to_dict(self):
        return asdict(self)


# ---------------------------------------------------------------------------
# STOP Strategies
# ---------------------------------------------------------------------------

@dataclass
class GeneratedStrategy:
    """A generated harness strategy."""
    strategy_id: str = ""
    hook_type: str = ""  # "selection_hook", "acceptance_hook", "prompt_hook", "scheduling_hook"
    code: str = ""
    description: str = ""
    llm_rationale: str = ""
    estimated_improvement: str = ""

    def to_dict(self):
        return asdict(self)


@dataclass
class SafetyCheckResult:
    """Result of strategy safety check."""
    safe: bool = False
    violations: list = field(default_factory=list)
    ast_nodes_checked: int = 0

    def to_dict(self):
        return asdict(self)


# ---------------------------------------------------------------------------
# Prompt Variants
# ---------------------------------------------------------------------------

@dataclass
class PromptVariant:
    """A prompt template variant."""
    variant_id: str = ""
    arm_id: str = ""
    template_text: str = ""
    variation_dimension: str = ""
    variation_description: str = ""
    parent_template_hash: str = ""

    def to_dict(self):
        return asdict(self)


@dataclass
class PromptEvalResult:
    """Result of prompt variant evaluation."""
    variant_id: str = ""
    arm_id: str = ""
    per_seed_ir: list = field(default_factory=list)
    mean_ir: float = 0.0
    std_ir: float = 0.0
    n_arm_selections: int = 0
    n_arm_successes: int = 0
    arm_success_rate: float = 0.0

    def to_dict(self):
        return asdict(self)


# ---------------------------------------------------------------------------
# Context & Evaluation Protocol
# ---------------------------------------------------------------------------

@dataclass
class ContextAllocation:
    """Context token allocation strategy."""
    allocation_id: str = ""
    code_fraction: float = 0.3
    history_fraction: float = 0.25
    diagnostics_fraction: float = 0.25
    constraints_fraction: float = 0.2
    is_dynamic: bool = False
    dynamic_rule: Optional[str] = None

    def to_dict(self):
        return asdict(self)


@dataclass
class EvalProtocol:
    """Evaluation protocol variant."""
    protocol_id: str = ""
    training_steps: int = 1000
    n_seeds: int = 1
    warmup_fraction: float = 0.2
    is_two_stage: bool = False
    stage1_steps: int = 500
    stage1_seeds: int = 1
    stage2_steps: int = 1500
    stage2_seeds: int = 2

    def to_dict(self):
        return asdict(self)


# ---------------------------------------------------------------------------
# Interactions
# ---------------------------------------------------------------------------

@dataclass
class Interaction:
    """Detected parameter interaction."""
    dim_i: str = ""
    dim_j: str = ""
    interaction_effect: float = 0.0
    p_value: float = 1.0
    synergy: bool = False
    antagonism: bool = False

    def to_dict(self):
        return asdict(self)


# ---------------------------------------------------------------------------
# Budget & ROI
# ---------------------------------------------------------------------------

@dataclass
class BudgetRecommendation:
    """Meta-budget recommendation."""
    current_fraction: float = 0.2
    recommended_fraction: float = 0.2
    reason: str = ""
    confidence: str = "medium"

    def to_dict(self):
        return asdict(self)


@dataclass
class ROIData:
    """Return on meta-investment."""
    total_meta_iterations: int = 0
    total_production_iterations: int = 0
    improvement_from_meta: float = 0.0
    cost_of_meta: float = 0.0
    roi: float = 0.0
    cumulative_val_bpb_improvement: float = 0.0
    attribution: dict = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)


# ---------------------------------------------------------------------------
# Convergence
# ---------------------------------------------------------------------------

@dataclass
class ConvergenceStatus:
    """Meta-convergence status."""
    converged: bool = False
    meta_experiments_since_last_promotion: int = 0
    max_posterior_variance: float = 0.0
    f_test_p_value: float = 1.0
    recommendation: str = "continue"  # "continue", "enter_maintenance", "already_in_maintenance"

    def to_dict(self):
        return asdict(self)


@dataclass
class DivergenceAlert:
    """Alert when production IR drops significantly."""
    triggered: bool = False
    current_ir: float = 0.0
    baseline_ir: float = 0.0
    drop_magnitude: float = 0.0
    windows_below_threshold: int = 0
    recommendation: str = ""

    def to_dict(self):
        return asdict(self)


# ---------------------------------------------------------------------------
# Documentation & Knowledge
# ---------------------------------------------------------------------------

@dataclass
class ConfigDocumentation:
    """Per-dimension evidence-based documentation."""
    dimensions: dict = field(default_factory=dict)
    total_experiments: int = 0
    total_promotions: int = 0
    best_config_ir: float = 0.0
    default_config_ir: float = 0.0
    improvement_over_defaults: float = 0.0

    def to_dict(self):
        return asdict(self)


@dataclass
class SensitivityReport:
    """Sensitivity analysis report."""
    per_dimension: dict = field(default_factory=dict)
    critical_dimensions: list = field(default_factory=list)
    robust_dimensions: list = field(default_factory=list)

    def to_dict(self):
        return asdict(self)


@dataclass
class MetaInsight:
    """A transferable meta-optimization insight."""
    insight_id: str = ""
    type: str = ""  # "universal", "phase_dependent", "scale_dependent", "interaction"
    description: str = ""
    evidence_experiments: list = field(default_factory=list)
    confidence: str = "low"  # "low", "medium", "high"
    transferability: str = "unknown"  # "universal", "conditional", "campaign_specific"
    recommended_default: Any = None

    def to_dict(self):
        return asdict(self)


@dataclass
class TransferValidationResult:
    """Result of validating insight transfer."""
    insight_id: str = ""
    validated: bool = False
    validation_ir: float = 0.0
    default_ir: float = 0.0
    improvement: float = 0.0
    p_value: float = 1.0

    def to_dict(self):
        return asdict(self)


# ---------------------------------------------------------------------------
# Validation & Monitoring
# ---------------------------------------------------------------------------

@dataclass
class ExtendedMetaValidationResult:
    """Extended validation result."""
    passed: bool = False
    n_iterations: int = 0
    boundary_violations: int = 0
    state_corruptions: int = 0
    eval_hash_changes: int = 0
    budget_deviation_percent: float = 0.0
    health_audit_failures: int = 0
    meta_experiments_run: int = 0
    promotions: int = 0

    def to_dict(self):
        return asdict(self)


@dataclass
class ComparisonResult:
    """Defaults vs meta-optimized comparison."""
    treatment_median_improvement: float = 0.0
    control_median_improvement: float = 0.0
    u_statistic: float = 0.0
    p_value: float = 1.0
    significant: bool = False
    per_dimension_contributions: dict = field(default_factory=dict)
    effect_size: float = 0.0
    verdict: str = ""

    def to_dict(self):
        return asdict(self)


@dataclass
class StabilityReport:
    """Long-term stability monitoring report."""
    ir_stable: bool = False
    mean_ir: float = 0.0
    std_ir: float = 0.0
    divergence_triggers: int = 0
    false_positives: int = 0
    maintenance_experiments: int = 0
    maintenance_discoveries: int = 0

    def to_dict(self):
        return asdict(self)


@dataclass
class VarianceCostReport:
    """Variance-cost analysis of evaluation protocols."""
    per_protocol: dict = field(default_factory=dict)
    recommended_protocol_id: str = ""
    recommended_two_stage: bool = False

    def to_dict(self):
        return asdict(self)


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def save_json(data, path: str):
    """Save data as JSON."""
    with open(path, 'w') as f:
        if hasattr(data, 'to_dict'):
            json.dump(data.to_dict(), f, indent=2)
        else:
            json.dump(data, f, indent=2)


def load_json(path: str) -> dict:
    """Load JSON from file."""
    try:
        with open(path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def save_jsonl(entry, path: str):
    """Append a single entry to a JSONL file."""
    with open(path, 'a') as f:
        data = entry.to_dict() if hasattr(entry, 'to_dict') else entry
        f.write(json.dumps(data) + '\n')


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
