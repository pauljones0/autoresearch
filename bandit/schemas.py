"""
Shared schemas for the Adaptive Bandit with Simulated Annealing pipeline.
"""

from dataclasses import dataclass, field, asdict
from typing import Optional
import json
import time
import math


# ---------------------------------------------------------------------------
# Arm Definition
# ---------------------------------------------------------------------------

@dataclass
class ArmDefinition:
    """Definition of a single bandit arm."""
    arm_id: str = ""
    display_name: str = ""
    source_type: str = ""  # "internal", "paper", "kernel"
    dispatch_target: str = ""
    queue_filter: Optional[dict] = None
    prompt_template_key: Optional[str] = None
    can_have_paper_variant: bool = False

    def to_dict(self):
        return asdict(self)


# ---------------------------------------------------------------------------
# Arm State (per-arm in strategy_state.json)
# ---------------------------------------------------------------------------

@dataclass
class ArmState:
    """Per-arm state within the bandit."""
    alpha: float = 1.0
    beta: float = 1.0
    temperature: float = 0.02
    consecutive_failures: int = 0
    last_selected: float = 0.0
    total_attempts: int = 0
    total_successes: int = 0
    diagnostics_boost: float = 0.0
    constraint_density: float = 0.0
    source_type: str = "internal"
    last_reheat: float = 0.0
    reheat_count: int = 0

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ArmState":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ---------------------------------------------------------------------------
# Bandit State (global strategy_state.json)
# ---------------------------------------------------------------------------

@dataclass
class BanditState:
    """Complete bandit state persisted to strategy_state.json."""
    arms: dict = field(default_factory=dict)  # arm_id -> ArmState dict
    global_iteration: int = 0
    T_base: float = 0.025
    reheat_factor: float = 3.0
    K_reheat_threshold: int = 5
    min_temperature: float = 0.001
    exploration_floor: float = 0.05
    regime: str = "no_bandit"  # "no_bandit", "conservative_bandit", "full_bandit"
    last_regime_change: float = 0.0
    paper_preference_ratio: float = 0.4
    delayed_corrections: list = field(default_factory=list)
    enable_rollback_safety: bool = True
    enable_auto_tuning: bool = False
    metadata: dict = field(default_factory=lambda: {
        "created_at": 0.0,
        "last_updated": 0.0,
        "schema_version": "1.0",
        "warm_start_source": "",
        "warm_start_entries": 0,
    })

    def to_dict(self):
        d = {}
        d["arms"] = {k: (v.to_dict() if hasattr(v, "to_dict") else v)
                      for k, v in self.arms.items()}
        d["global_iteration"] = self.global_iteration
        d["T_base"] = self.T_base
        d["reheat_factor"] = self.reheat_factor
        d["K_reheat_threshold"] = self.K_reheat_threshold
        d["min_temperature"] = self.min_temperature
        d["exploration_floor"] = self.exploration_floor
        d["regime"] = self.regime
        d["last_regime_change"] = self.last_regime_change
        d["paper_preference_ratio"] = self.paper_preference_ratio
        d["delayed_corrections"] = self.delayed_corrections
        d["enable_rollback_safety"] = self.enable_rollback_safety
        d["enable_auto_tuning"] = self.enable_auto_tuning
        d["metadata"] = self.metadata
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "BanditState":
        state = cls()
        arms_raw = d.get("arms", {})
        state.arms = {k: ArmState.from_dict(v) if isinstance(v, dict) else v
                      for k, v in arms_raw.items()}
        for key in ["global_iteration", "T_base", "reheat_factor",
                     "K_reheat_threshold", "min_temperature", "exploration_floor",
                     "regime", "last_regime_change", "paper_preference_ratio",
                     "delayed_corrections", "enable_rollback_safety",
                     "enable_auto_tuning", "metadata"]:
            if key in d:
                setattr(state, key, d[key])
        return state

    def validate(self) -> list:
        """Validate state integrity. Returns list of issues (empty = valid)."""
        issues = []
        for arm_id, arm in self.arms.items():
            if not isinstance(arm, ArmState):
                issues.append(f"Arm {arm_id} is not an ArmState instance")
                continue
            if arm.alpha < 1:
                issues.append(f"Arm {arm_id}: alpha={arm.alpha} < 1")
            if arm.beta < 1:
                issues.append(f"Arm {arm_id}: beta={arm.beta} < 1")
            if arm.temperature < 0:
                issues.append(f"Arm {arm_id}: temperature={arm.temperature} < 0")
        if self.global_iteration < 0:
            issues.append(f"global_iteration={self.global_iteration} < 0")
        return issues


# ---------------------------------------------------------------------------
# Selection Result
# ---------------------------------------------------------------------------

@dataclass
class SelectionResult:
    """Result of Thompson Sampling arm selection."""
    arm_id: str = ""
    sample_values: dict = field(default_factory=dict)
    selected_by: str = "thompson"  # "thompson", "exploration_floor", "fallback"
    retry_count: int = 0
    dispatch_path: str = "internal"  # "internal", "paper", "kernel"
    queue_entry_id: Optional[str] = None

    def to_dict(self):
        return asdict(self)


# ---------------------------------------------------------------------------
# Acceptance Decision
# ---------------------------------------------------------------------------

@dataclass
class AcceptanceDecision:
    """Result of annealing acceptance decision."""
    accepted: bool = False
    accepted_by: str = "rejected"  # "improvement", "annealing", "rejected"
    probability: float = 0.0
    random_draw: Optional[float] = None
    T_effective: float = 0.0
    delta: float = 0.0
    surrogate_predicted_delta: Optional[float] = None
    surrogate_modulation_factor: float = 1.0

    def to_dict(self):
        return asdict(self)


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

@dataclass
class DispatchContext:
    """Context for arm dispatch containing pipeline references."""
    model_scientist_pipeline: object = None
    surrogate_triage_pipeline: object = None
    gpu_kernel_pipeline: object = None
    queue_manager: object = None
    base_source: str = ""
    diagnostics_report: object = None

    def to_dict(self):
        return {"base_source": self.base_source}


@dataclass
class DispatchResult:
    """Result of dispatching an arm's action."""
    arm_id: str = ""
    dispatch_path: str = ""
    success: bool = False
    delta: Optional[float] = None
    verdict: str = ""
    journal_entry_id: str = ""
    elapsed_seconds: float = 0.0
    error: Optional[str] = None

    def to_dict(self):
        return asdict(self)


# ---------------------------------------------------------------------------
# Reheat
# ---------------------------------------------------------------------------

@dataclass
class ReheatEvent:
    """Record of a temperature reheat event."""
    arm_id: str = ""
    temperature_before: float = 0.0
    temperature_after: float = 0.0
    consecutive_failures_at_trigger: int = 0
    reheat_count: int = 0
    budget_remaining: int = 0

    def to_dict(self):
        return asdict(self)


# ---------------------------------------------------------------------------
# Rollback
# ---------------------------------------------------------------------------

@dataclass
class RollbackResult:
    """Result of a rollback safety net check."""
    rolled_back: bool = False
    arm_id: str = ""
    annealing_entry_id: str = ""
    subsequent_entries: list = field(default_factory=list)
    reverted_to_source_hash: str = ""

    def to_dict(self):
        return asdict(self)


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@dataclass
class HealthAlert:
    """Alert from posterior health checker."""
    severity: str = "info"  # "info", "warning", "critical"
    arm_id: Optional[str] = None
    message: str = ""
    recommendation: str = ""

    def to_dict(self):
        return asdict(self)


# ---------------------------------------------------------------------------
# Regime Change
# ---------------------------------------------------------------------------

@dataclass
class RegimeChangeEvent:
    """Detected regime change for an arm."""
    arm_id: str = ""
    rate_all: float = 0.0
    rate_rolling: float = 0.0
    rate_drop_magnitude: float = 0.0
    diagnostics_snapshot_summary: str = ""
    recommended_actions: list = field(default_factory=list)

    def to_dict(self):
        return asdict(self)


# ---------------------------------------------------------------------------
# Validation Reports
# ---------------------------------------------------------------------------

@dataclass
class ValidationReport:
    """Arm taxonomy validation report."""
    valid: bool = False
    arm_count: int = 0
    mapped_entries: int = 0
    unmapped_entries: int = 0
    orphan_categories: list = field(default_factory=list)
    duplicate_arms: list = field(default_factory=list)
    issues: list = field(default_factory=list)

    def to_dict(self):
        return asdict(self)


@dataclass
class WarmStartValidationReport:
    """Warm-start validation report."""
    valid: bool = False
    per_arm_checks: dict = field(default_factory=dict)
    evidence_conservation: dict = field(default_factory=dict)
    temperature_ordering_correct: bool = False
    issues: list = field(default_factory=list)

    def to_dict(self):
        return asdict(self)


@dataclass
class SelectionValidationReport:
    """Selection distribution validation report."""
    chi_squared_statistic: float = 0.0
    p_value: float = 0.0
    passed: bool = False
    per_arm_frequencies: dict = field(default_factory=dict)
    exploration_floor_violations: list = field(default_factory=list)

    def to_dict(self):
        return asdict(self)


@dataclass
class ConsistencyReport:
    """Cross-system consistency report."""
    consistent: bool = False
    journal_log_mismatches: int = 0
    posterior_arithmetic_errors: int = 0
    stale_queue_entries: int = 0
    orphan_kernel_configs: int = 0
    issues: list = field(default_factory=list)

    def to_dict(self):
        return asdict(self)


@dataclass
class ReheatDecayReport:
    """Reheat decay verification report."""
    per_arm: dict = field(default_factory=dict)
    all_decaying: bool = True

    def to_dict(self):
        return asdict(self)


@dataclass
class ReplayValidationReport:
    """Log replay validation report."""
    replay_matches: bool = False
    per_arm_discrepancies: dict = field(default_factory=dict)
    max_alpha_error: float = 0.0
    max_beta_error: float = 0.0
    max_temperature_error: float = 0.0
    log_entries_replayed: int = 0

    def to_dict(self):
        return asdict(self)


@dataclass
class BanditAuditReport:
    """Comprehensive bandit health audit report."""
    all_clear: bool = False
    checks_run: int = 0
    checks_passed: int = 0
    issues: list = field(default_factory=list)

    def to_dict(self):
        return asdict(self)


# ---------------------------------------------------------------------------
# A/B Test
# ---------------------------------------------------------------------------

@dataclass
class ABAnalysisReport:
    """A/B test analysis report."""
    n_seeds: int = 0
    treatment_median_improvement: float = 0.0
    control_median_improvement: float = 0.0
    u_statistic: float = 0.0
    p_value: float = 0.0
    significant: bool = False
    effect_size: float = 0.0
    per_arm_contributions: dict = field(default_factory=dict)
    annealing_stepping_stones: int = 0
    waste_rate_treatment: float = 0.0
    waste_rate_control: float = 0.0
    verdict: str = ""

    def to_dict(self):
        return asdict(self)


@dataclass
class AcceptanceAnalysisReport:
    """Acceptance analysis report."""
    total_decisions: int = 0
    acceptance_rate: float = 0.0
    greedy_rate: float = 0.0
    annealing_rate: float = 0.0
    per_arm_annealing_rates: dict = field(default_factory=dict)
    stepping_stone_rate: float = 0.0
    dead_end_rate: float = 0.0
    temperature_trajectory: dict = field(default_factory=dict)
    surrogate_modulation_impact: dict = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)


@dataclass
class AllocationComparisonReport:
    """Allocation efficiency comparison report."""
    treatment_allocation: dict = field(default_factory=dict)
    control_allocation: dict = field(default_factory=dict)
    per_arm_efficiency: dict = field(default_factory=dict)
    allocation_gaps: list = field(default_factory=list)
    allocation_mistakes: list = field(default_factory=list)

    def to_dict(self):
        return asdict(self)


# ---------------------------------------------------------------------------
# Tuning
# ---------------------------------------------------------------------------

@dataclass
class TuningRecommendation:
    """Auto-tuner recommendation."""
    parameter: str = ""
    current_value: float = 0.0
    recommended_value: float = 0.0
    reason: str = ""
    confidence: str = "medium"  # "low", "medium", "high"
    auto_applicable: bool = False

    def to_dict(self):
        return asdict(self)


# ---------------------------------------------------------------------------
# Fallback
# ---------------------------------------------------------------------------

@dataclass
class FallbackDecision:
    """Graceful degradation fallback decision."""
    action: str = ""  # "recover_state", "recover_log", "disable_arms", "fallback_to_scheduler"
    detail: str = ""

    def to_dict(self):
        return asdict(self)


# ---------------------------------------------------------------------------
# Iteration Result
# ---------------------------------------------------------------------------

@dataclass
class IterationResult:
    """Result of a single bandit loop iteration."""
    iteration: int = 0
    arm_selected: str = ""
    dispatch_path: str = ""
    delta: Optional[float] = None
    verdict: str = ""
    accepted: bool = False
    accepted_by: str = ""
    temperature: float = 0.0
    reheat_triggered: bool = False
    rollback_triggered: bool = False
    health_alerts: list = field(default_factory=list)
    elapsed_seconds: float = 0.0

    def to_dict(self):
        return asdict(self)


# ---------------------------------------------------------------------------
# Loop Context
# ---------------------------------------------------------------------------

@dataclass
class LoopContext:
    """Context for a bandit loop iteration."""
    model_scientist_pipeline: object = None
    surrogate_triage_pipeline: object = None
    gpu_kernel_pipeline: object = None
    queue_manager: object = None
    journal_reader: object = None
    journal_writer: object = None
    bandit_state: object = None
    log_writer: object = None
    rng: object = None
    diagnostics_report: object = None
    base_source: str = ""

    def to_dict(self):
        return {"base_source": self.base_source}


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
