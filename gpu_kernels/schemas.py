"""
Shared schemas for the GPU Kernel Creation Pipeline.
"""

from dataclasses import dataclass, field, asdict
from typing import Optional
import json
import time


# ---------------------------------------------------------------------------
# Phase 1.1: Profiling
# ---------------------------------------------------------------------------

@dataclass
class HardwareProfile:
    """Auto-detected GPU hardware capabilities."""
    gpu_name: str = ""
    compute_capability: tuple = (0, 0)
    peak_bandwidth_gbps: float = 0.0
    sm_count: int = 0
    max_shared_memory_per_block_bytes: int = 0
    max_threads_per_block: int = 0
    warp_size: int = 32
    max_registers_per_block: int = 0
    clock_rate_mhz: float = 0.0
    total_memory_bytes: int = 0

    def to_dict(self):
        return asdict(self)


@dataclass
class OperationProfile:
    """Per-operation GPU profiling data."""
    op_name: str = ""
    gpu_time_us: float = 0.0
    cpu_time_us: float = 0.0
    memory_read_bytes: int = 0
    memory_write_bytes: int = 0
    bandwidth_utilization: float = 0.0  # fraction of theoretical peak
    sm_occupancy: float = 0.0  # fraction
    is_fuseable: bool = False
    fuseable_group_id: str = ""
    call_stack: str = ""
    input_shapes: list = field(default_factory=list)
    output_shapes: list = field(default_factory=list)

    def to_dict(self):
        return asdict(self)


@dataclass
class FuseableGroup:
    """Group of operations that can be fused into a single Triton kernel."""
    group_id: str = ""
    op_names: list = field(default_factory=list)
    combined_gpu_time_us: float = 0.0
    estimated_fused_time_us: float = 0.0
    estimated_speedup_ratio: float = 0.0
    fusion_type: str = ""  # "elementwise", "reduction", "attention", "optimizer", "normalization"
    tensor_shapes: dict = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)


# ---------------------------------------------------------------------------
# Phase 1.2: Reference Catalog
# ---------------------------------------------------------------------------

@dataclass
class ReferenceImplementation:
    """Reference PyTorch implementation for a fuseable group."""
    group_id: str = ""
    reference_path: str = ""
    shapes_path: str = ""
    tolerances_path: str = ""
    test_inputs_dir: str = ""
    op_sequence: list = field(default_factory=list)
    source_lines: str = ""  # line range in train.py

    def to_dict(self):
        return asdict(self)


@dataclass
class ToleranceBounds:
    """Per-dtype correctness tolerance bounds."""
    fp32: dict = field(default_factory=lambda: {"atol": 1e-5, "rtol": 1e-5})
    fp16: dict = field(default_factory=lambda: {"atol": 1e-2, "rtol": 1e-2})
    bf16: dict = field(default_factory=lambda: {"atol": 1e-2, "rtol": 1e-2})

    def to_dict(self):
        return asdict(self)


# ---------------------------------------------------------------------------
# Phase 1.3: Verification
# ---------------------------------------------------------------------------

@dataclass
class CorrectnessResult:
    """Result of kernel correctness verification."""
    passed: bool = False
    tested_configs: int = 0
    passed_configs: int = 0
    failed_configs: list = field(default_factory=list)
    total_time_seconds: float = 0.0

    def to_dict(self):
        return asdict(self)


@dataclass
class StabilityResult:
    """Result of numerical stability probing."""
    is_deterministic: bool = True
    max_cv_across_runs: float = 0.0
    overflow_detected: bool = False
    underflow_detected: bool = False
    denormal_handling: str = "correct"
    extreme_input_results: list = field(default_factory=list)

    def to_dict(self):
        return asdict(self)


@dataclass
class DivergenceResult:
    """Result of training divergence detection."""
    passed: bool = False
    max_loss_divergence: float = 0.0
    max_grad_norm_divergence: float = 0.0
    final_param_match: bool = False
    diverged_at_step: Optional[int] = None
    loss_curve_reference: list = field(default_factory=list)
    loss_curve_kernel: list = field(default_factory=list)

    def to_dict(self):
        return asdict(self)


@dataclass
class KernelVerificationReport:
    """Aggregated verification report for a kernel."""
    kernel_id: str = ""
    group_id: str = ""
    correctness: dict = field(default_factory=dict)
    stability: dict = field(default_factory=dict)
    divergence: dict = field(default_factory=dict)
    verdict: str = ""  # "PASS", "FAIL", "CONDITIONAL_PASS"
    timestamp: float = field(default_factory=time.time)
    notes: list = field(default_factory=list)

    def to_dict(self):
        return asdict(self)

    def to_json(self, path: str):
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)


# ---------------------------------------------------------------------------
# Phase 1.4: Benchmarking
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkConfigResult:
    """Benchmark result for a single input configuration."""
    config_name: str = ""
    reference_median_us: float = 0.0
    kernel_median_us: float = 0.0
    speedup_ratio: float = 0.0
    cv_reference: float = 0.0
    cv_kernel: float = 0.0

    def to_dict(self):
        return asdict(self)


@dataclass
class BenchmarkResult:
    """Complete benchmark result for a kernel."""
    kernel_id: str = ""
    reference_group_id: str = ""
    input_configs_tested: int = 0
    results: list = field(default_factory=list)  # list of BenchmarkConfigResult
    overall_speedup: float = 0.0  # geometric mean
    peak_memory_kernel_bytes: int = 0
    peak_memory_reference_bytes: int = 0
    memory_savings_ratio: float = 0.0

    def to_dict(self):
        return asdict(self)


@dataclass
class BandwidthProfile:
    """Memory bandwidth profiling for a kernel."""
    bytes_read: int = 0
    bytes_written: int = 0
    execution_time_us: float = 0.0
    achieved_bandwidth_gbps: float = 0.0
    bandwidth_efficiency: float = 0.0
    is_memory_bound: bool = False
    is_compute_bound: bool = False

    def to_dict(self):
        return asdict(self)


@dataclass
class ThroughputEstimate:
    """Estimated end-to-end training throughput improvement."""
    kernel_id: str = ""
    target_operation: str = ""
    operation_fraction_of_step: float = 0.0
    estimated_step_time_reduction_us: float = 0.0
    estimated_tok_per_sec_delta: float = 0.0
    estimated_training_time_savings_percent: float = 0.0
    confidence: str = "medium"

    def to_dict(self):
        return asdict(self)


# ---------------------------------------------------------------------------
# Phase 2: Kernel Generation
# ---------------------------------------------------------------------------

@dataclass
class KernelTarget:
    """A selected target for kernel generation."""
    group_id: str = ""
    op_sequence: list = field(default_factory=list)
    shapes: dict = field(default_factory=dict)
    reference_path: str = ""
    estimated_speedup: float = 0.0
    fusion_type: str = ""

    def to_dict(self):
        return asdict(self)


@dataclass
class GeneratedKernel:
    """A generated Triton kernel variant."""
    kernel_id: str = ""
    group_id: str = ""
    variant_index: int = 0
    kernel_path: str = ""
    integration_diff: str = ""
    block_size: int = 256
    num_warps: int = 4
    memory_strategy: str = "row_major"
    fusion_type: str = ""

    def to_dict(self):
        return asdict(self)


@dataclass
class KernelConfigEntry:
    """Entry in kernel_config.json."""
    group_id: str = ""
    backend: str = "pytorch"  # "triton" or "pytorch"
    kernel_path: str = ""
    fallback: str = "pytorch"
    verified_at: float = 0.0
    speedup: float = 0.0
    verification_report: str = ""
    enabled: bool = True

    def to_dict(self):
        return asdict(self)


# ---------------------------------------------------------------------------
# Phase 4: Evolution
# ---------------------------------------------------------------------------

@dataclass
class MutatedKernel:
    """A mutated kernel variant."""
    mutation_id: str = ""
    parent_id: str = ""
    mutation_type: str = ""
    mutation_description: str = ""
    kernel_source: str = ""
    kernel_path: str = ""

    def to_dict(self):
        return asdict(self)


@dataclass
class GenerationResult:
    """Result of one evolutionary generation."""
    generation: int = 0
    parent_id: str = ""
    mutations_tested: int = 0
    mutations_passed: int = 0
    best_mutation_id: str = ""
    best_speedup: float = 0.0
    improvement_over_parent: float = 0.0
    crossover_attempted: bool = False
    crossover_result: dict = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)


@dataclass
class KernelOpportunity:
    """A ranked kernel optimization opportunity."""
    group_id: str = ""
    op_names: list = field(default_factory=list)
    estimated_impact_us: float = 0.0
    current_bandwidth_utilization: float = 0.0
    fusion_type: str = ""
    constraint_penalty: float = 0.0
    adjusted_score: float = 0.0
    already_attempted_count: int = 0

    def to_dict(self):
        return asdict(self)


@dataclass
class DiscoveryCycleResult:
    """Result of an autonomous kernel discovery cycle."""
    target_group_id: str = ""
    variants_generated: int = 0
    variants_passed_prescreen: int = 0
    variants_passed_full_verification: int = 0
    winner_kernel_id: str = ""
    winner_speedup: float = 0.0
    queued: bool = False

    def to_dict(self):
        return asdict(self)


# ---------------------------------------------------------------------------
# Phase 5: Monitoring
# ---------------------------------------------------------------------------

@dataclass
class RuntimeAlert:
    """Alert from runtime correctness monitoring."""
    kernel_id: str = ""
    step: int = 0
    max_abs_error: float = 0.0
    tolerance_threshold: float = 0.0
    severity: str = "warning"  # "warning" or "critical"
    timestamp: float = field(default_factory=time.time)

    def to_dict(self):
        return asdict(self)


@dataclass
class ExtendedValidationResult:
    """Result of extended training validation."""
    passed: bool = False
    n_steps: int = 0
    max_loss_divergence: float = 0.0
    max_grad_norm_divergence: float = 0.0
    max_param_divergence: float = 0.0
    failing_step: Optional[int] = None
    failing_metric: str = ""
    culprit_kernel_id: str = ""

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
