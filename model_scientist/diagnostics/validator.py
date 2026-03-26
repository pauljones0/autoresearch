"""
Validator for DiagnosticsReport: checks schema completeness, value ranges,
and flags anomalies.

Usage:
    is_valid, errors, warnings = validate_diagnostics_report(report_dict)
    # or from file:
    is_valid, errors, warnings = validate_diagnostics_report_from_file("report.json")
"""

import json
import math
from typing import Union

from ..schemas import DiagnosticsReport


# Required top-level fields in a DiagnosticsReport
REQUIRED_FIELDS = [
    "timestamp", "run_id", "step", "val_bpb", "training_seconds",
    "gradient_stats", "activation_stats", "attention_stats", "loss_decomposition",
]

# Required fields per sub-schema
GRADIENT_REQUIRED = ["layer_idx", "norm", "mean", "std", "max_abs", "dead_fraction"]
ACTIVATION_REQUIRED = ["layer_idx", "mean", "std", "max_abs", "dead_neuron_count", "dead_neuron_fraction"]
ATTENTION_REQUIRED = ["layer_idx", "head_idx", "entropy", "collapse_score", "max_attention_weight"]
LOSS_BUCKET_REQUIRED = ["bucket_name", "token_count", "mean_loss", "std_loss"]

# Thresholds for anomaly detection
MAX_GRADIENT_NORM = 1000.0
MAX_ACTIVATION_ABS = 1e6
DEAD_NEURON_FRACTION_WARN = 0.5
EXPECTED_LOSS_BUCKETS = {"top_1k", "1k_10k", "10k_plus", "rare"}


def validate_diagnostics_report(
    report: Union[dict, str, DiagnosticsReport],
) -> tuple[bool, list[str], list[str]]:
    """Validate a DiagnosticsReport.

    Args:
        report: A dict, JSON file path, or DiagnosticsReport instance.

    Returns:
        (is_valid, errors, warnings) where:
          - is_valid: True if no errors found
          - errors: List of critical issues (missing fields, NaN values, etc.)
          - warnings: List of non-critical anomalies
    """
    errors: list[str] = []
    warnings: list[str] = []

    # Normalize to dict
    if isinstance(report, DiagnosticsReport):
        data = report.to_dict()
    elif isinstance(report, str):
        try:
            with open(report) as f:
                data = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError) as e:
            return False, [f"Failed to load report: {e}"], []
    elif isinstance(report, dict):
        data = report
    else:
        return False, [f"Invalid report type: {type(report)}"], []

    # --- Schema completeness ---
    for field_name in REQUIRED_FIELDS:
        if field_name not in data:
            errors.append(f"Missing required field: {field_name}")

    # --- Validate gradient_stats ---
    grad_stats = data.get("gradient_stats", [])
    if not isinstance(grad_stats, list):
        errors.append("gradient_stats must be a list")
        grad_stats = []

    for i, gs in enumerate(grad_stats):
        prefix = f"gradient_stats[{i}]"
        _check_required_fields(gs, GRADIENT_REQUIRED, prefix, errors)
        _check_no_nan(gs, ["norm", "mean", "std", "max_abs", "dead_fraction"], prefix, errors)

        norm = gs.get("norm")
        if isinstance(norm, (int, float)) and norm > MAX_GRADIENT_NORM:
            warnings.append(f"{prefix}: gradient norm {norm:.2f} exceeds {MAX_GRADIENT_NORM}")

        dead_frac = gs.get("dead_fraction")
        if isinstance(dead_frac, (int, float)) and dead_frac > DEAD_NEURON_FRACTION_WARN:
            warnings.append(
                f"{prefix}: dead gradient fraction {dead_frac:.2%} exceeds {DEAD_NEURON_FRACTION_WARN:.0%}"
            )

    # --- Validate activation_stats ---
    act_stats = data.get("activation_stats", [])
    if not isinstance(act_stats, list):
        errors.append("activation_stats must be a list")
        act_stats = []

    for i, act in enumerate(act_stats):
        prefix = f"activation_stats[{i}]"
        _check_required_fields(act, ACTIVATION_REQUIRED, prefix, errors)
        _check_no_nan(act, ["mean", "std", "max_abs", "dead_neuron_fraction"], prefix, errors)

        max_abs = act.get("max_abs")
        if isinstance(max_abs, (int, float)) and max_abs > MAX_ACTIVATION_ABS:
            warnings.append(f"{prefix}: max_abs {max_abs:.2e} exceeds {MAX_ACTIVATION_ABS:.0e}")

        dead_frac = act.get("dead_neuron_fraction")
        if isinstance(dead_frac, (int, float)) and dead_frac > DEAD_NEURON_FRACTION_WARN:
            warnings.append(
                f"{prefix}: dead neuron fraction {dead_frac:.2%} exceeds {DEAD_NEURON_FRACTION_WARN:.0%}"
            )

        dead_count = act.get("dead_neuron_count")
        if isinstance(dead_count, (int, float)) and dead_count < 0:
            errors.append(f"{prefix}: dead_neuron_count cannot be negative")

    # --- Validate attention_stats ---
    attn_stats = data.get("attention_stats", [])
    if not isinstance(attn_stats, list):
        errors.append("attention_stats must be a list")
        attn_stats = []

    for i, att in enumerate(attn_stats):
        prefix = f"attention_stats[{i}]"
        _check_required_fields(att, ATTENTION_REQUIRED, prefix, errors)
        _check_no_nan(att, ["entropy", "collapse_score", "max_attention_weight"], prefix, errors)

        entropy = att.get("entropy")
        if isinstance(entropy, (int, float)) and entropy < 0:
            errors.append(f"{prefix}: entropy cannot be negative (got {entropy})")

        collapse = att.get("collapse_score")
        if isinstance(collapse, (int, float)):
            if collapse < 0 or collapse > 1:
                errors.append(f"{prefix}: collapse_score must be in [0, 1] (got {collapse})")

        max_w = att.get("max_attention_weight")
        if isinstance(max_w, (int, float)):
            if max_w < 0 or max_w > 1:
                errors.append(f"{prefix}: max_attention_weight must be in [0, 1] (got {max_w})")

    # --- Validate loss_decomposition ---
    loss_buckets = data.get("loss_decomposition", [])
    if not isinstance(loss_buckets, list):
        errors.append("loss_decomposition must be a list")
        loss_buckets = []

    bucket_names_seen = set()
    for i, lb in enumerate(loss_buckets):
        prefix = f"loss_decomposition[{i}]"
        _check_required_fields(lb, LOSS_BUCKET_REQUIRED, prefix, errors)
        _check_no_nan(lb, ["mean_loss", "std_loss"], prefix, errors)

        name = lb.get("bucket_name", "")
        bucket_names_seen.add(name)

        count = lb.get("token_count")
        if isinstance(count, (int, float)) and count == 0:
            warnings.append(f"{prefix}: bucket '{name}' has 0 tokens")

        mean_loss = lb.get("mean_loss")
        if isinstance(mean_loss, (int, float)) and mean_loss < 0:
            warnings.append(f"{prefix}: negative mean_loss {mean_loss} in bucket '{name}'")

    # Check that expected buckets are present
    missing_buckets = EXPECTED_LOSS_BUCKETS - bucket_names_seen
    if missing_buckets and loss_buckets:
        warnings.append(f"Missing expected loss buckets: {missing_buckets}")

    # --- Top-level value checks ---
    val_bpb = data.get("val_bpb")
    if isinstance(val_bpb, (int, float)):
        if math.isnan(val_bpb):
            errors.append("val_bpb is NaN")
        elif val_bpb < 0:
            warnings.append(f"val_bpb is negative: {val_bpb}")

    step = data.get("step")
    if isinstance(step, (int, float)) and step < 0:
        errors.append(f"step cannot be negative: {step}")

    is_valid = len(errors) == 0
    return is_valid, errors, warnings


def validate_diagnostics_report_from_file(
    path: str,
) -> tuple[bool, list[str], list[str]]:
    """Convenience wrapper that loads a JSON file and validates it."""
    return validate_diagnostics_report(path)


def _check_required_fields(
    obj: dict, required: list[str], prefix: str, errors: list[str]
) -> None:
    """Check that all required fields are present in a dict."""
    if not isinstance(obj, dict):
        errors.append(f"{prefix}: expected dict, got {type(obj).__name__}")
        return
    for field_name in required:
        if field_name not in obj:
            errors.append(f"{prefix}: missing required field '{field_name}'")


def _check_no_nan(
    obj: dict, fields: list[str], prefix: str, errors: list[str]
) -> None:
    """Check that specified numeric fields are not NaN."""
    if not isinstance(obj, dict):
        return
    for field_name in fields:
        val = obj.get(field_name)
        if isinstance(val, float) and math.isnan(val):
            errors.append(f"{prefix}: {field_name} is NaN")
