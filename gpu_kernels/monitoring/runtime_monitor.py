"""
Runtime correctness monitoring for deployed GPU kernels.

Every N steps, compares kernel outputs against PyTorch reference outputs.
Issues warnings or critical alerts on divergence and can auto-disable kernels.
"""

import json
import math
import os
import time

from ..schemas import RuntimeAlert


class RuntimeCorrectnessMonitor:
    """Monitor kernel correctness at runtime by periodic comparison."""

    def __init__(self, data_dir: str = None):
        self.data_dir = data_dir or os.path.join(
            os.path.dirname(__file__), "..", "data"
        )
        self._alert_log = []
        self._disabled_kernels = set()

    def should_check(self, step: int, check_interval: int = 100) -> bool:
        """Return True if a correctness check should run at this step."""
        return step > 0 and step % check_interval == 0

    def check_step(
        self,
        step: int,
        kernel_outputs: dict,
        reference_outputs: dict,
        tolerances: dict,
    ) -> list:
        """
        Compare kernel outputs against reference at a given step.

        Args:
            step: Current training step.
            kernel_outputs: Dict mapping kernel_id -> output value(s).
            reference_outputs: Dict mapping kernel_id -> reference output value(s).
            tolerances: Dict mapping kernel_id -> tolerance float.

        Returns:
            List of RuntimeAlert for any divergence detected.
        """
        alerts = []

        for kernel_id in kernel_outputs:
            if kernel_id in self._disabled_kernels:
                continue

            if kernel_id not in reference_outputs:
                continue

            k_out = kernel_outputs[kernel_id]
            r_out = reference_outputs[kernel_id]
            tol = tolerances.get(kernel_id, 1e-5)

            max_abs_err = self._compute_max_abs_error(k_out, r_out)

            # Check for NaN
            if math.isnan(max_abs_err) or self._has_nan(k_out):
                alert = RuntimeAlert(
                    kernel_id=kernel_id,
                    step=step,
                    max_abs_error=float("nan"),
                    tolerance_threshold=tol,
                    severity="critical",
                    timestamp=time.time(),
                )
                alerts.append(alert)
                self._auto_disable(kernel_id, "NaN detected in output")
                continue

            # Critical: > 10x tolerance
            if max_abs_err > tol * 10:
                alert = RuntimeAlert(
                    kernel_id=kernel_id,
                    step=step,
                    max_abs_error=max_abs_err,
                    tolerance_threshold=tol,
                    severity="critical",
                    timestamp=time.time(),
                )
                alerts.append(alert)
                self._auto_disable(kernel_id, f"Divergence {max_abs_err:.6e} > 10x tolerance {tol:.6e}")

            # Warning: > 2x tolerance
            elif max_abs_err > tol * 2:
                alert = RuntimeAlert(
                    kernel_id=kernel_id,
                    step=step,
                    max_abs_error=max_abs_err,
                    tolerance_threshold=tol,
                    severity="warning",
                    timestamp=time.time(),
                )
                alerts.append(alert)

        # Log alerts
        for alert in alerts:
            self._log_alert(alert)

        return alerts

    def _compute_max_abs_error(self, kernel_out, ref_out) -> float:
        """Compute maximum absolute error between outputs."""
        if isinstance(kernel_out, (int, float)) and isinstance(ref_out, (int, float)):
            if math.isnan(kernel_out) or math.isnan(ref_out):
                return float("nan")
            return abs(kernel_out - ref_out)

        if isinstance(kernel_out, (list, tuple)) and isinstance(ref_out, (list, tuple)):
            if len(kernel_out) != len(ref_out):
                return float("inf")
            max_err = 0.0
            for k, r in zip(kernel_out, ref_out):
                err = self._compute_max_abs_error(k, r)
                if math.isnan(err):
                    return float("nan")
                max_err = max(max_err, err)
            return max_err

        if isinstance(kernel_out, dict) and isinstance(ref_out, dict):
            max_err = 0.0
            for key in kernel_out:
                if key not in ref_out:
                    continue
                err = self._compute_max_abs_error(kernel_out[key], ref_out[key])
                if math.isnan(err):
                    return float("nan")
                max_err = max(max_err, err)
            return max_err

        # For tensor-like objects
        if hasattr(kernel_out, "sub"):
            try:
                diff = kernel_out.sub(ref_out).abs()
                if hasattr(diff, "max"):
                    return float(diff.max())
            except Exception:
                pass

        return 0.0

    def _has_nan(self, value) -> bool:
        """Check if any value is NaN."""
        if isinstance(value, float):
            return math.isnan(value)
        if isinstance(value, (list, tuple)):
            return any(self._has_nan(v) for v in value)
        if isinstance(value, dict):
            return any(self._has_nan(v) for v in value.values())
        if hasattr(value, "isnan"):
            try:
                return bool(value.isnan().any())
            except Exception:
                pass
        return False

    def _auto_disable(self, kernel_id: str, reason: str):
        """Mark a kernel as disabled due to critical alert."""
        self._disabled_kernels.add(kernel_id)
        self._log_disable(kernel_id, reason)

    def _log_alert(self, alert: RuntimeAlert):
        """Append alert to persistent log."""
        self._alert_log.append(alert.to_dict())
        log_path = os.path.join(self.data_dir, "runtime_alerts.jsonl")
        try:
            os.makedirs(os.path.dirname(log_path), exist_ok=True)
            with open(log_path, "a") as f:
                f.write(json.dumps(alert.to_dict()) + "\n")
        except OSError:
            pass

    def _log_disable(self, kernel_id: str, reason: str):
        """Log auto-disable event."""
        entry = {
            "kernel_id": kernel_id,
            "action": "auto_disable",
            "reason": reason,
            "timestamp": time.time(),
        }
        log_path = os.path.join(self.data_dir, "kernel_disable_log.jsonl")
        try:
            os.makedirs(os.path.dirname(log_path), exist_ok=True)
            with open(log_path, "a") as f:
                f.write(json.dumps(entry) + "\n")
        except OSError:
            pass

    def get_alerts(self) -> list:
        """Return all alerts from this session."""
        return list(self._alert_log)

    def get_disabled_kernels(self) -> set:
        """Return set of auto-disabled kernel IDs."""
        return set(self._disabled_kernels)
