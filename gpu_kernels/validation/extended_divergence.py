"""
Extended divergence validation — 2000-step parallel training comparison.

Compares loss, grad norms per layer, attention entropy, and final parameters
between kernel-enabled and reference (PyTorch-only) training runs.
"""

import json
import math
import os
import time

from ..schemas import ExtendedValidationResult, KernelConfigEntry, load_json


class ExtendedDivergenceValidator:
    """Run extended training comparison to detect subtle kernel divergence."""

    def __init__(self, data_dir: str = None):
        self.data_dir = data_dir or os.path.join(
            os.path.dirname(__file__), "..", "data"
        )

    def validate(
        self,
        kernel_config: dict,
        n_steps: int = 2000,
        seed: int = 42,
        check_interval: int = 100,
        loss_threshold: float = 0.0005,
        grad_norm_threshold: float = 0.01,
        entropy_threshold: float = 0.05,
        param_atol: float = 1e-3,
    ) -> ExtendedValidationResult:
        """
        Run extended divergence validation.

        Runs two parallel training simulations (kernel vs reference) for
        n_steps. Every check_interval steps, compares:
          - Loss: < loss_threshold relative divergence
          - Grad norms per layer: < grad_norm_threshold relative divergence
          - Attention entropy: < entropy_threshold relative divergence
        At the final step, compares all parameters with atol=param_atol.

        If any check fails, isolates culprit by disabling kernels one at a time.

        Args:
            kernel_config: Dict mapping kernel IDs to KernelConfigEntry dicts.
            n_steps: Number of training steps to simulate.
            seed: Random seed for reproducibility.
            check_interval: Steps between comparison checks.
            loss_threshold: Max relative loss divergence (0.05% = 0.0005).
            grad_norm_threshold: Max relative grad norm divergence per layer.
            entropy_threshold: Max relative attention entropy divergence.
            param_atol: Absolute tolerance for final parameter comparison.

        Returns:
            ExtendedValidationResult with pass/fail and diagnostics.
        """
        result = ExtendedValidationResult(n_steps=n_steps)

        # Collect active kernels from config
        active_kernels = {}
        for kid, entry in kernel_config.items():
            if isinstance(entry, dict):
                if entry.get("enabled", True) and entry.get("backend") == "triton":
                    active_kernels[kid] = entry
            elif hasattr(entry, "enabled"):
                if entry.enabled and entry.backend == "triton":
                    active_kernels[kid] = entry

        if not active_kernels:
            # No active kernels — trivially passes
            result.passed = True
            return result

        # Run the comparison
        ref_losses, kernel_losses = [], []
        ref_grad_norms, kernel_grad_norms = {}, {}
        ref_entropy, kernel_entropy = [], []

        failure = None

        for step in range(1, n_steps + 1):
            # Simulate training step metrics for both runs.
            # In a real implementation these would come from actual forward/backward
            # passes. Here we provide the scaffolding that calls into the training
            # loop when available, or records placeholder data.

            ref_step_data = self._run_reference_step(step, seed)
            kernel_step_data = self._run_kernel_step(step, seed, active_kernels)

            ref_losses.append(ref_step_data["loss"])
            kernel_losses.append(kernel_step_data["loss"])
            ref_entropy.append(ref_step_data.get("attention_entropy", 0.0))
            kernel_entropy.append(kernel_step_data.get("attention_entropy", 0.0))

            for layer, norm in ref_step_data.get("grad_norms", {}).items():
                ref_grad_norms.setdefault(layer, []).append(norm)
            for layer, norm in kernel_step_data.get("grad_norms", {}).items():
                kernel_grad_norms.setdefault(layer, []).append(norm)

            # Check at interval
            if step % check_interval == 0:
                check = self._check_step(
                    step,
                    ref_losses[-1],
                    kernel_losses[-1],
                    ref_step_data.get("grad_norms", {}),
                    kernel_step_data.get("grad_norms", {}),
                    ref_entropy[-1],
                    kernel_entropy[-1],
                    loss_threshold,
                    grad_norm_threshold,
                    entropy_threshold,
                )
                if not check["passed"]:
                    failure = check
                    break

        # Final parameter comparison at last step
        if failure is None:
            ref_params = self._get_reference_params(seed)
            kernel_params = self._get_kernel_params(seed, active_kernels)
            param_div = self._compare_params(ref_params, kernel_params, param_atol)
            if not param_div["match"]:
                failure = {
                    "step": n_steps,
                    "metric": "final_params",
                    "divergence": param_div["max_divergence"],
                }

        if failure is None:
            result.passed = True
            result.max_loss_divergence = self._max_relative_div(ref_losses, kernel_losses)
            result.max_grad_norm_divergence = self._max_grad_div(
                ref_grad_norms, kernel_grad_norms
            )
            result.max_param_divergence = 0.0
        else:
            result.passed = False
            result.failing_step = failure.get("step")
            result.failing_metric = failure.get("metric", "")
            result.max_loss_divergence = failure.get("divergence", 0.0)

            # Isolate culprit kernel
            culprit = self._isolate_culprit(
                failure, active_kernels, kernel_config, seed, n_steps,
                check_interval, loss_threshold, grad_norm_threshold,
                entropy_threshold, param_atol,
            )
            result.culprit_kernel_id = culprit

        self._save_result(result)
        return result

    def _check_step(
        self, step, ref_loss, kernel_loss, ref_grads, kernel_grads,
        ref_ent, kernel_ent, loss_thr, grad_thr, ent_thr,
    ) -> dict:
        """Check divergence at a single step."""
        # Loss check
        if ref_loss != 0:
            loss_div = abs(kernel_loss - ref_loss) / abs(ref_loss)
        else:
            loss_div = abs(kernel_loss - ref_loss)
        if loss_div > loss_thr:
            return {"passed": False, "step": step, "metric": "loss", "divergence": loss_div}

        # Grad norm check per layer
        for layer in ref_grads:
            rg = ref_grads[layer]
            kg = kernel_grads.get(layer, 0.0)
            if rg != 0:
                gd = abs(kg - rg) / abs(rg)
            else:
                gd = abs(kg - rg)
            if gd > grad_thr:
                return {
                    "passed": False, "step": step,
                    "metric": f"grad_norm_{layer}", "divergence": gd,
                }

        # Attention entropy check
        if ref_ent != 0:
            ent_div = abs(kernel_ent - ref_ent) / abs(ref_ent)
        else:
            ent_div = abs(kernel_ent - ref_ent)
        if ent_div > ent_thr:
            return {
                "passed": False, "step": step,
                "metric": "attention_entropy", "divergence": ent_div,
            }

        return {"passed": True}

    def _isolate_culprit(
        self, failure, active_kernels, full_config, seed, n_steps,
        check_interval, loss_thr, grad_thr, ent_thr, param_atol,
    ) -> str:
        """Disable kernels one at a time to find the culprit."""
        for kid in active_kernels:
            # Build config with this kernel disabled
            reduced_config = {}
            for k, v in full_config.items():
                entry = dict(v) if isinstance(v, dict) else v.to_dict()
                if k == kid:
                    entry["enabled"] = False
                reduced_config[k] = entry

            # Re-run validation with reduced config
            sub_result = self.validate(
                reduced_config,
                n_steps=min(n_steps, failure.get("step", n_steps)),
                seed=seed,
                check_interval=check_interval,
                loss_threshold=loss_thr,
                grad_norm_threshold=grad_thr,
                entropy_threshold=ent_thr,
                param_atol=param_atol,
            )
            if sub_result.passed:
                return kid
        return ""

    def _run_reference_step(self, step: int, seed: int) -> dict:
        """Run a reference (PyTorch-only) training step."""
        raise NotImplementedError(
            "_run_reference_step requires a real training harness. "
            "Override in a subclass with access to model and data."
        )

    def _run_kernel_step(self, step: int, seed: int, kernels: dict) -> dict:
        """Run a kernel-enabled training step."""
        raise NotImplementedError(
            "_run_kernel_step requires a real training harness. "
            "Override in a subclass with access to model, data, and kernels."
        )

    def _get_reference_params(self, seed: int) -> dict:
        """Get reference model parameters after training."""
        raise NotImplementedError(
            "_get_reference_params requires access to model state_dict. "
            "Override in a subclass."
        )

    def _get_kernel_params(self, seed: int, kernels: dict) -> dict:
        """Get kernel model parameters after training."""
        raise NotImplementedError(
            "_get_kernel_params requires access to model state_dict. "
            "Override in a subclass."
        )

    def _compare_params(self, ref: dict, kernel: dict, atol: float) -> dict:
        """Compare parameter dictionaries."""
        max_div = 0.0
        for key in ref:
            if key not in kernel:
                return {"match": False, "max_divergence": float("inf")}
            r_val = ref[key]
            k_val = kernel[key]
            if isinstance(r_val, (int, float)) and isinstance(k_val, (int, float)):
                div = abs(r_val - k_val)
                max_div = max(max_div, div)
                if div > atol:
                    return {"match": False, "max_divergence": div}
        return {"match": True, "max_divergence": max_div}

    def _max_relative_div(self, ref_list: list, kernel_list: list) -> float:
        """Compute max relative divergence between two loss lists."""
        max_d = 0.0
        for r, k in zip(ref_list, kernel_list):
            if r != 0:
                max_d = max(max_d, abs(k - r) / abs(r))
            else:
                max_d = max(max_d, abs(k - r))
        return max_d

    def _max_grad_div(self, ref_grads: dict, kernel_grads: dict) -> float:
        """Compute max relative grad norm divergence across layers."""
        max_d = 0.0
        for layer in ref_grads:
            for r, k in zip(
                ref_grads.get(layer, []),
                kernel_grads.get(layer, []),
            ):
                if r != 0:
                    max_d = max(max_d, abs(k - r) / abs(r))
                else:
                    max_d = max(max_d, abs(k - r))
        return max_d

    def _save_result(self, result: ExtendedValidationResult):
        """Persist result to data directory."""
        out_dir = os.path.join(self.data_dir, "extended_validation")
        try:
            os.makedirs(out_dir, exist_ok=True)
            path = os.path.join(out_dir, f"result_{int(time.time())}.json")
            with open(path, "w") as f:
                json.dump(result.to_dict(), f, indent=2)
        except OSError:
            pass
