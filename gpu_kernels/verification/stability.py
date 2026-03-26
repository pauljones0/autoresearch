"""
Numerical stability probing for generated kernels.
"""

import traceback

import torch

from gpu_kernels.schemas import StabilityResult


class NumericalStabilityProber:
    """Probe a kernel for numerical stability issues."""

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def probe(
        self,
        kernel_callable,
        test_inputs,
        n_runs: int = 10,
    ) -> StabilityResult:
        """Run stability probes on a kernel.

        Args:
            kernel_callable: The kernel function to test.
            test_inputs: A list/tuple of input tensors.
            n_runs: Number of repeated runs for determinism check.

        Returns:
            StabilityResult with determinism, overflow, and underflow info.
        """
        if not torch.cuda.is_available():
            return StabilityResult()

        if isinstance(test_inputs, torch.Tensor):
            test_inputs = [test_inputs]

        result = StabilityResult()

        # --- Determinism check: run n_runs times on identical inputs ---
        try:
            outputs = []
            for _ in range(n_runs):
                out = kernel_callable(*test_inputs)
                if isinstance(out, torch.Tensor):
                    outputs.append(out.detach().clone().float())
                elif isinstance(out, (list, tuple)):
                    outputs.append(out[0].detach().clone().float())

            if outputs:
                ref = outputs[0]
                diffs = torch.stack([o - ref for o in outputs[1:]])
                max_diff = diffs.abs().max().item()
                result.is_deterministic = max_diff == 0.0

                # Coefficient of variation across runs
                stacked = torch.stack(outputs)
                mean_vals = stacked.mean(dim=0)
                std_vals = stacked.std(dim=0)
                safe_mean = mean_vals.abs().clamp(min=1e-12)
                cv = (std_vals / safe_mean)
                result.max_cv_across_runs = cv.max().item()
        except Exception:
            result.is_deterministic = False
            result.extreme_input_results.append({
                "test": "determinism",
                "error": traceback.format_exc(),
            })

        # --- Extreme value tests ---
        sample = test_inputs[0] if isinstance(test_inputs, (list, tuple)) else test_inputs
        if isinstance(sample, torch.Tensor) and sample.is_floating_point():
            dtype = sample.dtype
            result.extreme_input_results.extend(self._test_extremes(
                kernel_callable, test_inputs, sample, dtype, result
            ))

        return result

    def _test_extremes(self, kernel_callable, test_inputs, sample, dtype, result):
        """Test near-overflow, near-underflow, and denormalized inputs."""
        extreme_results = []

        # Get dtype bounds
        finfo = torch.finfo(dtype)

        extreme_cases = [
            ("near_overflow", sample.new_full(sample.shape, finfo.max * 0.9)),
            ("near_underflow", sample.new_full(sample.shape, finfo.tiny * 2)),
            ("denormalized", sample.new_full(sample.shape, finfo.tiny * 0.5)),
        ]

        for name, extreme_tensor in extreme_cases:
            try:
                inputs = [extreme_tensor if i == 0 else t for i, t in enumerate(test_inputs)]
                out = kernel_callable(*inputs)
                if isinstance(out, (list, tuple)):
                    out = out[0]
                if isinstance(out, torch.Tensor):
                    has_inf = torch.isinf(out).any().item()
                    has_nan = torch.isnan(out).any().item()
                    if has_inf and name == "near_overflow":
                        result.overflow_detected = True
                    if has_nan and name in ("near_underflow", "denormalized"):
                        result.underflow_detected = True
                    if name == "denormalized" and (has_nan or has_inf):
                        result.denormal_handling = "problematic"
                    extreme_results.append({
                        "test": name,
                        "has_inf": has_inf,
                        "has_nan": has_nan,
                        "passed": not has_nan,
                    })
            except Exception:
                extreme_results.append({
                    "test": name,
                    "error": traceback.format_exc(),
                    "passed": False,
                })

        return extreme_results
