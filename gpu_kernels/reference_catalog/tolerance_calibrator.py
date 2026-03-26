"""
ToleranceBoundCalibrator: empirically determines the tightest achievable
tolerance per dtype by running a reference implementation multiple times
and measuring the maximum deviation.
"""

import json
import os
from typing import Callable

import torch

from ..schemas import ToleranceBounds, save_json


class ToleranceBoundCalibrator:
    """Calibrates tolerance bounds by running reference code repeatedly."""

    def __init__(self, n_runs: int = 100, safety_margin: float = 2.0):
        """
        Args:
            n_runs: Number of runs to measure deviation (default 100).
            safety_margin: Multiplier on observed max deviation to set
                tolerance (default 2.0x for safety).
        """
        self.n_runs = n_runs
        self.safety_margin = safety_margin

    def calibrate(
        self,
        reference_callable: Callable,
        input_tensors: list[torch.Tensor],
        dtypes: list[torch.dtype] | None = None,
    ) -> ToleranceBounds:
        """Run the reference function repeatedly and measure deviation.

        Args:
            reference_callable: Function to call with input_tensors.
            input_tensors: List of input tensors (will be cast to each dtype).
            dtypes: List of dtypes to test. Defaults to [fp32, fp16, bf16].

        Returns:
            ToleranceBounds with empirically calibrated atol/rtol per dtype.
        """
        if dtypes is None:
            dtypes = [torch.float32, torch.float16, torch.bfloat16]

        bounds = ToleranceBounds()

        for dtype in dtypes:
            atol, rtol = self._measure_deviation(
                reference_callable, input_tensors, dtype
            )
            tol_dict = {"atol": atol, "rtol": rtol}

            if dtype == torch.float32:
                bounds.fp32 = tol_dict
            elif dtype == torch.float16:
                bounds.fp16 = tol_dict
            elif dtype == torch.bfloat16:
                bounds.bf16 = tol_dict

        return bounds

    def calibrate_and_save(
        self,
        reference_callable: Callable,
        input_tensors: list[torch.Tensor],
        output_path: str = "tolerances.json",
        dtypes: list[torch.dtype] | None = None,
    ) -> ToleranceBounds:
        """Calibrate tolerances and save to JSON.

        Args:
            reference_callable: Function to call.
            input_tensors: Input tensors.
            output_path: Path to save tolerances.json.
            dtypes: Dtypes to test.

        Returns:
            The calibrated ToleranceBounds.
        """
        bounds = self.calibrate(reference_callable, input_tensors, dtypes)
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        save_json(bounds, output_path)
        return bounds

    def _measure_deviation(
        self,
        reference_callable: Callable,
        input_tensors: list[torch.Tensor],
        dtype: torch.dtype,
    ) -> tuple[float, float]:
        """Measure max absolute and relative deviation across n_runs.

        Returns:
            Tuple of (atol, rtol) with safety margin applied.
        """
        # Cast inputs to target dtype
        cast_inputs = []
        for t in input_tensors:
            if t.is_floating_point():
                cast_inputs.append(t.to(dtype=dtype))
            else:
                cast_inputs.append(t)

        # Run once to get the reference output
        with torch.no_grad():
            ref_output = reference_callable(*cast_inputs)
            if isinstance(ref_output, tuple):
                ref_output = ref_output[0]
            ref_output = ref_output.detach().clone()

        max_abs_diff = 0.0
        max_rel_diff = 0.0

        for _ in range(self.n_runs):
            with torch.no_grad():
                output = reference_callable(*cast_inputs)
                if isinstance(output, tuple):
                    output = output[0]
                output = output.detach()

            abs_diff = (output - ref_output).abs()
            max_abs_diff = max(max_abs_diff, abs_diff.max().item())

            # Relative diff: avoid division by zero
            denom = ref_output.abs().clamp(min=1e-12)
            rel_diff = abs_diff / denom
            max_rel_diff = max(max_rel_diff, rel_diff.max().item())

        # Apply safety margin, with minimum floors per dtype
        atol = max(max_abs_diff * self.safety_margin, _min_atol(dtype))
        rtol = max(max_rel_diff * self.safety_margin, _min_rtol(dtype))

        return atol, rtol


def _min_atol(dtype: torch.dtype) -> float:
    """Minimum atol floor per dtype."""
    if dtype == torch.float32:
        return 1e-6
    elif dtype == torch.float16:
        return 1e-3
    elif dtype == torch.bfloat16:
        return 1e-3
    return 1e-5


def _min_rtol(dtype: torch.dtype) -> float:
    """Minimum rtol floor per dtype."""
    if dtype == torch.float32:
        return 1e-5
    elif dtype == torch.float16:
        return 1e-3
    elif dtype == torch.bfloat16:
        return 1e-2
    return 1e-4
