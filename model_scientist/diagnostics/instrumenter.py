"""
DiagnosticsInstrumenter: hooks into a GPT model to capture per-layer
gradient norms, activation statistics, and dead neuron counts.

Usage:
    instrumenter = DiagnosticsInstrumenter(capture_every_n_steps=50)
    instrumenter.instrument(model)
    # ... training loop ...
    instrumenter.capture_step(step)
    # ... after training ...
    report = instrumenter.generate_report()
    instrumenter.remove_hooks()
"""

import math
from typing import Optional

import torch
import torch.nn as nn

from ..schemas import (
    DiagnosticsReport,
    GradientStats,
    ActivationStats,
)


class DiagnosticsInstrumenter:
    """Attaches forward/backward hooks to capture per-layer diagnostics."""

    def __init__(self, capture_every_n_steps: int = 50):
        self.capture_every_n_steps = capture_every_n_steps
        self._hooks = []
        self._model = None

        # Raw storage: list of dicts per captured step
        self._activation_records: list[dict] = []  # {layer_idx -> stats_dict}
        self._gradient_records: list[dict] = []     # {layer_idx -> stats_dict}

        # Temporary buffers for current step (populated by hooks)
        self._current_activations: dict[int, dict] = {}
        self._current_gradients: dict[int, dict] = {}
        self._capturing = False
        self._n_layers = 0

    def instrument(self, model: nn.Module) -> None:
        """Attach forward and backward hooks to all transformer blocks.

        Works with both raw and torch.compile'd models by finding Block
        instances through the module hierarchy.
        """
        self.remove_hooks()
        self._model = model

        # Unwrap compiled model if needed
        raw_model = model
        if hasattr(model, "_orig_mod"):
            raw_model = model._orig_mod

        blocks = raw_model.transformer.h
        self._n_layers = len(blocks)

        for layer_idx, block in enumerate(blocks):
            # Forward hook on the MLP submodule to capture activations
            # after the MLP forward pass (ReLU squared activation)
            mlp = block.mlp
            fh = mlp.register_forward_hook(
                self._make_forward_hook(layer_idx)
            )
            self._hooks.append(fh)

            # Full backward hook on the block to capture gradient norms
            bh = block.register_full_backward_hook(
                self._make_backward_hook(layer_idx)
            )
            self._hooks.append(bh)

    def _make_forward_hook(self, layer_idx: int):
        """Create a forward hook that captures activation stats for a layer."""
        def hook(module, input, output):
            if not self._capturing:
                return
            # output is the MLP output tensor: (B, T, 4*n_embd) after relu^2
            # We compute stats on the intermediate activation (input to c_proj)
            # but we only have access to the final output here.
            # Use the output of the full MLP block for activation health.
            with torch.no_grad():
                x = output.detach().float()
                self._current_activations[layer_idx] = {
                    "mean": x.mean().item(),
                    "std": x.std().item(),
                    "max_abs": x.abs().max().item(),
                    # Dead neurons: dimensions where activation is zero across
                    # all batch elements and positions
                    "dead_mask": (x.abs().sum(dim=(0, 1)) == 0),
                }
        return hook

    def _make_backward_hook(self, layer_idx: int):
        """Create a backward hook that captures gradient stats for a layer."""
        def hook(module, grad_input, grad_output):
            if not self._capturing:
                return
            # grad_output[0] is the gradient flowing back through the block
            with torch.no_grad():
                g = grad_output[0]
                if g is None:
                    return
                g = g.detach().float()
                norm_val = g.norm().item()
                self._current_gradients[layer_idx] = {
                    "norm": norm_val,
                    "mean": g.mean().item(),
                    "std": g.std().item(),
                    "max_abs": g.abs().max().item(),
                    # Fraction of gradient elements that are exactly zero
                    "dead_fraction": (g == 0).float().mean().item(),
                }
        return hook

    def capture_step(self, step: int) -> None:
        """Call this after each training step. Records stats if step aligns
        with the capture interval."""
        if step % self.capture_every_n_steps != 0:
            return

        # Enable capturing for this step's backward/forward pass
        # Note: hooks fire during the forward/backward pass itself.
        # This method is called AFTER the step, so we store whatever
        # was captured during the most recent forward+backward.
        if self._current_activations:
            self._activation_records.append(dict(self._current_activations))
        if self._current_gradients:
            self._gradient_records.append(dict(self._current_gradients))

        # Reset temporary buffers
        self._current_activations = {}
        self._current_gradients = {}

    def enable_capture(self) -> None:
        """Enable hook capture. Call before the forward pass of a step
        you want to capture."""
        self._capturing = True

    def disable_capture(self) -> None:
        """Disable hook capture after the backward pass completes."""
        self._capturing = False

    def should_capture(self, step: int) -> bool:
        """Check if this step should be captured."""
        return step % self.capture_every_n_steps == 0

    def generate_report(
        self,
        run_id: str = "",
        step: int = 0,
        val_bpb: float = 0.0,
        training_seconds: float = 0.0,
    ) -> DiagnosticsReport:
        """Aggregate captured stats into a DiagnosticsReport.

        Averages statistics across all captured steps to produce a single
        summary per layer.
        """
        gradient_stats = self._aggregate_gradient_stats()
        activation_stats = self._aggregate_activation_stats()

        report = DiagnosticsReport(
            run_id=run_id,
            step=step,
            val_bpb=val_bpb,
            training_seconds=training_seconds,
            gradient_stats=[vars(g) for g in gradient_stats],
            activation_stats=[vars(a) for a in activation_stats],
        )
        return report

    def _aggregate_gradient_stats(self) -> list[GradientStats]:
        """Average gradient stats across captured steps, per layer."""
        if not self._gradient_records:
            return []

        results = []
        for layer_idx in range(self._n_layers):
            norms, means, stds, max_abss, dead_fracs = [], [], [], [], []
            for record in self._gradient_records:
                if layer_idx in record:
                    s = record[layer_idx]
                    norms.append(s["norm"])
                    means.append(s["mean"])
                    stds.append(s["std"])
                    max_abss.append(s["max_abs"])
                    dead_fracs.append(s["dead_fraction"])

            if not norms:
                continue

            results.append(GradientStats(
                layer_idx=layer_idx,
                norm=_safe_mean(norms),
                mean=_safe_mean(means),
                std=_safe_mean(stds),
                max_abs=_safe_mean(max_abss),
                dead_fraction=_safe_mean(dead_fracs),
            ))
        return results

    def _aggregate_activation_stats(self) -> list[ActivationStats]:
        """Average activation stats across captured steps, per layer."""
        if not self._activation_records:
            return []

        results = []
        for layer_idx in range(self._n_layers):
            means, stds, max_abss = [], [], []
            dead_counts = []
            total_neurons = None

            for record in self._activation_records:
                if layer_idx in record:
                    s = record[layer_idx]
                    means.append(s["mean"])
                    stds.append(s["std"])
                    max_abss.append(s["max_abs"])
                    dead_mask = s["dead_mask"]
                    dead_counts.append(int(dead_mask.sum().item()))
                    if total_neurons is None:
                        total_neurons = dead_mask.numel()

            if not means:
                continue

            avg_dead = _safe_mean(dead_counts)
            total_n = total_neurons if total_neurons else 1

            results.append(ActivationStats(
                layer_idx=layer_idx,
                mean=_safe_mean(means),
                std=_safe_mean(stds),
                max_abs=_safe_mean(max_abss),
                dead_neuron_count=int(round(avg_dead)),
                dead_neuron_fraction=avg_dead / total_n,
            ))
        return results

    def remove_hooks(self) -> None:
        """Remove all attached hooks."""
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
        self._capturing = False

    def reset(self) -> None:
        """Clear all captured data."""
        self._activation_records.clear()
        self._gradient_records.clear()
        self._current_activations.clear()
        self._current_gradients.clear()


def _safe_mean(values: list) -> float:
    """Compute mean, returning 0.0 for empty lists. Handles NaN gracefully."""
    if not values:
        return 0.0
    total = sum(v for v in values if not math.isnan(v))
    count = sum(1 for v in values if not math.isnan(v))
    return total / count if count > 0 else 0.0
