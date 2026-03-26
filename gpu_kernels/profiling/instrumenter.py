"""
GPUProfilingInstrumenter: captures per-operation GPU timing and memory
using torch.profiler. Extends the DiagnosticsInstrumenter pattern for
GPU kernel optimization profiling.

Usage:
    instrumenter = GPUProfilingInstrumenter()
    profiles = instrumenter.profile_step(model, batch)
"""

import re
from typing import Optional

import torch
import torch.nn as nn
from torch.profiler import profile, ProfilerActivity

from ..schemas import OperationProfile


# Map CUDA kernel names to logical call sites in train.py
_CALL_SITE_PATTERNS = [
    (re.compile(r"rms_norm|layer_norm", re.I), "norm (F.rms_norm)"),
    (re.compile(r"addmm|linear|gemm|cublas", re.I), "nn.Linear (matmul)"),
    (re.compile(r"flash_attn|fmha|sdpa", re.I), "CausalSelfAttention (flash_attn)"),
    (re.compile(r"rotary|rope", re.I), "apply_rotary_emb"),
    (re.compile(r"relu", re.I), "MLP (F.relu)"),
    (re.compile(r"square|pow", re.I), "MLP (relu squared)"),
    (re.compile(r"cross_entropy|softmax.*loss|log_softmax", re.I), "GPT.forward (cross_entropy)"),
    (re.compile(r"tanh", re.I), "GPT.forward (softcap tanh)"),
    (re.compile(r"embedding|index_select", re.I), "Embedding lookup"),
    (re.compile(r"cat|concat", re.I), "torch.cat"),
    (re.compile(r"sigmoid", re.I), "CausalSelfAttention (ve_gate sigmoid)"),
    (re.compile(r"copy|contiguous", re.I), "memory layout"),
    (re.compile(r"mul_?$|multiply", re.I), "elementwise mul"),
    (re.compile(r"add_?$", re.I), "elementwise add"),
]


def _map_call_site(op_name: str) -> str:
    """Map a CUDA op name to a logical call site in train.py."""
    for pattern, site in _CALL_SITE_PATTERNS:
        if pattern.search(op_name):
            return site
    return ""


class GPUProfilingInstrumenter:
    """Profiles a single training step to capture per-operation GPU timing
    and memory statistics using torch.profiler."""

    def __init__(self, capture_every_n_steps: int = 50):
        self.capture_every_n_steps = capture_every_n_steps
        self._step_count = 0

    def should_capture(self, step: int) -> bool:
        """Check if this step should be profiled."""
        return step % self.capture_every_n_steps == 0

    def profile_step(
        self,
        model: nn.Module,
        batch: tuple,
        autocast_dtype: torch.dtype = torch.bfloat16,
    ) -> list[OperationProfile]:
        """Run a single forward+backward pass under the CUDA profiler.

        Args:
            model: The GPT model (compiled or raw).
            batch: Tuple of (input_ids, targets) tensors.
            autocast_dtype: dtype for autocast context.

        Returns:
            List of OperationProfile with per-op timing and memory data.
        """
        x, y = batch[0], batch[1]
        device = x.device

        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True,
            profile_memory=True,
            with_stack=False,
        ) as prof:
            with torch.amp.autocast(device_type="cuda", dtype=autocast_dtype):
                loss = model(x, y)
            loss.backward()

        return self._extract_profiles(prof)

    def _extract_profiles(self, prof) -> list[OperationProfile]:
        """Extract OperationProfile entries from profiler key_averages."""
        profiles = []
        key_averages = prof.key_averages()

        for event in key_averages:
            # Skip events with negligible GPU time
            gpu_time = event.cuda_time_total
            if gpu_time <= 0:
                continue

            op_name = event.key
            input_shapes = []
            if hasattr(event, "input_shapes") and event.input_shapes:
                input_shapes = [list(s) for s in event.input_shapes if s]

            mem_read = 0
            mem_write = 0
            if hasattr(event, "cuda_memory_usage"):
                # cuda_memory_usage is net allocation; approximate read/write
                mem_usage = event.cuda_memory_usage or 0
                if mem_usage > 0:
                    mem_write = mem_usage
                else:
                    mem_read = abs(mem_usage)

            # Better memory estimation from self CUDA time and flops
            if hasattr(event, "flops") and event.flops and event.flops > 0:
                # For compute ops, estimate bytes from tensor shapes
                pass

            call_site = _map_call_site(op_name)

            op_profile = OperationProfile(
                op_name=op_name,
                gpu_time_us=gpu_time,
                cpu_time_us=event.cpu_time_total,
                memory_read_bytes=mem_read,
                memory_write_bytes=mem_write,
                call_stack=call_site,
                input_shapes=input_shapes,
            )
            profiles.append(op_profile)

        # Sort by GPU time descending (hottest ops first)
        profiles.sort(key=lambda p: p.gpu_time_us, reverse=True)
        return profiles
