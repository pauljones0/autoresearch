"""
Optimizer analyzer: decomposes MuonAdamW.step() into memory operations
and computes fusion opportunities.
"""

import re
from typing import Dict, Any

import sys, os


class OptimizerAnalyzer:
    """Decompose MuonAdamW.step() into memory operations and compute
    theoretical minimum memory traffic vs actual."""

    def _parse_adamw_ops(self, source: str) -> list:
        """Extract AdamW memory operations from source."""
        ops = []

        # AdamW step operations (from adamw_step_fused)
        adamw_ops = [
            {"op": "read_param", "desc": "Read parameter tensor p", "rw": "read"},
            {"op": "read_grad", "desc": "Read gradient tensor", "rw": "read"},
            {"op": "read_exp_avg", "desc": "Read first moment (exp_avg)", "rw": "read"},
            {"op": "read_exp_avg_sq", "desc": "Read second moment (exp_avg_sq)", "rw": "read"},
            {"op": "write_param", "desc": "Write updated parameter (weight decay + update)", "rw": "write"},
            {"op": "write_exp_avg", "desc": "Write updated first moment (lerp)", "rw": "write"},
            {"op": "write_exp_avg_sq", "desc": "Write updated second moment (lerp)", "rw": "write"},
        ]
        if "adamw_step_fused" in source or "exp_avg" in source:
            ops.extend(adamw_ops)

        return ops

    def _parse_muon_ops(self, source: str) -> list:
        """Extract Muon memory operations from source."""
        ops = []

        muon_ops = [
            {"op": "read_stacked_grads", "desc": "Read stacked gradient tensors", "rw": "read"},
            {"op": "read_stacked_params", "desc": "Read stacked parameter tensors", "rw": "read"},
            {"op": "read_momentum_buffer", "desc": "Read Nesterov momentum buffer", "rw": "read"},
            {"op": "read_second_momentum", "desc": "Read NorMuon second momentum buffer", "rw": "read"},
            {"op": "write_stacked_params", "desc": "Write updated parameters", "rw": "write"},
            {"op": "write_momentum_buffer", "desc": "Write updated momentum buffer", "rw": "write"},
            {"op": "write_second_momentum", "desc": "Write updated second momentum", "rw": "write"},
            {"op": "polar_express_matmuls", "desc": "Polar express orthogonalization (compute-bound)", "rw": "compute"},
        ]
        if "muon_step_fused" in source or "momentum_buffer" in source:
            ops.extend(muon_ops)

        return ops

    def _compute_traffic(self, n_params: int, dtype_bytes: int = 2) -> dict:
        """Compute theoretical vs actual memory traffic for AdamW.

        For a single AdamW step on n_params parameters (bf16):
        - Actual: reads p, grad, m, v; writes p, m, v = 7 * n_params * dtype_bytes
        - Theoretical minimum (fused): read grad, m, v; write p, m, v = 6 * n_params * dtype_bytes
          (p can be read-modify-write in registers if fused)
        - Best case fused: read grad; read+write p, m, v in single pass
        """
        actual_reads = 4  # p, grad, exp_avg, exp_avg_sq
        actual_writes = 3  # p, exp_avg, exp_avg_sq
        actual_total = (actual_reads + actual_writes) * n_params * dtype_bytes

        # Fused: single pass reads all, writes all
        fused_reads = 4  # still need to read all
        fused_writes = 3  # still need to write all
        fused_total = (fused_reads + fused_writes) * n_params * dtype_bytes

        # But unfused PyTorch launches separate kernels for each op:
        # mul_(1-lr*wd) -> read+write p
        # lerp_(grad, 1-beta1) -> read+write exp_avg, read grad
        # lerp_(grad.square(), 1-beta2) -> read+write exp_avg_sq, read grad
        # bias correction + update -> read exp_avg, exp_avg_sq, write p
        unfused_reads = 4 + 2 + 2 + 2  # 10 tensor reads across 4 kernel launches
        unfused_writes = 1 + 1 + 1 + 1  # 4 tensor writes
        unfused_total = (unfused_reads + unfused_writes) * n_params * dtype_bytes

        return {
            "fused_bytes": fused_total,
            "unfused_bytes": unfused_total,
            "savings_ratio": unfused_total / fused_total if fused_total > 0 else 0,
            "fused_reads": fused_reads,
            "fused_writes": fused_writes,
            "unfused_reads": unfused_reads,
            "unfused_writes": unfused_writes,
        }

    def analyze(self, train_source: str) -> Dict[str, Any]:
        """Decompose MuonAdamW.step() into memory operations and compute
        fusion opportunities.

        Args:
            train_source: Source code of train.py.

        Returns:
            Dict with keys:
                - adamw_ops: List of AdamW memory operations
                - muon_ops: List of Muon memory operations
                - adamw_traffic: Memory traffic analysis for AdamW
                - muon_traffic: Memory traffic analysis for Muon
                - fusion_opportunities: List of recommended fusions
                - summary: Human-readable summary
        """
        adamw_ops = self._parse_adamw_ops(train_source)
        muon_ops = self._parse_muon_ops(train_source)

        # Extract param group info
        param_groups = []
        if "kind='adamw'" in train_source or "kind=\'adamw\'" in train_source:
            param_groups.append("adamw")
        if "kind='muon'" in train_source or "kind=\'muon\'" in train_source:
            param_groups.append("muon")

        # Estimate param counts (from train.py architecture)
        # Model dim 512, 8 layers -> ~26M params, most are matrix params (Muon)
        # AdamW handles: embeddings, lm_head, scalars
        # Muon handles: transformer matrix params (bulk of parameters)
        adamw_param_estimate = 32768 * 512 + 32768 * 512  # wte + lm_head ~33M
        muon_param_estimate = 8 * (512 * 2048 + 2048 * 512 + 512 * 512 * 3)  # ~40M

        adamw_traffic = self._compute_traffic(adamw_param_estimate)
        muon_traffic = self._compute_traffic(muon_param_estimate)

        fusion_opportunities = []

        # AdamW fusion: weight_decay + momentum + bias_correction + update
        adamw_read_ops = [op for op in adamw_ops if op["rw"] == "read"]
        adamw_write_ops = [op for op in adamw_ops if op["rw"] == "write"]
        fusion_opportunities.append({
            "name": "adamw_fused_update",
            "description": (
                "Fuse weight decay, momentum update, bias correction, and "
                "parameter update into a single Triton kernel pass. "
                "Eliminates redundant reads/writes between separate PyTorch kernels."
            ),
            "ops_fused": [op["op"] for op in adamw_ops],
            "estimated_traffic_reduction": adamw_traffic["savings_ratio"],
            "complexity": "medium",
            "already_compiled": "adamw_step_fused" in train_source,
        })

        # Muon is harder to fuse due to polar express matmuls
        fusion_opportunities.append({
            "name": "muon_post_orthogonalization_fused",
            "description": (
                "Fuse NorMuon variance reduction + cautious weight decay + "
                "parameter update after polar express. The orthogonalization "
                "itself involves matmuls and is better left to cuBLAS/torch.compile."
            ),
            "ops_fused": ["normalization", "weight_decay", "param_update"],
            "estimated_traffic_reduction": 1.5,
            "complexity": "high",
            "already_compiled": "muon_step_fused" in train_source,
        })

        # Note: both are already torch.compiled in train.py
        already_compiled = (
            "@torch.compile" in train_source
            and "adamw_step_fused" in train_source
        )

        summary = (
            f"MuonAdamW uses {len(adamw_ops)} AdamW memory ops and "
            f"{len(muon_ops)} Muon memory ops. "
            f"AdamW traffic savings from fusion: {adamw_traffic['savings_ratio']:.1f}x. "
            f"Note: both adamw_step_fused and muon_step_fused are already "
            f"@torch.compile'd, so fusion gains may be limited to cases where "
            f"the compiler misses optimization opportunities."
        )

        return {
            "adamw_ops": adamw_ops,
            "muon_ops": muon_ops,
            "adamw_traffic": adamw_traffic,
            "muon_traffic": muon_traffic,
            "fusion_opportunities": fusion_opportunities,
            "param_groups_detected": param_groups,
            "already_torch_compiled": already_compiled,
            "summary": summary,
        }
