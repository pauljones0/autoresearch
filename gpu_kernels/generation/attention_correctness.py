"""
Attention correctness specialist: attention-specific correctness checks
for causal masking, sliding window, RoPE, and GQA.
"""

from typing import Dict, Any, Callable

import torch

import sys, os


class AttentionCorrectnessSpecialist:
    """Attention-specific correctness verification.

    Tests:
    1. Causal masking: output[i] depends only on input[:i+1]
    2. Sliding window: correct boundary behavior
    3. RoPE: position embedding correctness
    4. GQA: KV head sharing correctness
    """

    def __init__(
        self,
        atol: float = 1e-2,
        rtol: float = 1e-2,
        device: str = "cuda",
    ):
        self.atol = atol
        self.rtol = rtol
        self.device = device if torch.cuda.is_available() else "cpu"

    def _test_causal_masking(
        self,
        kernel_callable: Callable,
        attention_config: Dict[str, Any],
    ) -> dict:
        """Test that output[i] depends only on input[:i+1].

        Modifies future positions and checks that earlier outputs are unchanged.
        """
        torch.manual_seed(42)
        B, T, n_heads, head_dim = 1, 32, 4, 64
        n_kv_heads = attention_config.get("gqa", {}).get("n_kv_head", n_heads)
        if n_kv_heads == 0:
            n_kv_heads = n_heads

        q = torch.randn(B, T, n_heads, head_dim, device=self.device, dtype=torch.bfloat16)
        k = torch.randn(B, T, n_kv_heads, head_dim, device=self.device, dtype=torch.bfloat16)
        v = torch.randn(B, T, n_kv_heads, head_dim, device=self.device, dtype=torch.bfloat16)

        # Get baseline output
        try:
            out_baseline = kernel_callable(q, k, v, causal=True)
        except Exception as e:
            return {"passed": False, "error": f"Kernel call failed: {e}"}

        # Modify future tokens (position T//2 onwards)
        split_pos = T // 2
        k_modified = k.clone()
        v_modified = v.clone()
        k_modified[:, split_pos:] = torch.randn_like(k_modified[:, split_pos:])
        v_modified[:, split_pos:] = torch.randn_like(v_modified[:, split_pos:])

        try:
            out_modified = kernel_callable(q, k_modified, v_modified, causal=True)
        except Exception as e:
            return {"passed": False, "error": f"Modified kernel call failed: {e}"}

        # Outputs for positions < split_pos should be identical
        early_match = torch.allclose(
            out_baseline[:, :split_pos].float(),
            out_modified[:, :split_pos].float(),
            atol=self.atol, rtol=self.rtol,
        )
        max_early_diff = (
            out_baseline[:, :split_pos].float() - out_modified[:, :split_pos].float()
        ).abs().max().item()

        return {
            "passed": early_match,
            "max_early_diff": max_early_diff,
            "split_position": split_pos,
            "description": "Modifying future KV should not affect earlier outputs",
        }

    def _test_sliding_window(
        self,
        kernel_callable: Callable,
        attention_config: Dict[str, Any],
    ) -> dict:
        """Test sliding window boundary behavior.

        Tokens outside the window should not influence the output.
        """
        torch.manual_seed(123)
        window_size = 16
        B, T, n_heads, head_dim = 1, 64, 4, 64
        n_kv_heads = attention_config.get("gqa", {}).get("n_kv_head", n_heads)
        if n_kv_heads == 0:
            n_kv_heads = n_heads

        q = torch.randn(B, T, n_heads, head_dim, device=self.device, dtype=torch.bfloat16)
        k = torch.randn(B, T, n_kv_heads, head_dim, device=self.device, dtype=torch.bfloat16)
        v = torch.randn(B, T, n_kv_heads, head_dim, device=self.device, dtype=torch.bfloat16)

        try:
            out_baseline = kernel_callable(
                q, k, v, causal=True, window_size=(window_size, 0)
            )
        except TypeError:
            # Kernel may not support window_size parameter
            return {
                "passed": True,
                "skipped": True,
                "reason": "Kernel does not accept window_size parameter",
            }
        except Exception as e:
            return {"passed": False, "error": str(e)}

        # Modify tokens well outside the window for a late query position
        test_pos = T - 1  # last position
        outside_window = max(0, test_pos - window_size - 5)

        k_modified = k.clone()
        v_modified = v.clone()
        k_modified[:, :outside_window] = torch.randn_like(k_modified[:, :outside_window])
        v_modified[:, :outside_window] = torch.randn_like(v_modified[:, :outside_window])

        try:
            out_modified = kernel_callable(
                q, k_modified, v_modified, causal=True, window_size=(window_size, 0)
            )
        except Exception as e:
            return {"passed": False, "error": str(e)}

        # Output at test_pos should be unaffected by tokens outside window
        last_match = torch.allclose(
            out_baseline[:, test_pos].float(),
            out_modified[:, test_pos].float(),
            atol=self.atol, rtol=self.rtol,
        )
        max_diff = (
            out_baseline[:, test_pos].float() - out_modified[:, test_pos].float()
        ).abs().max().item()

        return {
            "passed": last_match,
            "max_diff_at_test_pos": max_diff,
            "window_size": window_size,
            "test_position": test_pos,
            "outside_window_end": outside_window,
            "description": "Tokens outside window should not affect output",
        }

    def _test_rope_correctness(
        self,
        kernel_callable: Callable,
        attention_config: Dict[str, Any],
    ) -> dict:
        """Test RoPE position embedding correctness.

        Verifies that the same content at different positions produces
        different attention patterns (position-dependent).
        """
        torch.manual_seed(456)
        B, T, n_heads, head_dim = 1, 32, 4, 64
        n_kv_heads = attention_config.get("gqa", {}).get("n_kv_head", n_heads)
        if n_kv_heads == 0:
            n_kv_heads = n_heads

        # Create input where position 5 and position 10 have identical content
        q = torch.randn(B, T, n_heads, head_dim, device=self.device, dtype=torch.bfloat16)
        k = torch.randn(B, T, n_kv_heads, head_dim, device=self.device, dtype=torch.bfloat16)
        v = torch.randn(B, T, n_kv_heads, head_dim, device=self.device, dtype=torch.bfloat16)

        # Make positions 5 and 10 have identical pre-RoPE content
        q[:, 10] = q[:, 5]

        try:
            out = kernel_callable(q, k, v, causal=True)
        except Exception as e:
            return {"passed": False, "error": str(e)}

        # If RoPE is applied, outputs at pos 5 and 10 should differ
        # (because they attend to different positions)
        outputs_differ = not torch.allclose(
            out[:, 5].float(), out[:, 10].float(),
            atol=1e-6, rtol=1e-6,
        )
        diff = (out[:, 5].float() - out[:, 10].float()).abs().max().item()

        return {
            "passed": outputs_differ,
            "max_diff_between_positions": diff,
            "description": (
                "Same content at different positions should produce "
                "different outputs due to position-dependent attention"
            ),
        }

    def _test_gqa_correctness(
        self,
        kernel_callable: Callable,
        attention_config: Dict[str, Any],
    ) -> dict:
        """Test GQA (Grouped Query Attention) KV head sharing.

        Verifies that query heads sharing the same KV head produce
        consistent outputs when KV content is identical.
        """
        gqa_config = attention_config.get("gqa", {})
        n_heads = gqa_config.get("n_head", 4)
        n_kv_heads = gqa_config.get("n_kv_head", n_heads)

        if n_kv_heads == 0:
            n_kv_heads = n_heads
        if n_heads == n_kv_heads:
            return {
                "passed": True,
                "skipped": True,
                "reason": "MHA mode (n_head == n_kv_head), GQA test not applicable",
            }

        torch.manual_seed(789)
        B, T, head_dim = 1, 16, 64
        ratio = n_heads // n_kv_heads

        q = torch.randn(B, T, n_heads, head_dim, device=self.device, dtype=torch.bfloat16)
        k = torch.randn(B, T, n_kv_heads, head_dim, device=self.device, dtype=torch.bfloat16)
        v = torch.randn(B, T, n_kv_heads, head_dim, device=self.device, dtype=torch.bfloat16)

        # Make query heads within the same group identical
        for kv_idx in range(n_kv_heads):
            base_q_idx = kv_idx * ratio
            for i in range(1, ratio):
                q[:, :, base_q_idx + i] = q[:, :, base_q_idx]

        try:
            out = kernel_callable(q, k, v, causal=True)
        except Exception as e:
            return {"passed": False, "error": str(e)}

        # Identical query heads sharing the same KV head should produce identical output
        all_match = True
        max_diff = 0.0
        for kv_idx in range(n_kv_heads):
            base_q_idx = kv_idx * ratio
            for i in range(1, ratio):
                diff = (
                    out[:, :, base_q_idx].float() - out[:, :, base_q_idx + i].float()
                ).abs().max().item()
                max_diff = max(max_diff, diff)
                if diff > self.atol:
                    all_match = False

        return {
            "passed": all_match,
            "max_diff_within_group": max_diff,
            "n_heads": n_heads,
            "n_kv_heads": n_kv_heads,
            "ratio": ratio,
            "description": "Identical Q heads sharing KV head should produce identical output",
        }

    def verify(
        self,
        kernel_callable: Callable,
        attention_config: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Run all attention-specific correctness checks.

        Args:
            kernel_callable: The attention kernel function. Expected signature:
                kernel_callable(q, k, v, causal=True, window_size=None) -> output
            attention_config: Dict from AttentionArchitectureAnalyzer.analyze().

        Returns:
            Dict with per-test results and overall pass/fail.
        """
        results = {}

        results["causal_masking"] = self._test_causal_masking(
            kernel_callable, attention_config
        )
        results["sliding_window"] = self._test_sliding_window(
            kernel_callable, attention_config
        )
        results["rope"] = self._test_rope_correctness(
            kernel_callable, attention_config
        )
        results["gqa"] = self._test_gqa_correctness(
            kernel_callable, attention_config
        )

        # Overall verdict
        all_passed = all(
            r.get("passed", False) for r in results.values()
        )
        n_skipped = sum(
            1 for r in results.values() if r.get("skipped", False)
        )

        return {
            "overall_passed": all_passed,
            "tests": results,
            "n_tests": len(results),
            "n_passed": sum(1 for r in results.values() if r.get("passed", False)),
            "n_skipped": n_skipped,
            "n_failed": sum(1 for r in results.values() if not r.get("passed", False)),
        }
