"""
AttentionAnalyzer: computes attention entropy and collapse scores per head.

Since the model uses Flash Attention 3 (which does not return attention weights),
this module manually computes Q*K^T on a small sample batch for diagnostic
purposes only — never during training.

Usage:
    analyzer = AttentionAnalyzer()
    stats = analyzer.analyze(model, sample_batch_x)
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..schemas import AttentionStats


def _rms_norm(x: torch.Tensor) -> torch.Tensor:
    """RMS normalization matching train.py's norm()."""
    return F.rms_norm(x, (x.size(-1),))


def _apply_rotary_emb(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    """Apply rotary position embeddings, matching train.py."""
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3)


class AttentionAnalyzer:
    """Computes attention entropy and collapse scores by manually
    computing attention weights from Q and K projections."""

    def __init__(self, max_seq_len: int = 512, max_batch: int = 2):
        """
        Args:
            max_seq_len: Truncate sequences to this length to limit memory.
            max_batch: Maximum batch elements to analyze.
        """
        self.max_seq_len = max_seq_len
        self.max_batch = max_batch

    @torch.no_grad()
    def analyze(
        self, model: nn.Module, sample_x: torch.Tensor
    ) -> list[dict]:
        """Compute per-head attention entropy and collapse scores.

        Args:
            model: The GPT model (compiled or raw).
            sample_x: Input token IDs, shape (B, T).

        Returns:
            List of AttentionStats dicts (one per layer per head).
        """
        raw_model = model
        if hasattr(model, "_orig_mod"):
            raw_model = model._orig_mod

        # Architecture version check — fail loudly on mismatch
        required_attrs = ["config", "cos", "sin", "transformer", "resid_lambdas", "window_sizes"]
        missing = [a for a in required_attrs if not hasattr(raw_model, a)]
        if missing:
            raise AttributeError(
                f"Model is missing required attributes for attention analysis: {missing}. "
                f"Ensure the model matches the expected GPT architecture from train.py."
            )

        # Truncate to keep memory reasonable
        B = min(sample_x.size(0), self.max_batch)
        T = min(sample_x.size(1), self.max_seq_len)
        x_input = sample_x[:B, :T]

        device = x_input.device
        config = raw_model.config

        # Get embeddings and rotary
        cos = raw_model.cos[:, :T]
        sin = raw_model.sin[:, :T]

        # Forward through embedding + norm
        x = raw_model.transformer.wte(x_input)
        x = _rms_norm(x)
        x0 = x

        results = []

        for layer_idx, block in enumerate(raw_model.transformer.h):
            # Apply residual scaling as in GPT.forward
            x = raw_model.resid_lambdas[layer_idx] * x + raw_model.x0_lambdas[layer_idx] * x0

            attn = block.attn
            x_normed = _rms_norm(x)

            # Compute Q and K
            q = attn.c_q(x_normed).view(B, T, attn.n_head, attn.head_dim)
            k = attn.c_k(x_normed).view(B, T, attn.n_kv_head, attn.head_dim)

            # Apply rotary embeddings
            q = _apply_rotary_emb(q, cos, sin)
            k = _apply_rotary_emb(k, cos, sin)

            # Apply QK norm as in the model
            q = _rms_norm(q)
            k = _rms_norm(k)

            # Handle GQA: expand K to match Q heads
            if attn.n_kv_head < attn.n_head:
                repeats = attn.n_head // attn.n_kv_head
                k = k.unsqueeze(3).expand(
                    B, T, attn.n_kv_head, repeats, attn.head_dim
                ).reshape(B, T, attn.n_head, attn.head_dim)

            # Compute attention scores: (B, n_head, T, T)
            # q, k are (B, T, n_head, head_dim) -> transpose to (B, n_head, T, head_dim)
            q_t = q.permute(0, 2, 1, 3).float()
            k_t = k.permute(0, 2, 1, 3).float()
            scores = torch.matmul(q_t, k_t.transpose(-2, -1))

            # Causal mask
            causal_mask = torch.triu(
                torch.ones(T, T, device=device, dtype=torch.bool), diagonal=1
            )
            scores.masked_fill_(causal_mask.unsqueeze(0).unsqueeze(0), float("-inf"))

            # Apply window mask if applicable
            window_size = raw_model.window_sizes[layer_idx][0]
            if window_size < T:
                # Mask positions beyond the window
                row_idx = torch.arange(T, device=device).unsqueeze(1)
                col_idx = torch.arange(T, device=device).unsqueeze(0)
                window_mask = (row_idx - col_idx) >= window_size
                scores.masked_fill_(
                    window_mask.unsqueeze(0).unsqueeze(0), float("-inf")
                )

            # Softmax to get attention weights
            attn_weights = torch.softmax(scores, dim=-1)  # (B, n_head, T, T)

            # Compute per-head entropy and collapse score
            # Average over batch and query positions
            for head_idx in range(attn.n_head):
                w = attn_weights[:, head_idx, :, :]  # (B, T, T)

                # Entropy: H = -sum(p * log(p)), averaged over queries and batch
                # Clamp to avoid log(0)
                log_w = torch.log(w.clamp(min=1e-10))
                entropy_per_query = -(w * log_w).sum(dim=-1)  # (B, T)
                avg_entropy = entropy_per_query.mean().item()

                # Max entropy for a causal attention head varies by position:
                # at position t, max entropy = log(t+1) (uniform over t+1 tokens)
                # But with windowing, max entropy = log(min(t+1, window_size))
                # We compute the average max entropy across positions
                positions = torch.arange(1, T + 1, device=device, dtype=torch.float32)
                if window_size < T:
                    effective_positions = torch.clamp(positions, max=float(window_size))
                else:
                    effective_positions = positions
                max_entropy_per_pos = torch.log(effective_positions)
                avg_max_entropy = max_entropy_per_pos.mean().item()

                # Collapse score: 1 - (H_actual / H_max)
                # High collapse = attention is peaky (low entropy relative to max)
                # Note: original spec says "near-uniform = low information = collapse"
                # but that would be HIGH entropy. We interpret collapse_score as
                # measuring deviation from uniform, so collapsed = peaky attention.
                if avg_max_entropy > 0:
                    collapse_score = 1.0 - (avg_entropy / avg_max_entropy)
                else:
                    collapse_score = 0.0
                collapse_score = max(0.0, min(1.0, collapse_score))

                # Max attention weight (averaged over batch and queries)
                max_weight = w.max(dim=-1).values.mean().item()

                results.append(vars(AttentionStats(
                    layer_idx=layer_idx,
                    head_idx=head_idx,
                    entropy=avg_entropy,
                    collapse_score=collapse_score,
                    max_attention_weight=max_weight,
                )))

            # Continue forward pass for next layer: run full block
            # We need v and value embeddings too
            v = attn.c_v(x_normed).view(B, T, attn.n_kv_head, attn.head_dim)

            # Value residual
            ve = None
            if str(layer_idx) in raw_model.value_embeds:
                ve = raw_model.value_embeds[str(layer_idx)](x_input)
                ve = ve.view(B, T, attn.n_kv_head, attn.head_dim)
                gate = 2 * torch.sigmoid(
                    attn.ve_gate(x_normed[..., :attn.ve_gate_channels])
                )
                v = v + gate.unsqueeze(-1) * ve

            # Use manual attention (no flash attention) for the forward pass
            if attn.n_kv_head < attn.n_head:
                repeats = attn.n_head // attn.n_kv_head
                v_expanded = v.unsqueeze(3).expand(
                    B, T, attn.n_kv_head, repeats, attn.head_dim
                ).reshape(B, T, attn.n_head, attn.head_dim)
            else:
                v_expanded = v

            v_t = v_expanded.permute(0, 2, 1, 3).float()
            attn_out = torch.matmul(attn_weights, v_t)  # (B, n_head, T, head_dim)
            attn_out = attn_out.permute(0, 2, 1, 3).contiguous()
            attn_out = attn_out.reshape(B, T, -1).to(x.dtype)
            attn_out = attn.c_proj(attn_out)

            x = x + attn_out

            # MLP forward
            mlp_in = _rms_norm(x)
            mlp_out = block.mlp.c_fc(mlp_in)
            mlp_out = F.relu(mlp_out).square()
            mlp_out = block.mlp.c_proj(mlp_out)
            x = x + mlp_out

        return results
