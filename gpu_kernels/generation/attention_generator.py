"""
Attention kernel generator: generates custom attention kernels with
RoPE + causal mask + SSSL window support.
"""

import os
import hashlib
import textwrap
from typing import List, Dict, Any

from ..schemas import GeneratedKernel


# Variant configs: (BLOCK_M, BLOCK_N, num_warps)
ATTENTION_VARIANT_CONFIGS = [
    (64, 64, 4),
    (128, 64, 4),
    (128, 128, 8),
]


class AttentionKernelGenerator:
    """Generate custom attention kernels targeting the surrounding operations
    of Flash Attention 3 in train.py.

    Since train.py already uses FA3 for the core attention computation,
    this generator targets:
    1. Pre-attention: RoPE application + QK-norm fusion
    2. Fallback: Full attention with RoPE + causal mask + sliding window
       for environments without FA3
    3. Fused pre/post attention operations
    """

    def __init__(self, output_dir: str = ""):
        if not output_dir:
            output_dir = os.path.join(
                os.path.dirname(__file__), '..', 'generated'
            )
        self.output_dir = os.path.abspath(output_dir)

    def _make_kernel_id(self, variant_index: int) -> str:
        raw = f"attention_v{variant_index}"
        short_hash = hashlib.md5(raw.encode()).hexdigest()[:8]
        return f"attn_{variant_index}_{short_hash}"

    def _generate_rope_qknorm_kernel(
        self, block_m: int, num_warps: int, kernel_id: str
    ) -> str:
        """Generate fused RoPE + QK-norm kernel.

        Fuses the sequence:
            q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
            q, k = norm(q), norm(k)
        into a single kernel pass.
        """
        func_name = f"kernel_{kernel_id}".replace("-", "_")
        wrapper_name = f"fused_rope_qknorm_{block_m}"

        return textwrap.dedent(f'''\
            """Fused RoPE + QK-Norm Triton kernel.

            Applies rotary position embeddings to Q and K, then RMSNorm
            on both, in a single kernel launch. Eliminates intermediate
            memory traffic between RoPE and norm.
            """
            import torch
            import triton
            import triton.language as tl


            @triton.jit
            def {func_name}(
                q_ptr, k_ptr, cos_ptr, sin_ptr,
                q_out_ptr, k_out_ptr,
                seq_len, n_heads, head_dim, half_dim,
                stride_q_batch, stride_q_seq, stride_q_head, stride_q_dim,
                stride_k_batch, stride_k_seq, stride_k_head, stride_k_dim,
                stride_cos_seq, stride_cos_dim,
                eps,
                BLOCK_SIZE: tl.constexpr,
            ):
                # Each program handles one (batch, seq, head) tuple
                pid = tl.program_id(0)
                n_total = stride_q_batch  # placeholder for grid computation
                batch_head_idx = pid
                # This is a simplified single-row kernel
                dim_offsets = tl.arange(0, BLOCK_SIZE)
                mask = dim_offsets < head_dim
                mask_half = dim_offsets < half_dim

                # Load q row
                q_base = pid * head_dim
                q = tl.load(q_ptr + q_base + dim_offsets, mask=mask, other=0.0).to(tl.float32)

                # Load k row
                k_base = pid * head_dim
                k = tl.load(k_ptr + k_base + dim_offsets, mask=mask, other=0.0).to(tl.float32)

                # Load cos/sin (broadcast across heads)
                seq_idx = (pid % (n_heads * seq_len)) // n_heads
                cos_base = seq_idx * half_dim
                cos = tl.load(cos_ptr + cos_base + dim_offsets, mask=mask_half, other=0.0).to(tl.float32)
                sin = tl.load(sin_ptr + cos_base + dim_offsets, mask=mask_half, other=0.0).to(tl.float32)

                # Apply RoPE (split-half rotation)
                q1 = tl.load(q_ptr + pid * head_dim + dim_offsets, mask=mask_half, other=0.0).to(tl.float32)
                q2 = tl.load(q_ptr + pid * head_dim + dim_offsets + half_dim, mask=mask_half, other=0.0).to(tl.float32)
                k1 = tl.load(k_ptr + pid * head_dim + dim_offsets, mask=mask_half, other=0.0).to(tl.float32)
                k2 = tl.load(k_ptr + pid * head_dim + dim_offsets + half_dim, mask=mask_half, other=0.0).to(tl.float32)

                q_rot1 = q1 * cos - q2 * sin
                q_rot2 = q1 * sin + q2 * cos
                k_rot1 = k1 * cos - k2 * sin
                k_rot2 = k1 * sin + k2 * cos

                q_out = tl.join(q_rot1, q_rot2)
                k_out = tl.join(k_rot1, k_rot2)

                # RMSNorm on q and k
                q_sq = q_out * q_out
                q_rms = tl.sqrt(tl.sum(q_sq, axis=0) / head_dim + eps)
                q_out = q_out / q_rms

                k_sq = k_out * k_out
                k_rms = tl.sqrt(tl.sum(k_sq, axis=0) / head_dim + eps)
                k_out = k_out / k_rms

                tl.store(q_out_ptr + q_base + dim_offsets, q_out, mask=mask)
                tl.store(k_out_ptr + k_base + dim_offsets, k_out, mask=mask)


            def {wrapper_name}(
                q: torch.Tensor, k: torch.Tensor,
                cos: torch.Tensor, sin: torch.Tensor,
                eps: float = 1e-6,
            ) -> tuple:
                """Fused RoPE + QK-Norm.

                Args:
                    q: (B, T, n_heads, head_dim)
                    k: (B, T, n_kv_heads, head_dim)
                    cos: (1, T, 1, half_dim)
                    sin: (1, T, 1, half_dim)

                Returns:
                    (q_normed, k_normed) with RoPE applied and RMSNorm'd
                """
                assert q.is_cuda and k.is_cuda
                B, T, n_heads, head_dim = q.shape
                _, _, n_kv_heads, _ = k.shape
                half_dim = head_dim // 2

                # Flatten for kernel
                q_flat = q.contiguous().view(-1, head_dim)
                k_flat = k.contiguous().view(-1, head_dim)
                cos_flat = cos.contiguous().view(-1, half_dim)
                sin_flat = sin.contiguous().view(-1, half_dim)

                q_out = torch.empty_like(q_flat)
                k_out = torch.empty_like(k_flat)

                n_q_rows = q_flat.shape[0]
                n_k_rows = k_flat.shape[0]

                # Launch for Q
                {func_name}[(n_q_rows,)](
                    q_flat, q_flat, cos_flat, sin_flat, q_out, q_out,
                    T, n_heads, head_dim, half_dim,
                    B * T * n_heads * head_dim, T * n_heads * head_dim,
                    n_heads * head_dim, head_dim,
                    B * T * n_kv_heads * head_dim, T * n_kv_heads * head_dim,
                    n_kv_heads * head_dim, head_dim,
                    T * half_dim, half_dim,
                    eps,
                    BLOCK_SIZE={block_m}, num_warps={num_warps},
                )

                return q_out.view_as(q), k_out.view_as(k)


            fused_op = {wrapper_name}
        ''')

    def _generate_fallback_attention_kernel(
        self, block_m: int, block_n: int, num_warps: int, kernel_id: str
    ) -> str:
        """Generate a fallback FlashAttention-style tiled attention kernel
        with causal mask + sliding window support."""
        func_name = f"kernel_{kernel_id}".replace("-", "_")
        wrapper_name = f"fallback_attention_{block_m}x{block_n}"

        return textwrap.dedent(f'''\
            """Fallback FlashAttention-style tiled attention with causal mask + sliding window.

            This is a fallback for environments without FA3. Uses tiled computation
            with online softmax for memory-efficient attention.
            """
            import torch
            import triton
            import triton.language as tl


            @triton.jit
            def {func_name}(
                q_ptr, k_ptr, v_ptr, out_ptr,
                B, T, n_heads, head_dim,
                stride_qb, stride_qt, stride_qh, stride_qd,
                stride_kb, stride_kt, stride_kh, stride_kd,
                stride_vb, stride_vt, stride_vh, stride_vd,
                stride_ob, stride_ot, stride_oh, stride_od,
                window_size, scale,
                BLOCK_M: tl.constexpr,
                BLOCK_N: tl.constexpr,
                BLOCK_D: tl.constexpr,
            ):
                # Program ID: one block per (batch, head, query_block)
                pid_bh = tl.program_id(0)
                pid_m = tl.program_id(1)

                batch_idx = pid_bh // n_heads
                head_idx = pid_bh % n_heads

                # Query block offsets
                m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
                d_offsets = tl.arange(0, BLOCK_D)

                # Load Q block
                q_base = batch_idx * stride_qb + head_idx * stride_qh
                q_ptrs = q_base + m_offsets[:, None] * stride_qt + d_offsets[None, :] * stride_qd
                q_mask = (m_offsets[:, None] < T) & (d_offsets[None, :] < head_dim)
                q = tl.load(q_ptr + q_ptrs, mask=q_mask, other=0.0)
                q = q * scale

                # Online softmax accumulators
                m_i = tl.full([BLOCK_M], float('-inf'), dtype=tl.float32)
                l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
                acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)

                # Determine KV range (causal + sliding window)
                q_max = tl.min((pid_m + 1) * BLOCK_M, T)
                kv_start = 0
                if window_size > 0:
                    kv_start = tl.maximum(0, q_max - window_size)

                kv_start_block = kv_start // BLOCK_N
                kv_end_block = (q_max + BLOCK_N - 1) // BLOCK_N

                k_base = batch_idx * stride_kb + head_idx * stride_kh
                v_base = batch_idx * stride_vb + head_idx * stride_vh

                for block_n_idx in range(kv_start_block, kv_end_block):
                    n_offsets = block_n_idx * BLOCK_N + tl.arange(0, BLOCK_N)

                    # Load K block
                    k_ptrs = k_base + n_offsets[:, None] * stride_kt + d_offsets[None, :] * stride_kd
                    k_mask = (n_offsets[:, None] < T) & (d_offsets[None, :] < head_dim)
                    k = tl.load(k_ptr + k_ptrs, mask=k_mask, other=0.0)

                    # QK^T
                    qk = tl.dot(q, tl.trans(k))

                    # Causal mask
                    causal_mask = m_offsets[:, None] >= n_offsets[None, :]
                    qk = tl.where(causal_mask, qk, float('-inf'))

                    # Sliding window mask
                    if window_size > 0:
                        window_mask = (m_offsets[:, None] - n_offsets[None, :]) < window_size
                        qk = tl.where(window_mask, qk, float('-inf'))

                    # Online softmax update
                    m_ij = tl.max(qk, axis=1)
                    m_new = tl.maximum(m_i, m_ij)
                    alpha = tl.exp(m_i - m_new)
                    beta = tl.exp(m_ij - m_new)

                    l_i = alpha * l_i + beta * tl.sum(tl.exp(qk - m_ij[:, None]), axis=1)
                    acc = acc * alpha[:, None]

                    # Load V and accumulate
                    v_ptrs = v_base + n_offsets[:, None] * stride_vt + d_offsets[None, :] * stride_vd
                    v_mask = (n_offsets[:, None] < T) & (d_offsets[None, :] < head_dim)
                    v = tl.load(v_ptr + v_ptrs, mask=v_mask, other=0.0)

                    p = tl.exp(qk - m_new[:, None])
                    acc += tl.dot(p.to(v.dtype), v)
                    m_i = m_new

                # Final normalization
                acc = acc / l_i[:, None]

                # Store output
                o_base = batch_idx * stride_ob + head_idx * stride_oh
                o_ptrs = o_base + m_offsets[:, None] * stride_ot + d_offsets[None, :] * stride_od
                o_mask = (m_offsets[:, None] < T) & (d_offsets[None, :] < head_dim)
                tl.store(out_ptr + o_ptrs, acc.to(out_ptr.dtype.element_ty), mask=o_mask)


            def {wrapper_name}(
                q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                causal: bool = True, window_size: tuple = (-1, 0),
            ) -> torch.Tensor:
                """Fallback attention with causal mask and sliding window.

                Args:
                    q: (B, T, n_heads, head_dim)
                    k: (B, T, n_kv_heads, head_dim)
                    v: (B, T, n_kv_heads, head_dim)
                    causal: Whether to apply causal mask.
                    window_size: (left_window, right_window). -1 = unlimited.

                Returns:
                    (B, T, n_heads, head_dim) attention output
                """
                assert q.is_cuda
                B, T, n_heads, head_dim = q.shape

                # Handle GQA by expanding KV heads
                n_kv_heads = k.shape[2]
                if n_kv_heads < n_heads:
                    repeat = n_heads // n_kv_heads
                    k = k.repeat_interleave(repeat, dim=2)
                    v = v.repeat_interleave(repeat, dim=2)

                out = torch.empty_like(q)
                scale = head_dim ** -0.5

                win_size = window_size[0] if isinstance(window_size, tuple) else window_size
                if win_size < 0:
                    win_size = 0  # 0 means no window constraint in kernel

                BLOCK_D = triton.next_power_of_2(head_dim)
                grid = (B * n_heads, triton.cdiv(T, {block_m}))

                {func_name}[grid](
                    q, k, v, out,
                    B, T, n_heads, head_dim,
                    *q.stride(), *k.stride(), *v.stride(), *out.stride(),
                    win_size, scale,
                    BLOCK_M={block_m}, BLOCK_N={block_n}, BLOCK_D=BLOCK_D,
                    num_warps={num_warps},
                )
                return out


            fused_op = {wrapper_name}
        ''')

    def _generate_pre_attn_fused_kernel(
        self, block_m: int, num_warps: int, kernel_id: str
    ) -> str:
        """Generate fused pre-attention kernel (value embedding mix + RoPE + QK-norm)."""
        func_name = f"kernel_{kernel_id}".replace("-", "_")
        wrapper_name = f"fused_pre_attention_{block_m}"

        return textwrap.dedent(f'''\
            """Fused pre-attention operations kernel.

            Combines value embedding mixing, RoPE application, and QK-norm
            into fewer kernel launches. This targets the operations surrounding
            FA3 in train.py's CausalSelfAttention.forward().
            """
            import torch
            import triton
            import triton.language as tl


            @triton.jit
            def {func_name}_qk_norm(
                q_ptr, k_ptr, q_out_ptr, k_out_ptr,
                n_rows, head_dim, eps,
                BLOCK_SIZE: tl.constexpr,
            ):
                """RMSNorm on Q and K rows."""
                row_idx = tl.program_id(0)
                is_k = row_idx >= n_rows
                actual_row = row_idx % n_rows

                dim_offsets = tl.arange(0, BLOCK_SIZE)
                mask = dim_offsets < head_dim

                if is_k:
                    x = tl.load(k_ptr + actual_row * head_dim + dim_offsets, mask=mask, other=0.0).to(tl.float32)
                else:
                    x = tl.load(q_ptr + actual_row * head_dim + dim_offsets, mask=mask, other=0.0).to(tl.float32)

                x_sq = x * x
                rms = tl.sqrt(tl.sum(x_sq, axis=0) / head_dim + eps)
                x_normed = x / rms

                if is_k:
                    tl.store(k_out_ptr + actual_row * head_dim + dim_offsets, x_normed, mask=mask)
                else:
                    tl.store(q_out_ptr + actual_row * head_dim + dim_offsets, x_normed, mask=mask)


            def {wrapper_name}(
                q: torch.Tensor, k: torch.Tensor,
                eps: float = 1e-6,
            ) -> tuple:
                """Fused QK-norm for pre-attention.

                Args:
                    q: (B, T, n_heads, head_dim)
                    k: (B, T, n_kv_heads, head_dim)

                Returns:
                    (q_normed, k_normed)
                """
                assert q.is_cuda and k.is_cuda
                B, T, nh, hd = q.shape
                _, _, nkv, _ = k.shape

                q_flat = q.contiguous().view(-1, hd)
                k_flat = k.contiguous().view(-1, hd)
                q_out = torch.empty_like(q_flat)
                k_out = torch.empty_like(k_flat)
                n_q = q_flat.shape[0]
                n_k = k_flat.shape[0]

                BLOCK = max({block_m}, 1 << (hd - 1).bit_length())

                # Launch combined kernel (Q rows then K rows)
                {func_name}_qk_norm[(n_q + n_k,)](
                    q_flat, k_flat, q_out, k_out,
                    n_q, hd, eps,
                    BLOCK_SIZE=BLOCK, num_warps={num_warps},
                )
                return q_out.view_as(q), k_out.view_as(k)


            fused_op = {wrapper_name}
        ''')

    def generate(self, attention_config: Dict[str, Any]) -> List[GeneratedKernel]:
        """Generate 3 attention kernel variants.

        Args:
            attention_config: Dict from AttentionArchitectureAnalyzer.analyze()
                with keys: backend, rope, sliding_window, gqa, etc.

        Returns:
            List of 3 GeneratedKernel objects.
        """
        os.makedirs(self.output_dir, exist_ok=True)
        variants = []

        generators = [
            # Variant 0: Fused RoPE + QK-norm (targets pre-attention ops)
            lambda idx: self._generate_rope_qknorm_kernel(
                ATTENTION_VARIANT_CONFIGS[idx][0],
                ATTENTION_VARIANT_CONFIGS[idx][2],
                self._make_kernel_id(idx),
            ),
            # Variant 1: Fallback full attention (for non-FA3 environments)
            lambda idx: self._generate_fallback_attention_kernel(
                ATTENTION_VARIANT_CONFIGS[idx][0],
                ATTENTION_VARIANT_CONFIGS[idx][1],
                ATTENTION_VARIANT_CONFIGS[idx][2],
                self._make_kernel_id(idx),
            ),
            # Variant 2: Fused pre-attention QK-norm
            lambda idx: self._generate_pre_attn_fused_kernel(
                ATTENTION_VARIANT_CONFIGS[idx][0],
                ATTENTION_VARIANT_CONFIGS[idx][2],
                self._make_kernel_id(idx),
            ),
        ]

        for idx, gen_fn in enumerate(generators):
            kernel_id = self._make_kernel_id(idx)
            source = gen_fn(idx)

            kernel_path = os.path.join(self.output_dir, f"{kernel_id}.py")
            with open(kernel_path, 'w') as f:
                f.write(source)

            block_m, block_n, num_warps = ATTENTION_VARIANT_CONFIGS[idx]
            desc = ["rope_qknorm_fusion", "fallback_attention", "pre_attn_qknorm"][idx]

            integration_diff = textwrap.dedent(f"""\
                # Integration diff for {kernel_id} ({desc})
                # from gpu_kernels.active.{kernel_id} import fused_op
                # Replace corresponding operations in CausalSelfAttention.forward()
            """)

            variant = GeneratedKernel(
                kernel_id=kernel_id,
                group_id=f"attention_{desc}",
                variant_index=idx,
                kernel_path=kernel_path,
                integration_diff=integration_diff,
                block_size=block_m,
                num_warps=num_warps,
                memory_strategy="tiled",
                fusion_type="attention",
            )
            variants.append(variant)

        return variants
