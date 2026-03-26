"""
Normalization kernel generator: generates fused RMSNorm + residual Triton kernels.
"""

import os
import hashlib
import textwrap
from typing import List

from ..schemas import GeneratedKernel


# Variant configurations: (BLOCK_SIZE, residual_strategy)
NORM_VARIANT_CONFIGS = [
    (256, "separate"),      # Separate norm and residual
    (512, "fused_add"),     # Fused norm + residual add
    (1024, "fused_add"),    # Larger block, fused
    (2048, "fused_scale"),  # Fused norm + residual with per-layer scaling
]


class NormalizationKernelGenerator:
    """Generate fused RMSNorm + residual Triton kernels.

    Targets the pattern in train.py Block.forward():
        x = resid_lambda * x + x0_lambda * x0  (residual scaling)
        x = x + self.attn(norm(x), ...)         (norm + attention)
        x = x + self.mlp(norm(x))               (norm + MLP)

    The fused kernel combines RMSNorm with residual addition to avoid
    extra memory reads/writes for intermediate results.
    """

    def __init__(self, output_dir: str = ""):
        if not output_dir:
            output_dir = os.path.join(
                os.path.dirname(__file__), '..', 'generated'
            )
        self.output_dir = os.path.abspath(output_dir)

    def _make_kernel_id(self, variant_index: int) -> str:
        raw = f"rmsnorm_resid_v{variant_index}"
        short_hash = hashlib.md5(raw.encode()).hexdigest()[:8]
        return f"norm_resid_{variant_index}_{short_hash}"

    def _generate_separate_kernel(
        self, block_size: int, kernel_id: str
    ) -> str:
        """Generate RMSNorm-only kernel (baseline, no residual fusion)."""
        func_name = f"kernel_{kernel_id}".replace("-", "_")
        wrapper_name = f"fused_rmsnorm_{block_size}"

        return textwrap.dedent(f'''\
            """Fused RMSNorm Triton kernel (no residual fusion)."""
            import torch
            import triton
            import triton.language as tl


            @triton.jit
            def {func_name}(
                x_ptr, out_ptr,
                n_rows, n_cols, eps,
                BLOCK_SIZE: tl.constexpr,
            ):
                row_idx = tl.program_id(0)
                col_offsets = tl.arange(0, BLOCK_SIZE)
                mask = col_offsets < n_cols

                x = tl.load(x_ptr + row_idx * n_cols + col_offsets, mask=mask, other=0.0).to(tl.float32)
                # RMS norm
                x_sq = x * x
                mean_sq = tl.sum(x_sq, axis=0) / n_cols
                rms = tl.sqrt(mean_sq + eps)
                out = x / rms
                tl.store(out_ptr + row_idx * n_cols + col_offsets, out, mask=mask)


            def {wrapper_name}(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
                """RMSNorm kernel wrapper."""
                assert x.is_cuda
                orig_shape = x.shape
                x_2d = x.reshape(-1, x.shape[-1])
                n_rows, n_cols = x_2d.shape
                out = torch.empty_like(x_2d)
                {func_name}[(n_rows,)](x_2d, out, n_rows, n_cols, eps, BLOCK_SIZE={block_size}, num_warps=4)
                return out.view(orig_shape)


            fused_op = {wrapper_name}
        ''')

    def _generate_fused_add_kernel(
        self, block_size: int, num_warps: int, kernel_id: str
    ) -> str:
        """Generate fused RMSNorm + residual add kernel.

        Computes: norm_out = RMSNorm(x + residual), returning both
        the normalized output and the pre-norm sum.
        """
        func_name = f"kernel_{kernel_id}".replace("-", "_")
        wrapper_name = f"fused_rmsnorm_residual_{block_size}"

        return textwrap.dedent(f'''\
            """Fused RMSNorm + Residual Add Triton kernel.
            Computes: hidden = x + residual; norm_out = RMSNorm(hidden)
            Returns both norm_out and hidden (for next residual).
            """
            import torch
            import triton
            import triton.language as tl


            @triton.jit
            def {func_name}(
                x_ptr, residual_ptr, out_norm_ptr, out_hidden_ptr,
                n_rows, n_cols, eps,
                BLOCK_SIZE: tl.constexpr,
            ):
                row_idx = tl.program_id(0)
                col_offsets = tl.arange(0, BLOCK_SIZE)
                mask = col_offsets < n_cols
                base = row_idx * n_cols

                x = tl.load(x_ptr + base + col_offsets, mask=mask, other=0.0).to(tl.float32)
                residual = tl.load(residual_ptr + base + col_offsets, mask=mask, other=0.0).to(tl.float32)

                # Fused residual add
                hidden = x + residual

                # RMS norm on the sum
                hidden_sq = hidden * hidden
                mean_sq = tl.sum(hidden_sq, axis=0) / n_cols
                rms = tl.sqrt(mean_sq + eps)
                norm_out = hidden / rms

                tl.store(out_hidden_ptr + base + col_offsets, hidden, mask=mask)
                tl.store(out_norm_ptr + base + col_offsets, norm_out, mask=mask)


            def {wrapper_name}(
                x: torch.Tensor, residual: torch.Tensor, eps: float = 1e-6
            ) -> tuple:
                """Fused RMSNorm + residual add.

                Returns:
                    (norm_out, hidden) where hidden = x + residual
                """
                assert x.is_cuda and residual.is_cuda
                orig_shape = x.shape
                x_2d = x.reshape(-1, x.shape[-1])
                r_2d = residual.reshape(-1, residual.shape[-1])
                n_rows, n_cols = x_2d.shape
                out_norm = torch.empty_like(x_2d)
                out_hidden = torch.empty_like(x_2d)
                {func_name}[(n_rows,)](
                    x_2d, r_2d, out_norm, out_hidden,
                    n_rows, n_cols, eps,
                    BLOCK_SIZE={block_size}, num_warps={num_warps},
                )
                return out_norm.view(orig_shape), out_hidden.view(orig_shape)


            fused_op = {wrapper_name}
        ''')

    def _generate_fused_scale_kernel(
        self, block_size: int, kernel_id: str
    ) -> str:
        """Generate fused RMSNorm + residual with per-layer scaling.

        Matches train.py: x = resid_lambda * x + x0_lambda * x0
        then norm(x) for the block input.
        """
        func_name = f"kernel_{kernel_id}".replace("-", "_")
        wrapper_name = f"fused_rmsnorm_scaled_residual_{block_size}"

        return textwrap.dedent(f'''\
            """Fused RMSNorm + Scaled Residual Triton kernel.
            Computes: hidden = resid_lambda * x + x0_lambda * x0
                      norm_out = RMSNorm(hidden)
            """
            import torch
            import triton
            import triton.language as tl


            @triton.jit
            def {func_name}(
                x_ptr, x0_ptr, out_norm_ptr, out_hidden_ptr,
                n_rows, n_cols, resid_lambda, x0_lambda, eps,
                BLOCK_SIZE: tl.constexpr,
            ):
                row_idx = tl.program_id(0)
                col_offsets = tl.arange(0, BLOCK_SIZE)
                mask = col_offsets < n_cols
                base = row_idx * n_cols

                x = tl.load(x_ptr + base + col_offsets, mask=mask, other=0.0).to(tl.float32)
                x0 = tl.load(x0_ptr + base + col_offsets, mask=mask, other=0.0).to(tl.float32)

                # Scaled residual connection
                hidden = resid_lambda * x + x0_lambda * x0

                # RMS norm
                hidden_sq = hidden * hidden
                mean_sq = tl.sum(hidden_sq, axis=0) / n_cols
                rms = tl.sqrt(mean_sq + eps)
                norm_out = hidden / rms

                tl.store(out_hidden_ptr + base + col_offsets, hidden, mask=mask)
                tl.store(out_norm_ptr + base + col_offsets, norm_out, mask=mask)


            def {wrapper_name}(
                x: torch.Tensor, x0: torch.Tensor,
                resid_lambda: float, x0_lambda: float,
                eps: float = 1e-6,
            ) -> tuple:
                """Fused RMSNorm + scaled residual.

                Args:
                    x: Current hidden state.
                    x0: Initial hidden state (post-embedding norm).
                    resid_lambda: Per-layer residual scaling factor.
                    x0_lambda: Per-layer x0 mixing factor.

                Returns:
                    (norm_out, hidden) where hidden = resid_lambda*x + x0_lambda*x0
                """
                assert x.is_cuda and x0.is_cuda
                orig_shape = x.shape
                x_2d = x.reshape(-1, x.shape[-1])
                x0_2d = x0.reshape(-1, x0.shape[-1])
                n_rows, n_cols = x_2d.shape
                out_norm = torch.empty_like(x_2d)
                out_hidden = torch.empty_like(x_2d)
                {func_name}[(n_rows,)](
                    x_2d, x0_2d, out_norm, out_hidden,
                    n_rows, n_cols, resid_lambda, x0_lambda, eps,
                    BLOCK_SIZE={block_size}, num_warps=8,
                )
                return out_norm.view(orig_shape), out_hidden.view(orig_shape)


            fused_op = {wrapper_name}
        ''')

    def generate(self, model_dim: int, n_layers: int) -> List[GeneratedKernel]:
        """Generate 4 fused RMSNorm + residual kernel variants.

        Args:
            model_dim: Model embedding dimension (used for BLOCK_SIZE validation).
            n_layers: Number of transformer layers (for documentation).

        Returns:
            List of 4 GeneratedKernel objects.
        """
        os.makedirs(self.output_dir, exist_ok=True)
        variants = []

        for idx, (block_size, strategy) in enumerate(NORM_VARIANT_CONFIGS):
            kernel_id = self._make_kernel_id(idx)

            # Ensure BLOCK_SIZE >= model_dim for row-wise kernels
            effective_block = max(block_size, model_dim)
            # Round up to next power of 2 for Triton
            effective_block = 1 << (effective_block - 1).bit_length()

            if strategy == "separate":
                source = self._generate_separate_kernel(effective_block, kernel_id)
            elif strategy == "fused_add":
                num_warps = 4 if block_size <= 512 else 8
                source = self._generate_fused_add_kernel(effective_block, num_warps, kernel_id)
            elif strategy == "fused_scale":
                source = self._generate_fused_scale_kernel(effective_block, kernel_id)
            else:
                continue

            kernel_path = os.path.join(self.output_dir, f"{kernel_id}.py")
            with open(kernel_path, 'w') as f:
                f.write(source)

            integration_diff = textwrap.dedent(f"""\
                # Integration diff for {kernel_id} (strategy: {strategy})
                # Target: norm() function and Block.forward() residual connection in train.py
                # Replace norm(x) calls with fused kernel that combines norm + residual.
                # from gpu_kernels.active.{kernel_id} import fused_op
            """)

            variant = GeneratedKernel(
                kernel_id=kernel_id,
                group_id="rmsnorm_residual",
                variant_index=idx,
                kernel_path=kernel_path,
                integration_diff=integration_diff,
                block_size=effective_block,
                num_warps=4 if block_size <= 512 else 8,
                memory_strategy=strategy,
                fusion_type="normalization",
            )
            variants.append(variant)

        return variants
