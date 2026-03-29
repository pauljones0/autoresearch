"""
Triton elementwise kernel generator: generates 5 kernel variants per target
with varying BLOCK_SIZE and num_warps configurations.
"""

import os
import hashlib
import textwrap
from typing import List

from ..schemas import KernelTarget, GeneratedKernel


# Variant configurations: (BLOCK_SIZE, num_warps)
VARIANT_CONFIGS = [
    (128, 2),
    (256, 4),
    (512, 4),
    (1024, 8),
    (2048, 8),
]


class TritonElementwiseGenerator:
    """Generate Triton kernel variants for elementwise fusion targets.

    Produces 5 variants per target with different BLOCK_SIZE and num_warps
    configurations. Each variant is an importable Python module containing
    a @triton.jit kernel and a Python wrapper function.
    """

    def __init__(self, output_dir: str = ""):
        if not output_dir:
            output_dir = os.path.join(
                os.path.dirname(__file__), '..', 'generated'
            )
        self.output_dir = os.path.abspath(output_dir)

    def _make_kernel_id(self, group_id: str, variant_index: int) -> str:
        """Generate a deterministic kernel ID."""
        raw = f"{group_id}_elementwise_v{variant_index}"
        short_hash = hashlib.md5(raw.encode()).hexdigest()[:8]
        return f"ew_{group_id}_{variant_index}_{short_hash}"

    def _detect_op_chain(self, target: KernelTarget) -> str:
        """Detect the operation chain from the target's op_sequence and return
        the Triton computation expression and any needed imports."""
        ops = [op.lower().replace("aten::", "") for op in target.op_sequence]
        ops_str = "_".join(ops)

        # Match known patterns from train.py
        if "relu" in ops_str and "square" in ops_str:
            return "relu_square"
        if "rms_norm" in ops_str or "rmsnorm" in ops_str:
            return "rms_norm"
        if "tanh" in ops_str and ("mul" in ops_str or "softcap" in ops_str):
            return "softcap_tanh"
        if "sigmoid" in ops_str and "mul" in ops_str:
            return "silu"

        # Unrecognized op chain — refuse to generate a no-op kernel
        raise ValueError(
            f"Unrecognized elementwise op chain: {ops_str} "
            f"(from ops: {target.op_sequence}). "
            f"Supported patterns: relu_square, rms_norm, softcap_tanh, silu."
        )

    def _generate_kernel_source(
        self, target: KernelTarget, block_size: int, num_warps: int,
        kernel_id: str
    ) -> str:
        """Generate Triton kernel source code as a string."""
        op_chain = self._detect_op_chain(target)
        func_name = f"kernel_{kernel_id}".replace("-", "_")
        wrapper_name = f"fused_{op_chain}_{block_size}"

        if op_chain == "relu_square":
            kernel_body = textwrap.dedent(f'''\
                """Fused ReLU + Square elementwise kernel."""
                import torch
                import triton
                import triton.language as tl


                @triton.jit
                def {func_name}(
                    x_ptr, out_ptr, n_elements,
                    BLOCK_SIZE: tl.constexpr,
                ):
                    pid = tl.program_id(0)
                    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
                    mask = offsets < n_elements
                    x = tl.load(x_ptr + offsets, mask=mask)
                    # Fused relu().square()
                    x = tl.where(x > 0, x, 0.0)
                    x = x * x
                    tl.store(out_ptr + offsets, x, mask=mask)


                def {wrapper_name}(x: torch.Tensor) -> torch.Tensor:
                    """Python wrapper for fused relu+square kernel."""
                    assert x.is_cuda, "Input must be on CUDA"
                    out = torch.empty_like(x)
                    n_elements = x.numel()
                    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
                    {func_name}[grid](x, out, n_elements, BLOCK_SIZE={block_size}, num_warps={num_warps})
                    return out


                # Entry point for integration
                fused_op = {wrapper_name}
            ''')

        elif op_chain == "softcap_tanh":
            kernel_body = textwrap.dedent(f'''\
                """Fused softcap (x = cap * tanh(x / cap)) elementwise kernel."""
                import torch
                import triton
                import triton.language as tl


                @triton.jit
                def {func_name}(
                    x_ptr, out_ptr, n_elements, softcap,
                    BLOCK_SIZE: tl.constexpr,
                ):
                    pid = tl.program_id(0)
                    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
                    mask = offsets < n_elements
                    x = tl.load(x_ptr + offsets, mask=mask)
                    # Fused softcap: cap * tanh(x / cap)
                    x = x / softcap
                    x = tl.extra.cuda.libdevice.tanh(x)
                    x = x * softcap
                    tl.store(out_ptr + offsets, x, mask=mask)


                def {wrapper_name}(x: torch.Tensor, softcap: float = 15.0) -> torch.Tensor:
                    """Python wrapper for fused softcap kernel."""
                    assert x.is_cuda, "Input must be on CUDA"
                    out = torch.empty_like(x)
                    n_elements = x.numel()
                    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
                    {func_name}[grid](x, out, n_elements, softcap, BLOCK_SIZE={block_size}, num_warps={num_warps})
                    return out


                # Entry point for integration
                fused_op = {wrapper_name}
            ''')

        elif op_chain == "rms_norm":
            kernel_body = textwrap.dedent(f'''\
                """Fused RMSNorm elementwise kernel."""
                import torch
                import triton
                import triton.language as tl


                @triton.jit
                def {func_name}(
                    x_ptr, out_ptr, n_rows, n_cols, eps,
                    BLOCK_SIZE: tl.constexpr,
                ):
                    row_idx = tl.program_id(0)
                    col_offsets = tl.arange(0, BLOCK_SIZE)
                    mask = col_offsets < n_cols
                    x = tl.load(x_ptr + row_idx * n_cols + col_offsets, mask=mask, other=0.0)
                    # RMS norm: x / sqrt(mean(x^2) + eps)
                    x_sq = x * x
                    mean_sq = tl.sum(x_sq, axis=0) / n_cols
                    rms = tl.sqrt(mean_sq + eps)
                    out = x / rms
                    tl.store(out_ptr + row_idx * n_cols + col_offsets, out, mask=mask)


                def {wrapper_name}(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
                    """Python wrapper for fused RMSNorm kernel."""
                    assert x.is_cuda, "Input must be on CUDA"
                    orig_shape = x.shape
                    x_2d = x.view(-1, x.shape[-1])
                    n_rows, n_cols = x_2d.shape
                    out = torch.empty_like(x_2d)
                    {func_name}[(n_rows,)](x_2d, out, n_rows, n_cols, eps, BLOCK_SIZE={block_size}, num_warps={num_warps})
                    return out.view(orig_shape)


                # Entry point for integration
                fused_op = {wrapper_name}
            ''')

        elif op_chain == "silu":
            kernel_body = textwrap.dedent(f'''\
                """Fused SiLU (x * sigmoid(x)) elementwise kernel."""
                import torch
                import triton
                import triton.language as tl


                @triton.jit
                def {func_name}(
                    x_ptr, out_ptr, n_elements,
                    BLOCK_SIZE: tl.constexpr,
                ):
                    pid = tl.program_id(0)
                    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
                    mask = offsets < n_elements
                    x = tl.load(x_ptr + offsets, mask=mask)
                    # Fused SiLU: x * sigmoid(x)
                    sigmoid_x = 1.0 / (1.0 + tl.exp(-x))
                    out = x * sigmoid_x
                    tl.store(out_ptr + offsets, out, mask=mask)


                def {wrapper_name}(x: torch.Tensor) -> torch.Tensor:
                    """Python wrapper for fused SiLU kernel."""
                    assert x.is_cuda, "Input must be on CUDA"
                    out = torch.empty_like(x)
                    n_elements = x.numel()
                    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
                    {func_name}[grid](x, out, n_elements, BLOCK_SIZE={block_size}, num_warps={num_warps})
                    return out


                # Entry point for integration
                fused_op = {wrapper_name}
            ''')

        else:
            # Generic elementwise fusion template
            kernel_body = textwrap.dedent(f'''\
                """Generic fused elementwise kernel for ops: {target.op_sequence}."""
                import torch
                import triton
                import triton.language as tl


                @triton.jit
                def {func_name}(
                    x_ptr, out_ptr, n_elements,
                    BLOCK_SIZE: tl.constexpr,
                ):
                    pid = tl.program_id(0)
                    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
                    mask = offsets < n_elements
                    x = tl.load(x_ptr + offsets, mask=mask)
                    # Placeholder: identity (customize per op chain)
                    tl.store(out_ptr + offsets, x, mask=mask)


                def {wrapper_name}(x: torch.Tensor) -> torch.Tensor:
                    """Python wrapper for generic fused elementwise kernel."""
                    assert x.is_cuda, "Input must be on CUDA"
                    out = torch.empty_like(x)
                    n_elements = x.numel()
                    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
                    {func_name}[grid](x, out, n_elements, BLOCK_SIZE={block_size}, num_warps={num_warps})
                    return out


                # Entry point for integration
                fused_op = {wrapper_name}
            ''')

        return kernel_body

    def _generate_integration_diff(
        self, target: KernelTarget, kernel_id: str, op_chain: str
    ) -> str:
        """Generate a diff showing how to integrate the kernel into train.py."""
        module_path = f"gpu_kernels.active.{kernel_id}"

        if op_chain == "relu_square":
            return textwrap.dedent(f"""\
                # Integration diff for {kernel_id}
                # In MLP.forward(), replace:
                #     x = F.relu(x).square()
                # With:
                #     from {module_path} import fused_op as fused_relu_square
                #     x = fused_relu_square(x)
            """)
        elif op_chain == "softcap_tanh":
            return textwrap.dedent(f"""\
                # Integration diff for {kernel_id}
                # In GPT.forward(), replace:
                #     logits = softcap * torch.tanh(logits / softcap)
                # With:
                #     from {module_path} import fused_op as fused_softcap
                #     logits = fused_softcap(logits, softcap=softcap)
            """)
        elif op_chain == "rms_norm":
            return textwrap.dedent(f"""\
                # Integration diff for {kernel_id}
                # In norm(), replace:
                #     return F.rms_norm(x, (x.size(-1),))
                # With:
                #     from {module_path} import fused_op as fused_rms_norm
                #     return fused_rms_norm(x)
            """)
        else:
            return textwrap.dedent(f"""\
                # Integration diff for {kernel_id}
                # Import: from {module_path} import fused_op
                # Replace target operation with: fused_op(x)
            """)

    def generate(self, target: KernelTarget) -> List[GeneratedKernel]:
        """Generate 5 Triton kernel variants for the given target.

        Args:
            target: KernelTarget describing the fusion opportunity.

        Returns:
            List of 5 GeneratedKernel objects with different configurations.
        """
        os.makedirs(self.output_dir, exist_ok=True)

        op_chain = self._detect_op_chain(target)
        variants = []

        for idx, (block_size, num_warps) in enumerate(VARIANT_CONFIGS):
            kernel_id = self._make_kernel_id(target.group_id, idx)

            # Generate kernel source
            source = self._generate_kernel_source(
                target, block_size, num_warps, kernel_id
            )

            # Write kernel file
            kernel_filename = f"{kernel_id}.py"
            kernel_path = os.path.join(self.output_dir, kernel_filename)
            with open(kernel_path, 'w') as f:
                f.write(source)

            # Generate integration diff
            integration_diff = self._generate_integration_diff(
                target, kernel_id, op_chain
            )

            variant = GeneratedKernel(
                kernel_id=kernel_id,
                group_id=target.group_id,
                variant_index=idx,
                kernel_path=kernel_path,
                integration_diff=integration_diff,
                block_size=block_size,
                num_warps=num_warps,
                memory_strategy="row_major",
                fusion_type="elementwise",
            )
            variants.append(variant)

        return variants
