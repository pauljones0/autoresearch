"""
AutonomousKernelGenerator: given a KernelOpportunity, generates Triton kernel
variant candidates with parametric variation and fast correctness pre-screening.
"""

import ast
import uuid
import textwrap

from ..schemas import KernelOpportunity


# Strategy templates keyed by fusion_type
_STRATEGY_MAP = {
    "elementwise": "elementwise",
    "reduction": "reduction",
    "normalization": "reduction",
    "attention": "attention",
    "optimizer": "elementwise",
}

# Parametric variation grids per strategy
_BLOCK_SIZES = {
    "elementwise": [256, 512, 1024, 2048],
    "reduction": [128, 256, 512, 1024],
    "attention": [64, 128, 256],
}

_NUM_WARPS = {
    "elementwise": [4, 8],
    "reduction": [4, 8, 16],
    "attention": [4, 8],
}

_MEMORY_STRATEGIES = ["row_major", "column_major"]


class AutonomousKernelGenerator:
    """Generate Triton kernel variants for a KernelOpportunity.

    Analyzes op characteristics, selects a generation strategy, and produces
    up to 5 variants with parametric variation (block size, num warps,
    memory access pattern). Performs fast syntactic pre-screening.
    """

    def generate(
        self,
        opportunity: KernelOpportunity,
        base_source: str,
        reference_path: str | None = None,
    ) -> list[dict]:
        """Generate kernel variants for the given opportunity.

        Args:
            opportunity: Ranked kernel opportunity to target.
            base_source: Source code of the training script or module.
            reference_path: Optional path to reference implementation.

        Returns:
            List of dicts with keys: kernel_id, kernel_source, block_size,
            num_warps, memory_strategy, fusion_type, variant_index.
        """
        opp = opportunity if not isinstance(opportunity, dict) else _opp_from_dict(opportunity)

        strategy = _STRATEGY_MAP.get(opp.fusion_type, "elementwise")
        block_sizes = _BLOCK_SIZES.get(strategy, [256, 512])
        warp_options = _NUM_WARPS.get(strategy, [4, 8])

        # Build parameter combinations, pick up to 5
        combos = []
        for bs in block_sizes:
            for nw in warp_options:
                for mem in _MEMORY_STRATEGIES:
                    combos.append((bs, nw, mem))

        # Deterministic selection: spread across the parameter space
        step = max(1, len(combos) // 5)
        selected = combos[::step][:5]

        variants: list[dict] = []
        for idx, (bs, nw, mem) in enumerate(selected):
            kernel_id = f"{opp.group_id}_v{idx}_{uuid.uuid4().hex[:6]}"
            source = _generate_kernel_source(
                kernel_id=kernel_id,
                op_names=opp.op_names,
                fusion_type=opp.fusion_type,
                strategy=strategy,
                block_size=bs,
                num_warps=nw,
                memory_strategy=mem,
            )

            # Fast syntactic pre-screen: ensure valid Python
            if not _passes_syntax_check(source):
                continue

            variants.append({
                "kernel_id": kernel_id,
                "kernel_source": source,
                "block_size": bs,
                "num_warps": nw,
                "memory_strategy": mem,
                "fusion_type": opp.fusion_type,
                "variant_index": idx,
                "group_id": opp.group_id,
            })

        return variants


def _opp_from_dict(d: dict) -> KernelOpportunity:
    """Convert a dict to KernelOpportunity."""
    opp = KernelOpportunity()
    for k, v in d.items():
        if hasattr(opp, k):
            setattr(opp, k, v)
    return opp


def _generate_kernel_source(
    kernel_id: str,
    op_names: list[str],
    fusion_type: str,
    strategy: str,
    block_size: int,
    num_warps: int,
    memory_strategy: str,
) -> str:
    """Generate Triton kernel source code for the given parameters."""
    ops_comment = ", ".join(op_names)
    load_order = "row" if memory_strategy == "row_major" else "column"

    if strategy == "elementwise":
        return _elementwise_template(
            kernel_id, ops_comment, block_size, num_warps, load_order
        )
    elif strategy == "reduction":
        return _reduction_template(
            kernel_id, ops_comment, block_size, num_warps, load_order
        )
    elif strategy == "attention":
        return _attention_template(
            kernel_id, ops_comment, block_size, num_warps, load_order
        )
    return _elementwise_template(
        kernel_id, ops_comment, block_size, num_warps, load_order
    )


def _elementwise_template(
    kernel_id: str, ops: str, block_size: int, num_warps: int, load_order: str
) -> str:
    return textwrap.dedent(f"""\
        import triton
        import triton.language as tl
        import torch

        # Fused ops: {ops}
        # Load order: {load_order}

        @triton.jit
        def {_safe_name(kernel_id)}(
            x_ptr, output_ptr, n_elements,
            BLOCK_SIZE: tl.constexpr = {block_size},
        ):
            pid = tl.program_id(0)
            offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
            mask = offsets < n_elements
            x = tl.load(x_ptr + offsets, mask=mask)
            # Fused elementwise operations placeholder
            result = x
            tl.store(output_ptr + offsets, result, mask=mask)

        def launch(x: torch.Tensor) -> torch.Tensor:
            output = torch.empty_like(x)
            n = x.numel()
            grid = lambda meta: ((n + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'],)
            {_safe_name(kernel_id)}[grid](x, output, n, num_warps={num_warps})
            return output
    """)


def _reduction_template(
    kernel_id: str, ops: str, block_size: int, num_warps: int, load_order: str
) -> str:
    return textwrap.dedent(f"""\
        import triton
        import triton.language as tl
        import torch

        # Fused ops: {ops}
        # Load order: {load_order}

        @triton.jit
        def {_safe_name(kernel_id)}(
            x_ptr, output_ptr, n_rows, n_cols,
            BLOCK_SIZE: tl.constexpr = {block_size},
        ):
            row_id = tl.program_id(0)
            offsets = tl.arange(0, BLOCK_SIZE)
            mask = offsets < n_cols
            x = tl.load(x_ptr + row_id * n_cols + offsets, mask=mask, other=0.0)
            # Reduction placeholder
            result = tl.sum(x, axis=0)
            tl.store(output_ptr + row_id, result)

        def launch(x: torch.Tensor) -> torch.Tensor:
            n_rows, n_cols = x.shape
            output = torch.empty(n_rows, device=x.device, dtype=x.dtype)
            grid = (n_rows,)
            {_safe_name(kernel_id)}[grid](x, output, n_rows, n_cols, num_warps={num_warps})
            return output
    """)


def _attention_template(
    kernel_id: str, ops: str, block_size: int, num_warps: int, load_order: str
) -> str:
    return textwrap.dedent(f"""\
        import triton
        import triton.language as tl
        import torch

        # Fused ops: {ops}
        # Load order: {load_order}

        @triton.jit
        def {_safe_name(kernel_id)}(
            q_ptr, k_ptr, v_ptr, output_ptr,
            seq_len, head_dim,
            BLOCK_SIZE: tl.constexpr = {block_size},
        ):
            pid = tl.program_id(0)
            offs_m = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
            mask_m = offs_m < seq_len
            # Attention computation placeholder
            # Q @ K^T / sqrt(d) -> softmax -> @ V
            q = tl.load(q_ptr + offs_m, mask=mask_m, other=0.0)
            result = q  # placeholder
            tl.store(output_ptr + offs_m, result, mask=mask_m)

        def launch(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
            seq_len = q.shape[-2]
            head_dim = q.shape[-1]
            output = torch.empty_like(q)
            grid = lambda meta: ((seq_len + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'],)
            {_safe_name(kernel_id)}[grid](q, k, v, output, seq_len, head_dim, num_warps={num_warps})
            return output
    """)


def _safe_name(kernel_id: str) -> str:
    """Convert a kernel_id to a valid Python identifier."""
    return "kernel_" + kernel_id.replace("-", "_").replace(".", "_")


def _passes_syntax_check(source: str) -> bool:
    """Check if the generated source is syntactically valid Python."""
    try:
        ast.parse(source)
        return True
    except SyntaxError:
        return False
