"""
KernelMutationEngine: applies parametric mutations to a parent kernel source.
Each mutation changes exactly one aspect of the kernel.
"""

import ast
import re
import uuid

from ..schemas import MutatedKernel


# Mutation types and their descriptions
_MUTATION_TYPES = [
    "block_size_double",
    "block_size_half",
    "num_warps_increase",
    "num_warps_decrease",
    "memory_access_swap",
    "loop_unroll_add",
    "loop_unroll_remove",
    "accumulation_dtype_fp64",
    "accumulation_dtype_fp32",
    "shared_memory_add",
    "shared_memory_remove",
]


class KernelMutationEngine:
    """Apply parametric mutations to a parent Triton kernel source.

    Supported mutations:
      - block_size: x2 or /2
      - num_warps: +1 or -1
      - memory_access: swap row<->column major
      - loop_unroll: add or remove tl.static_range
      - accumulation_dtype: fp32 <-> fp64
      - shared_memory: add or remove caching
    """

    def mutate(
        self,
        parent_source: str,
        parent_id: str,
        n_mutations: int = 5,
    ) -> list[MutatedKernel]:
        """Generate mutations of the parent kernel.

        Args:
            parent_source: Triton kernel source code.
            parent_id: Identifier of the parent kernel.
            n_mutations: Number of mutations to attempt (default 5).

        Returns:
            List of MutatedKernel with valid syntax. Syntactically invalid
            mutations are discarded.
        """
        results: list[MutatedKernel] = []
        attempted = set()

        for mut_type in _MUTATION_TYPES:
            if len(results) >= n_mutations:
                break
            if mut_type in attempted:
                continue
            attempted.add(mut_type)

            mutated_source = _apply_mutation(parent_source, mut_type)
            if mutated_source is None or mutated_source == parent_source:
                continue

            # Discard syntactically invalid results
            if not _is_valid_python(mutated_source):
                continue

            mutation_id = f"{parent_id}_mut_{mut_type}_{uuid.uuid4().hex[:6]}"
            results.append(MutatedKernel(
                mutation_id=mutation_id,
                parent_id=parent_id,
                mutation_type=mut_type,
                mutation_description=_describe_mutation(mut_type),
                kernel_source=mutated_source,
                kernel_path="",
            ))

        return results


def _apply_mutation(source: str, mut_type: str) -> str | None:
    """Apply a single mutation type to the source. Returns None if not applicable."""
    if mut_type == "block_size_double":
        return _mutate_block_size(source, factor=2)
    elif mut_type == "block_size_half":
        return _mutate_block_size(source, factor=0.5)
    elif mut_type == "num_warps_increase":
        return _mutate_num_warps(source, delta=1)
    elif mut_type == "num_warps_decrease":
        return _mutate_num_warps(source, delta=-1)
    elif mut_type == "memory_access_swap":
        return _mutate_memory_access(source)
    elif mut_type == "loop_unroll_add":
        return _mutate_loop_unroll(source, add=True)
    elif mut_type == "loop_unroll_remove":
        return _mutate_loop_unroll(source, add=False)
    elif mut_type == "accumulation_dtype_fp64":
        return _mutate_accumulation_dtype(source, target="tl.float64")
    elif mut_type == "accumulation_dtype_fp32":
        return _mutate_accumulation_dtype(source, target="tl.float32")
    elif mut_type == "shared_memory_add":
        return _mutate_shared_memory(source, add=True)
    elif mut_type == "shared_memory_remove":
        return _mutate_shared_memory(source, add=False)
    return None


def _mutate_block_size(source: str, factor: float) -> str | None:
    """Multiply BLOCK_SIZE constants by factor."""
    pattern = re.compile(r'(BLOCK_SIZE\s*(?::\s*tl\.constexpr\s*)?=\s*)(\d+)')
    match = pattern.search(source)
    if not match:
        return None
    old_val = int(match.group(2))
    new_val = max(64, int(old_val * factor))
    # Ensure power of 2
    new_val = 1 << (new_val - 1).bit_length() if new_val > 0 else 64
    if new_val == old_val:
        return None
    return pattern.sub(lambda m: f"{m.group(1)}{new_val}", source, count=1)


def _mutate_num_warps(source: str, delta: int) -> str | None:
    """Adjust num_warps parameter."""
    pattern = re.compile(r'(num_warps\s*=\s*)(\d+)')
    match = pattern.search(source)
    if not match:
        return None
    old_val = int(match.group(2))
    new_val = max(1, old_val + delta)
    if new_val == old_val:
        return None
    return pattern.sub(lambda m: f"{m.group(1)}{new_val}", source, count=1)


def _mutate_memory_access(source: str) -> str | None:
    """Swap row_major <-> column_major in comments/access patterns."""
    if "row_major" in source:
        result = source.replace("row_major", "column_major", 1)
        # Swap access pattern: [row * cols + col] -> [col * rows + row]
        result = re.sub(
            r'(\w+)\s*\*\s*n_cols\s*\+\s*offsets',
            r'\1 + offsets * n_cols',
            result,
            count=1,
        )
        return result if result != source else None
    elif "column_major" in source:
        result = source.replace("column_major", "row_major", 1)
        result = re.sub(
            r'(\w+)\s*\+\s*offsets\s*\*\s*n_cols',
            r'\1 * n_cols + offsets',
            result,
            count=1,
        )
        return result if result != source else None
    return None


def _mutate_loop_unroll(source: str, add: bool) -> str | None:
    """Add or remove tl.static_range for loop unrolling."""
    if add:
        # Replace range() with tl.static_range() in for loops
        pattern = re.compile(r'for\s+(\w+)\s+in\s+range\(')
        if not pattern.search(source):
            return None
        return pattern.sub(r'for \1 in tl.static_range(', source, count=1)
    else:
        # Replace tl.static_range() with range()
        pattern = re.compile(r'for\s+(\w+)\s+in\s+tl\.static_range\(')
        if not pattern.search(source):
            return None
        return pattern.sub(r'for \1 in range(', source, count=1)


def _mutate_accumulation_dtype(source: str, target: str) -> str | None:
    """Switch accumulation dtype."""
    # Common pattern: tl.sum(..., dtype=tl.float32) or .to(tl.float32)
    if target == "tl.float64":
        old, new = "tl.float32", "tl.float64"
    else:
        old, new = "tl.float64", "tl.float32"

    if old not in source:
        # Try adding dtype to tl.sum calls
        if target == "tl.float64" and "tl.sum(" in source:
            return source.replace("tl.sum(", f"tl.sum(", 1).replace(
                "axis=0)", f"axis=0).to({target})", 1
            )
        return None
    return source.replace(old, new)


def _mutate_shared_memory(source: str, add: bool) -> str | None:
    """Add or remove shared memory caching."""
    if add:
        # Add eviction_policy='evict_last' to tl.load calls (caching hint)
        if "eviction_policy" in source:
            return None
        pattern = re.compile(r"(tl\.load\([^)]+)(mask=[^)]+)\)")
        if not pattern.search(source):
            return None
        return pattern.sub(
            r'\1\2, eviction_policy="evict_last")',
            source,
            count=1,
        )
    else:
        # Remove eviction_policy from tl.load
        if "eviction_policy" not in source:
            return None
        return re.sub(
            r',\s*eviction_policy\s*=\s*"[^"]*"',
            "",
            source,
            count=1,
        )


def _describe_mutation(mut_type: str) -> str:
    """Human-readable description of a mutation type."""
    descriptions = {
        "block_size_double": "Double BLOCK_SIZE",
        "block_size_half": "Halve BLOCK_SIZE",
        "num_warps_increase": "Increase num_warps by 1",
        "num_warps_decrease": "Decrease num_warps by 1",
        "memory_access_swap": "Swap memory access pattern (row<->column)",
        "loop_unroll_add": "Add loop unrolling with tl.static_range",
        "loop_unroll_remove": "Remove loop unrolling",
        "accumulation_dtype_fp64": "Switch accumulation to fp64",
        "accumulation_dtype_fp32": "Switch accumulation to fp32",
        "shared_memory_add": "Add shared memory caching hint",
        "shared_memory_remove": "Remove shared memory caching hint",
    }
    return descriptions.get(mut_type, mut_type)


def _is_valid_python(source: str) -> bool:
    """Check if source is syntactically valid Python."""
    try:
        ast.parse(source)
        return True
    except SyntaxError:
        return False
