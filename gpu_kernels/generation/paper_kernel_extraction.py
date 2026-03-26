"""
Phase 3 — KernelPaperExtractor: extract kernel-specific structured data from
papers and generate template-based Triton diffs for kernel techniques.
"""

import difflib
import logging
import math
import re
import uuid

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Patterns for extracting kernel-specific information from paper text
# ---------------------------------------------------------------------------

_BLOCK_SIZE_PATTERN = re.compile(
    r"block\s*size\s*(?:of\s+)?(\d+)", re.IGNORECASE
)
_SPEEDUP_PATTERN = re.compile(
    r"(\d+(?:\.\d+)?)\s*[xX×]\s*(?:speed\s*up|faster|improvement)", re.IGNORECASE
)
_SPEEDUP_PERCENT_PATTERN = re.compile(
    r"(\d+(?:\.\d+)?)\s*%\s*(?:speed\s*up|faster|improvement|reduction)", re.IGNORECASE
)
_HARDWARE_PATTERN = re.compile(
    r"(A100|H100|V100|A10|T4|RTX\s*\d{4}|MI\d{3}|GH200)", re.IGNORECASE
)
_TARGET_OP_PATTERN = re.compile(
    r"(?:for|targeting|applied to|optimiz\w+ the)\s+"
    r"(attention|softmax|layernorm|layer norm|matmul|gemm|"
    r"elementwise|reduction|normalization|linear|embedding|"
    r"cross[- ]entropy|dropout|gelu|relu|silu|activation)\b",
    re.IGNORECASE,
)
_MEMORY_STRATEGY_PATTERN = re.compile(
    r"(tiling|tiled|row[- ]major|column[- ]major|"
    r"coalesced|vectorized|shared memory|register blocking)",
    re.IGNORECASE,
)
_CODE_BLOCK_PATTERN = re.compile(
    r"```(?:python|triton|cuda)?\s*\n(.*?)```",
    re.DOTALL,
)
_TRITON_INDICATOR = re.compile(
    r"(?:@triton\.jit|tl\.load|tl\.store|triton\.language|tl\.arange)",
    re.IGNORECASE,
)
_CUDA_INDICATOR = re.compile(
    r"(?:__global__|__shared__|__syncthreads|blockIdx|threadIdx|cudaMalloc)",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Default block size variants for diff generation
# ---------------------------------------------------------------------------

_BLOCK_SIZE_VARIANTS = [128, 256, 512]

# ---------------------------------------------------------------------------
# Triton kernel template for elementwise / fused operations
# ---------------------------------------------------------------------------

_TRITON_KERNEL_TEMPLATE = '''\
"""Auto-generated Triton kernel for {target_op} ({technique_name})."""

import triton
import triton.language as tl
import torch


@triton.jit
def {kernel_fn_name}(
    input_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr = {block_size},
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(input_ptr + offsets, mask=mask)
    # --- technique-specific compute ---
    y = x  # placeholder for {target_op} operation
    # --- end technique-specific compute ---
    tl.store(output_ptr + offsets, y, mask=mask)


def launch_{kernel_fn_name}(x: torch.Tensor) -> torch.Tensor:
    """Launch the Triton kernel on *x* and return the result."""
    output = torch.empty_like(x)
    n = x.numel()
    grid = lambda meta: ((n + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    {kernel_fn_name}[grid](x, output, n, BLOCK_SIZE={block_size})
    return output
'''

# ---------------------------------------------------------------------------
# Integration diff template (patch for train.py)
# ---------------------------------------------------------------------------

_INTEGRATION_TEMPLATE = '''\
--- train.py
+++ train.py
@@ -1,3 +1,4 @@
+from {kernel_module} import launch_{kernel_fn_name}
 # ... existing imports ...
'''

# ---------------------------------------------------------------------------
# Simple syntax check for Triton code
# ---------------------------------------------------------------------------

_TRITON_SYNTAX_ESSENTIALS = [
    "tl.load",
    "tl.store",
    "@triton.jit",
    "tl.program_id",
]


class KernelPaperExtractor:
    """Extract kernel-specific structured data from papers and generate
    template-based Triton diffs for kernel techniques.
    """

    def extract_kernel_technique(self, paper_metadata: dict) -> dict:
        """Extract kernel-specific technique fields from paper metadata.

        Parameters
        ----------
        paper_metadata : dict
            Must contain at least 'abstract'. May contain 'title', 'arxiv_id',
            and 'full_text'.

        Returns
        -------
        dict
            Technique dict with kernel-specific fields:
            - technique_id, paper_id, name, target_operation,
              block_size_recommendation, hardware_target, memory_strategy,
              reported_speedup, kernel_code_blocks
        """
        abstract = paper_metadata.get("abstract", "")
        title = paper_metadata.get("title", "")
        full_text = paper_metadata.get("full_text", "")
        paper_id = paper_metadata.get("arxiv_id") or paper_metadata.get("paper_id", "")

        combined = f"{title}. {abstract}"
        search_text = f"{combined} {full_text}" if full_text else combined

        # Extract fields
        target_op = self._extract_target_operation(combined)
        block_size = self._extract_block_size(search_text)
        hardware = self._extract_hardware_target(search_text)
        memory_strategy = self._extract_memory_strategy(search_text)
        speedup = self._extract_reported_speedup(search_text)
        code_blocks = self._extract_code_blocks(search_text)

        technique_name = self._derive_technique_name(title, abstract, target_op)
        technique_id = f"ktech_{paper_id}_{uuid.uuid4().hex[:8]}"

        return {
            "technique_id": technique_id,
            "paper_id": paper_id,
            "name": technique_name,
            "target_operation": target_op,
            "block_size_recommendation": block_size,
            "hardware_target": hardware,
            "memory_strategy": memory_strategy,
            "reported_speedup": speedup,
            "kernel_code_blocks": code_blocks,
            "modification_category": "kernel",
        }

    def generate_kernel_diffs(
        self,
        technique: dict,
        train_source: str,
    ) -> list[dict]:
        """Generate template-based Triton diffs for a kernel technique.

        Produces 3 variants, each with a different block size.

        Parameters
        ----------
        technique : dict
            Technique dict from extract_kernel_technique().
        train_source : str
            Current train.py source code.

        Returns
        -------
        list[dict]
            List of SyntheticDiff-like dicts, each containing:
            - diff_id, technique_id, paper_id, variant_index, diff_text,
              modification_category, kernel_source, block_size
        """
        if not train_source or not train_source.strip():
            return []

        technique_id = technique.get("technique_id", "")
        paper_id = technique.get("paper_id", "")
        target_op = technique.get("target_operation", "custom_op")
        technique_name = technique.get("name", "paper_kernel")
        recommended_bs = technique.get("block_size_recommendation", 0)

        # Choose block size variants: center on recommendation if available
        if recommended_bs and recommended_bs in _BLOCK_SIZE_VARIANTS:
            variants = _BLOCK_SIZE_VARIANTS
        elif recommended_bs and recommended_bs > 0:
            # Use the recommended size and two neighbors (power-of-2 adjusted)
            log2 = int(math.log2(max(recommended_bs, 32)))
            variants = [2 ** (log2 - 1), 2 ** log2, 2 ** (log2 + 1)]
            variants = [max(32, min(v, 2048)) for v in variants]
        else:
            variants = _BLOCK_SIZE_VARIANTS

        # Deduplicate while preserving order
        seen = set()
        unique_variants = []
        for v in variants:
            if v not in seen:
                seen.add(v)
                unique_variants.append(v)
        variants = unique_variants

        results = []
        safe_op = re.sub(r"[^a-zA-Z0-9_]", "_", target_op)
        kernel_fn_name = f"triton_{safe_op}_kernel"
        kernel_module = f"kernels.{safe_op}_kernel"

        for variant_idx, block_size in enumerate(variants):
            # Generate kernel source
            kernel_source = _TRITON_KERNEL_TEMPLATE.format(
                target_op=target_op,
                technique_name=technique_name,
                kernel_fn_name=kernel_fn_name,
                block_size=block_size,
            )

            # Generate integration diff
            integration_diff = _INTEGRATION_TEMPLATE.format(
                kernel_module=kernel_module,
                kernel_fn_name=kernel_fn_name,
            )

            # Combine kernel file + integration diff
            diff_text = (
                f"# === New file: kernels/{safe_op}_kernel.py ===\n"
                f"{kernel_source}\n"
                f"# === Patch: train.py ===\n"
                f"{integration_diff}"
            )

            diff_id = f"kdiff_{technique_id}_{variant_idx}_{uuid.uuid4().hex[:6]}"

            results.append({
                "diff_id": diff_id,
                "technique_id": technique_id,
                "paper_id": paper_id,
                "variant_index": variant_idx,
                "diff_text": diff_text,
                "modification_category": "kernel",
                "kernel_source": kernel_source,
                "block_size": block_size,
            })

        return results

    def fast_correctness_prescreen(
        self,
        diff: dict,
        train_source: str,
    ) -> bool:
        """Quick syntax check on generated Triton code.

        Checks that the kernel source contains essential Triton constructs
        and compiles as valid Python syntax.

        Parameters
        ----------
        diff : dict
            A diff dict from generate_kernel_diffs().
        train_source : str
            Current train.py source (used for context, not modified).

        Returns
        -------
        bool
            True if the kernel source passes basic syntax checks.
        """
        kernel_source = diff.get("kernel_source", "")
        if not kernel_source:
            return False

        # Check essential Triton constructs are present
        for essential in _TRITON_SYNTAX_ESSENTIALS:
            if essential not in kernel_source:
                logger.debug("Prescreen fail: missing '%s'", essential)
                return False

        # Check Python syntax validity
        try:
            compile(kernel_source, "<kernel>", "exec")
        except SyntaxError as exc:
            logger.debug("Prescreen fail: syntax error: %s", exc)
            return False

        return True

    # ------------------------------------------------------------------
    # Internal extraction helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_target_operation(text: str) -> str:
        m = _TARGET_OP_PATTERN.search(text)
        return m.group(1).lower().strip() if m else ""

    @staticmethod
    def _extract_block_size(text: str) -> int:
        m = _BLOCK_SIZE_PATTERN.search(text)
        if m:
            val = int(m.group(1))
            if 32 <= val <= 4096:
                return val
        return 0

    @staticmethod
    def _extract_hardware_target(text: str) -> str:
        m = _HARDWARE_PATTERN.search(text)
        return m.group(1).upper().strip() if m else ""

    @staticmethod
    def _extract_memory_strategy(text: str) -> str:
        m = _MEMORY_STRATEGY_PATTERN.search(text)
        return m.group(1).lower().strip() if m else ""

    @staticmethod
    def _extract_reported_speedup(text: str) -> float:
        # Try "NxX speedup" pattern first
        m = _SPEEDUP_PATTERN.search(text)
        if m:
            return float(m.group(1))
        # Try "N% improvement" pattern
        m = _SPEEDUP_PERCENT_PATTERN.search(text)
        if m:
            pct = float(m.group(1))
            return round(1.0 + pct / 100.0, 2)
        return 0.0

    @staticmethod
    def _extract_code_blocks(text: str) -> list[dict]:
        blocks = []
        for m in _CODE_BLOCK_PATTERN.finditer(text):
            code = m.group(1).strip()
            if not code:
                continue
            lang = "unknown"
            if _TRITON_INDICATOR.search(code):
                lang = "triton"
            elif _CUDA_INDICATOR.search(code):
                lang = "cuda"
            else:
                lang = "python"
            blocks.append({"language": lang, "code": code})
        return blocks

    @staticmethod
    def _derive_technique_name(title: str, abstract: str, target_op: str) -> str:
        """Derive a short technique name from title/abstract."""
        if title:
            # Use first clause of title, truncated
            name = title.split(":")[0].split("—")[0].strip()
            if len(name) > 80:
                name = name[:80].rsplit(" ", 1)[0]
            return name
        if target_op:
            return f"Kernel optimization for {target_op}"
        # Fallback
        if abstract:
            first_sentence = abstract.split(".")[0].strip()
            if len(first_sentence) > 80:
                first_sentence = first_sentence[:80].rsplit(" ", 1)[0]
            return first_sentence
        return "Unknown kernel technique"
