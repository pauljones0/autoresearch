"""
Phase 3 — KernelPaperIngestion: kernel-specific paper relevance scoring
and diagnostics-informed ingestion for the surrogate triage pipeline.
"""

import logging
import re

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Kernel-specific keyword weights
# ---------------------------------------------------------------------------

KERNEL_KEYWORDS: dict[str, float] = {
    "triton": 1.2,
    "cuda kernel": 1.2,
    "gpu optimization": 1.0,
    "kernel fusion": 1.2,
    "flash attention": 1.0,
    "memory coalescing": 0.8,
    "shared memory": 0.8,
    "warp shuffle": 0.7,
    "operator fusion": 1.0,
    "custom kernel": 1.0,
    "tiled computation": 0.8,
    "block size": 0.7,
    "occupancy": 0.7,
}

# ---------------------------------------------------------------------------
# Bottleneck-type to search term mappings
# ---------------------------------------------------------------------------

KERNEL_BOTTLENECK_MAPPINGS: dict[str, list[str]] = {
    "memory_bandwidth_underutilization": [
        "memory access pattern",
        "coalesced access",
        "memory bandwidth",
        "data layout",
        "memory tiling",
        "vectorized load",
    ],
    "low_sm_occupancy": [
        "occupancy optimization",
        "register pressure",
        "shared memory allocation",
        "thread block size",
        "warp scheduling",
        "register spilling",
    ],
    "kernel_launch_overhead": [
        "kernel fusion",
        "operator fusion",
        "graph compilation",
        "persistent kernel",
        "cuda graph",
        "launch latency",
    ],
    "elementwise_bottleneck": [
        "elementwise fusion",
        "pointwise kernel",
        "activation fusion",
        "vectorized elementwise",
        "fused activation",
    ],
    "reduction_bottleneck": [
        "parallel reduction",
        "warp reduction",
        "tree reduction",
        "cooperative groups",
        "multi-stage reduction",
    ],
    "attention_bottleneck": [
        "flash attention",
        "memory efficient attention",
        "tiled attention",
        "online softmax",
        "block sparse attention",
    ],
}

# ---------------------------------------------------------------------------
# Additional arXiv categories to fetch for kernel papers
# ---------------------------------------------------------------------------

KERNEL_ARXIV_CATEGORIES: list[str] = ["cs.DC", "cs.AR"]

# ---------------------------------------------------------------------------
# Precompiled keyword patterns
# ---------------------------------------------------------------------------

_COMPILED_KERNEL_PATTERNS: list[tuple[re.Pattern, float]] = [
    (re.compile(r"\b" + re.escape(kw) + r"\b", re.IGNORECASE), weight)
    for kw, weight in KERNEL_KEYWORDS.items()
]


class KernelPaperIngestion:
    """Kernel-specific paper relevance scoring and ingestion control.

    Extends the surrogate triage paper filtering with kernel-domain knowledge,
    mapping GPU profiling bottlenecks to paper search terms and providing
    kernel-specific relevance scores.
    """

    def score_kernel_relevance(
        self,
        abstract: str,
        fuseable_groups: list = None,
    ) -> float:
        """Compute a kernel-specific relevance score for a paper abstract.

        Parameters
        ----------
        abstract : str
            The paper abstract text.
        fuseable_groups : list, optional
            List of FuseableGroup-like dicts/objects from GPU profiling.

        Returns
        -------
        float
            Kernel relevance score (sum of matched keyword weights plus
            any diagnostics-informed boost).
        """
        if not abstract:
            return 0.0

        base_score = self._keyword_score(abstract)
        boost = self.compute_kernel_boost(abstract, fuseable_groups)
        return base_score + boost

    def should_fetch_system_papers(self, fuseable_groups: list) -> bool:
        """Determine whether system/architecture papers should be fetched.

        Returns True if any operation in the fuseable groups has a
        bandwidth_utilization below 0.3, indicating significant room for
        kernel-level optimization.

        Parameters
        ----------
        fuseable_groups : list
            List of FuseableGroup-like dicts/objects.

        Returns
        -------
        bool
        """
        if not fuseable_groups:
            return False

        for group in fuseable_groups:
            # Support both dict and dataclass access
            if isinstance(group, dict):
                ops = group.get("op_profiles", [])
                # Also check top-level bandwidth if present
                bw = group.get("bandwidth_utilization", None)
                if bw is not None and bw < 0.3:
                    return True
            else:
                ops = getattr(group, "op_profiles", [])

            for op in ops:
                if isinstance(op, dict):
                    bw = op.get("bandwidth_utilization", 1.0)
                else:
                    bw = getattr(op, "bandwidth_utilization", 1.0)
                if bw < 0.3:
                    return True

        return False

    def compute_kernel_boost(
        self,
        abstract: str,
        fuseable_groups: list = None,
    ) -> float:
        """Compute a diagnostics-informed boost for kernel papers.

        The boost is based on matching bottleneck search terms against the
        abstract, weighted by how severe the bottleneck is (inferred from
        fuseable group diagnostics).

        Parameters
        ----------
        abstract : str
            The paper abstract text.
        fuseable_groups : list, optional
            List of FuseableGroup-like dicts/objects with profiling data.

        Returns
        -------
        float
            Boost value (0.0 if no fuseable groups or no matches).
        """
        if not abstract or not fuseable_groups:
            return 0.0

        abstract_lower = abstract.lower()
        boost = 0.0

        # Detect which bottleneck types are active
        active_bottlenecks = self._detect_bottlenecks(fuseable_groups)

        for bottleneck_type, severity in active_bottlenecks.items():
            search_terms = KERNEL_BOTTLENECK_MAPPINGS.get(bottleneck_type, [])
            for term in search_terms:
                if term.lower() in abstract_lower:
                    boost += severity * 0.3
                    break  # one match per bottleneck type

        return round(boost, 4)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _keyword_score(abstract: str) -> float:
        """Sum matched keyword weights."""
        abstract_lower = abstract.lower()
        total = 0.0
        for pattern, weight in _COMPILED_KERNEL_PATTERNS:
            if pattern.search(abstract_lower):
                total += weight
        return total

    @staticmethod
    def _detect_bottlenecks(fuseable_groups: list) -> dict[str, float]:
        """Detect active bottleneck types from fuseable groups.

        Returns a dict of bottleneck_type -> severity (0.0-1.0).
        """
        bottlenecks: dict[str, float] = {}

        for group in fuseable_groups:
            if isinstance(group, dict):
                ops = group.get("op_profiles", [])
                fusion_type = group.get("fusion_type", "")
            else:
                ops = getattr(group, "op_profiles", [])
                fusion_type = getattr(group, "fusion_type", "")

            for op in ops:
                if isinstance(op, dict):
                    bw = op.get("bandwidth_utilization", 1.0)
                    occ = op.get("sm_occupancy", 1.0)
                else:
                    bw = getattr(op, "bandwidth_utilization", 1.0)
                    occ = getattr(op, "sm_occupancy", 1.0)

                if bw < 0.5:
                    severity = 1.0 - bw
                    current = bottlenecks.get("memory_bandwidth_underutilization", 0.0)
                    bottlenecks["memory_bandwidth_underutilization"] = max(current, severity)

                if occ < 0.5:
                    severity = 1.0 - occ
                    current = bottlenecks.get("low_sm_occupancy", 0.0)
                    bottlenecks["low_sm_occupancy"] = max(current, severity)

            # Fusion type hints
            if fusion_type == "elementwise":
                bottlenecks.setdefault("elementwise_bottleneck", 0.5)
            elif fusion_type == "attention":
                bottlenecks.setdefault("attention_bottleneck", 0.5)
            elif fusion_type == "reduction":
                bottlenecks.setdefault("reduction_bottleneck", 0.5)

        return bottlenecks
