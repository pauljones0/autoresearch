"""
Phase 3 — KernelScoringExtension: kernel-specific surrogate features,
candidate routing with verification gating, and scheduling fraction control.
"""

import logging
import math

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# The 8 kernel-specific features for surrogate enrichment
# ---------------------------------------------------------------------------
# Index | Feature name                       | Range
# ------+------------------------------------+---------
#   0   | target_op_gpu_time_fraction         | [0, 1]
#   1   | target_op_bandwidth_utilization     | [0, 1]
#   2   | fusion_degree                       | [1, ∞)  (log2 scaled to ~[0,4])
#   3   | has_paper_code                      | {0, 1}
#   4   | block_size_log2                     | ~[5,12] (log2 of block size)
#   5   | hardware_match                      | {0, 1}
#   6   | correctness_prescreened             | {0, 1}
#   7   | estimated_speedup_from_paper        | [0, ∞)  (clamped to 10)

KERNEL_FEATURE_NAMES = [
    "target_op_gpu_time_fraction",
    "target_op_bandwidth_utilization",
    "fusion_degree",
    "has_paper_code",
    "block_size_log2",
    "hardware_match",
    "correctness_prescreened",
    "estimated_speedup_from_paper",
]

_DEFAULT_KERNEL_SCHEDULING_FRACTION = 0.15


class KernelScoringExtension:
    """Kernel-specific feature enrichment, candidate routing, and scheduling.

    Provides 8 kernel-specific features that can be appended to the surrogate
    model's feature vector, along with verification-gated routing and
    scheduling fraction control.
    """

    def enrich_kernel_features(
        self,
        diff: dict,
        diagnostics: dict = None,
        hardware: dict = None,
    ) -> list[float]:
        """Compute the 8 kernel-specific features for a diff candidate.

        Parameters
        ----------
        diff : dict
            A kernel diff dict (from KernelPaperExtractor.generate_kernel_diffs).
            Expected keys: block_size, kernel_source, technique metadata.
        diagnostics : dict, optional
            GPU profiling diagnostics with per-op timing and bandwidth data.
            Expected keys: total_gpu_time_us, op_profiles (list of dicts with
            op_name, gpu_time_us, bandwidth_utilization).
        hardware : dict, optional
            Current hardware profile dict. Expected keys: gpu_name.

        Returns
        -------
        list[float]
            8-element feature vector.
        """
        features = [0.0] * 8

        # Feature 0: target_op_gpu_time_fraction
        features[0] = self._compute_gpu_time_fraction(diff, diagnostics)

        # Feature 1: target_op_bandwidth_utilization
        features[1] = self._compute_bandwidth_utilization(diff, diagnostics)

        # Feature 2: fusion_degree (log2 of number of ops fused)
        features[2] = self._compute_fusion_degree(diff)

        # Feature 3: has_paper_code (binary)
        features[3] = self._compute_has_paper_code(diff)

        # Feature 4: block_size_log2
        block_size = diff.get("block_size", 256)
        features[4] = math.log2(max(block_size, 1))

        # Feature 5: hardware_match (binary)
        features[5] = self._compute_hardware_match(diff, hardware)

        # Feature 6: correctness_prescreened (binary)
        features[6] = 1.0 if diff.get("correctness_prescreened", False) else 0.0

        # Feature 7: estimated_speedup_from_paper
        speedup = diff.get("reported_speedup", 0.0)
        if not speedup:
            # Try nested technique data
            technique = diff.get("technique", {})
            if isinstance(technique, dict):
                speedup = technique.get("reported_speedup", 0.0)
        features[7] = min(float(speedup), 10.0)

        return features

    def route_kernel_candidate(
        self,
        candidate: dict,
        base_source: str,
        verification_pipeline=None,
        model_scientist_pipeline=None,
    ) -> dict:
        """Route a kernel candidate through verification, then GPU evaluation.

        Runs verification first; if verification fails, skips GPU evaluation
        to save compute.

        Parameters
        ----------
        candidate : dict
            Kernel candidate with diff_text, kernel_source, technique metadata.
        base_source : str
            Current train.py source code.
        verification_pipeline : object, optional
            Kernel verification pipeline with a verify(kernel_source, ...) method.
            Must return a dict with 'verdict' field.
        model_scientist_pipeline : object, optional
            Model scientist pipeline with evaluate_modification(...) method.

        Returns
        -------
        dict
            Evaluation result with verdict, verification_result, and optionally
            gpu_eval_result.
        """
        candidate_id = candidate.get("diff_id", "unknown")
        kernel_source = candidate.get("kernel_source", "")

        result = {
            "candidate_id": candidate_id,
            "verdict": "pending",
            "verification_result": None,
            "gpu_eval_result": None,
        }

        # Step 1: Verification
        if verification_pipeline is not None:
            try:
                verification = verification_pipeline.verify(
                    kernel_source=kernel_source,
                    base_source=base_source,
                )
                result["verification_result"] = verification

                verdict = verification.get("verdict", "FAIL")
                if isinstance(verdict, str):
                    verdict = verdict.upper()

                if verdict not in ("PASS", "CONDITIONAL_PASS"):
                    result["verdict"] = "verification_failed"
                    logger.info(
                        "Kernel candidate %s failed verification: %s",
                        candidate_id, verdict,
                    )
                    return result
            except Exception as exc:
                logger.error(
                    "Verification crashed for candidate %s: %s",
                    candidate_id, exc,
                )
                result["verdict"] = "verification_crashed"
                result["error"] = str(exc)
                return result
        else:
            logger.debug(
                "No verification pipeline for candidate %s, skipping",
                candidate_id,
            )

        # Step 2: GPU evaluation (only if verification passed or was skipped)
        if model_scientist_pipeline is not None:
            diff_text = candidate.get("diff_text", "")
            hypothesis = (
                f"Kernel technique: {candidate.get('name', 'unknown')} "
                f"(paper: {candidate.get('paper_id', 'unknown')})"
            )
            tags = [
                "source:kernel_paper",
                f"paper_id:{candidate.get('paper_id', '')}",
                f"block_size:{candidate.get('block_size', 0)}",
            ]

            try:
                gpu_result = model_scientist_pipeline.evaluate_modification(
                    modified_source=base_source,
                    hypothesis=hypothesis,
                    predicted_delta=candidate.get("surrogate_score", 0.0),
                    modification_diff=diff_text,
                    tags=tags,
                )
                result["gpu_eval_result"] = gpu_result
                result["verdict"] = gpu_result.get("verdict", "evaluated")
            except Exception as exc:
                logger.error(
                    "GPU evaluation crashed for candidate %s: %s",
                    candidate_id, exc,
                )
                result["verdict"] = "gpu_eval_crashed"
                result["error"] = str(exc)
                return result
        else:
            # Dry run — verification passed but no GPU eval available
            result["verdict"] = "verification_passed_dry_run"

        return result

    def get_kernel_scheduling_fraction(
        self,
        queue_entries: list,
    ) -> float:
        """Compute the fraction of next iterations to allocate to kernel candidates.

        Parameters
        ----------
        queue_entries : list
            Current evaluation queue entries (dicts with 'modification_category'
            and optionally 'surrogate_score').

        Returns
        -------
        float
            Fraction of iterations for kernel candidates (default 0.15).
        """
        if not queue_entries:
            return _DEFAULT_KERNEL_SCHEDULING_FRACTION

        kernel_count = 0
        total_count = len(queue_entries)
        high_score_kernel_count = 0

        for entry in queue_entries:
            cat = ""
            if isinstance(entry, dict):
                cat = entry.get("modification_category", "")
                score = entry.get("surrogate_score", 0.0)
            else:
                cat = getattr(entry, "modification_category", "")
                score = getattr(entry, "surrogate_score", 0.0)

            if cat == "kernel":
                kernel_count += 1
                if score > 0.5:
                    high_score_kernel_count += 1

        if kernel_count == 0:
            return _DEFAULT_KERNEL_SCHEDULING_FRACTION

        # Adjust fraction based on queue composition:
        # - More high-scoring kernel candidates → increase fraction (up to 0.30)
        # - Few kernel candidates → stay at default
        base = _DEFAULT_KERNEL_SCHEDULING_FRACTION
        if total_count > 0 and high_score_kernel_count > 0:
            quality_ratio = high_score_kernel_count / total_count
            # Scale from 0.15 to 0.30 based on quality
            adjusted = base + quality_ratio * 0.15
            return min(round(adjusted, 3), 0.30)

        return base

    # ------------------------------------------------------------------
    # Internal feature computation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_gpu_time_fraction(diff: dict, diagnostics: dict) -> float:
        """Fraction of total GPU time spent on the target operation."""
        if not diagnostics:
            return 0.0

        target_op = diff.get("target_operation", "")
        if not target_op:
            technique = diff.get("technique", {})
            if isinstance(technique, dict):
                target_op = technique.get("target_operation", "")

        if not target_op:
            return 0.0

        total_time = diagnostics.get("total_gpu_time_us", 0.0)
        if total_time <= 0:
            return 0.0

        target_lower = target_op.lower()
        op_profiles = diagnostics.get("op_profiles", [])
        matched_time = 0.0

        for op in op_profiles:
            op_name = op.get("op_name", "").lower() if isinstance(op, dict) else ""
            if target_lower in op_name:
                matched_time += op.get("gpu_time_us", 0.0) if isinstance(op, dict) else 0.0

        return min(matched_time / total_time, 1.0)

    @staticmethod
    def _compute_bandwidth_utilization(diff: dict, diagnostics: dict) -> float:
        """Bandwidth utilization of the target operation."""
        if not diagnostics:
            return 0.0

        target_op = diff.get("target_operation", "")
        if not target_op:
            technique = diff.get("technique", {})
            if isinstance(technique, dict):
                target_op = technique.get("target_operation", "")

        if not target_op:
            return 0.0

        target_lower = target_op.lower()
        op_profiles = diagnostics.get("op_profiles", [])

        for op in op_profiles:
            if not isinstance(op, dict):
                continue
            op_name = op.get("op_name", "").lower()
            if target_lower in op_name:
                return op.get("bandwidth_utilization", 0.0)

        return 0.0

    @staticmethod
    def _compute_fusion_degree(diff: dict) -> float:
        """Log2 of the number of operations being fused."""
        # Check for fusion info in technique metadata
        technique = diff.get("technique", {})
        if isinstance(technique, dict):
            n_ops = technique.get("n_fused_ops", 1)
        else:
            n_ops = diff.get("n_fused_ops", 1)
        return math.log2(max(int(n_ops), 1))

    @staticmethod
    def _compute_has_paper_code(diff: dict) -> float:
        """1.0 if the source paper included code blocks."""
        technique = diff.get("technique", {})
        if isinstance(technique, dict):
            code_blocks = technique.get("kernel_code_blocks", [])
        else:
            code_blocks = diff.get("kernel_code_blocks", [])
        return 1.0 if code_blocks else 0.0

    @staticmethod
    def _compute_hardware_match(diff: dict, hardware: dict) -> float:
        """1.0 if the paper's target hardware matches current hardware."""
        if not hardware:
            return 0.0

        technique = diff.get("technique", {})
        if isinstance(technique, dict):
            paper_hw = technique.get("hardware_target", "")
        else:
            paper_hw = diff.get("hardware_target", "")

        if not paper_hw:
            return 0.0

        current_gpu = hardware.get("gpu_name", "")
        if not current_gpu:
            return 0.0

        # Normalize and compare
        paper_hw_upper = paper_hw.upper().replace(" ", "")
        current_upper = current_gpu.upper().replace(" ", "")

        return 1.0 if paper_hw_upper in current_upper or current_upper in paper_hw_upper else 0.0
