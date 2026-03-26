"""
Aggregate verification results into a unified report.
"""

import os
import time

from gpu_kernels.schemas import (
    CorrectnessResult,
    StabilityResult,
    DivergenceResult,
    KernelVerificationReport,
)


class CorrectnessReportGenerator:
    """Generate an aggregated verification report from correctness, stability, and divergence results."""

    def __init__(self, reports_dir: str = None):
        if reports_dir is None:
            reports_dir = os.path.join(os.path.dirname(__file__), "..", "reports")
        self.reports_dir = reports_dir

    def generate(
        self,
        correctness: CorrectnessResult,
        stability: StabilityResult,
        divergence: DivergenceResult,
        kernel_id: str,
        group_id: str = "",
    ) -> KernelVerificationReport:
        """Aggregate verification results and produce a verdict.

        Verdict logic:
            PASS: all three checks pass.
            FAIL: any of correctness or stability fail, or divergence shows
                  max_loss_divergence >= 0.1% of mean loss.
            CONDITIONAL_PASS: correctness and stability pass, divergence shows
                  marginal drift (0.05%-0.1% of mean loss).

        Args:
            correctness: Result from KernelCorrectnessVerifier.
            stability: Result from NumericalStabilityProber.
            divergence: Result from TrainingDivergenceDetector.
            kernel_id: Unique kernel identifier.
            group_id: Associated fuseable group ID.

        Returns:
            KernelVerificationReport saved to disk.
        """
        notes = []

        # Determine sub-verdicts
        correctness_ok = correctness.passed
        stability_ok = stability.is_deterministic and not stability.overflow_detected

        # Divergence assessment
        divergence_ok = divergence.passed
        marginal_drift = False
        if divergence.loss_curve_reference:
            mean_loss = sum(abs(l) for l in divergence.loss_curve_reference) / len(divergence.loss_curve_reference)
            if mean_loss > 0:
                relative_div = divergence.max_loss_divergence / mean_loss
                if 0.0005 <= relative_div < 0.001:
                    marginal_drift = True
                    notes.append(
                        f"Marginal loss drift detected: {relative_div:.6f} relative divergence"
                    )

        # Overall verdict
        if correctness_ok and stability_ok and divergence_ok:
            if marginal_drift:
                verdict = "CONDITIONAL_PASS"
                notes.append("All checks pass but divergence shows marginal drift")
            else:
                verdict = "PASS"
        elif correctness_ok and stability_ok and marginal_drift:
            verdict = "CONDITIONAL_PASS"
        else:
            verdict = "FAIL"
            if not correctness_ok:
                notes.append(f"Correctness failed: {correctness.passed_configs}/{correctness.tested_configs} configs passed")
            if not stability_ok:
                if not stability.is_deterministic:
                    notes.append("Kernel is non-deterministic")
                if stability.overflow_detected:
                    notes.append("Overflow detected with extreme inputs")
            if not divergence_ok and not marginal_drift:
                notes.append(f"Training divergence detected at step {divergence.diverged_at_step}")

        report = KernelVerificationReport(
            kernel_id=kernel_id,
            group_id=group_id,
            correctness=correctness.to_dict(),
            stability=stability.to_dict(),
            divergence=divergence.to_dict(),
            verdict=verdict,
            timestamp=time.time(),
            notes=notes,
        )

        # Save to disk
        os.makedirs(self.reports_dir, exist_ok=True)
        report_path = os.path.join(self.reports_dir, f"{kernel_id}_verification.json")
        report.to_json(report_path)

        return report
