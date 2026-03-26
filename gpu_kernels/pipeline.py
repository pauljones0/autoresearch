"""
GPU Kernel Creation Pipeline — Full orchestrator.

Chains all five phases together:
  1. Profile training run → identify optimization targets → build reference catalog
  2. Generate Triton kernels → verify correctness → benchmark → integrate
  3. Ingest kernel papers → extract techniques → score → route through pipeline
  4. Autonomous discovery of new kernel targets → evolutionary refinement
  5. Extended validation → runtime monitoring → auto-rollback → dashboard

Usage:
    from gpu_kernels.pipeline import GPUKernelPipeline

    pipeline = GPUKernelPipeline(train_path="train.py")
    pipeline.initialize()

    # Profile current model:
    targets = pipeline.run_profiling_cycle(model, sample_batch)

    # Generate and test kernels for top targets:
    results = pipeline.run_kernel_generation(targets)

    # Autonomous discovery from diagnostics:
    discovery = pipeline.run_kernel_discovery(diagnostics_report)

    # Evolutionary refinement of existing kernel:
    evolution = pipeline.run_evolutionary_refinement(kernel_id)

    # Paper-sourced kernel pipeline:
    paper_results = pipeline.run_paper_kernel_pipeline(paper_techniques)

    # Extended validation of all active kernels:
    validation = pipeline.run_extended_validation()
"""

import logging

logger = logging.getLogger(__name__)

import os
import time
import json

from .schemas import (
    HardwareProfile, OperationProfile, FuseableGroup,
    KernelConfigEntry, KernelVerificationReport,
    BenchmarkResult, DiscoveryCycleResult, GenerationResult,
    ExtendedValidationResult, load_json, save_json,
)

# Phase 1.1: Profiling
from .profiling.instrumenter import GPUProfilingInstrumenter
from .profiling.bandwidth import BandwidthUtilizationCalculator
from .profiling.fuseable_detector import FuseableOperationDetector

# Phase 1.1: Hardware
from .benchmarking.hardware import HardwareCapabilityDetector

# Phase 1.2: Reference Catalog
from .reference_catalog.extractor import ReferenceCodeExtractor
from .reference_catalog.shape_documenter import TensorShapeDocumenter
from .reference_catalog.tolerance_calibrator import ToleranceBoundCalibrator
from .reference_catalog.test_generator import TestInputGenerator

# Phase 1.3: Verification
from .verification.correctness import KernelCorrectnessVerifier
from .verification.stability import NumericalStabilityProber
from .verification.divergence import TrainingDivergenceDetector
from .verification.report import CorrectnessReportGenerator

# Phase 1.4: Benchmarking
from .benchmarking.benchmarker import KernelBenchmarker
from .benchmarking.bandwidth_profiler import MemoryBandwidthProfiler
from .benchmarking.throughput import TrainingThroughputEstimator

# Phase 2: Generation
from .generation.elementwise_selector import ElementwiseTargetSelector
from .generation.elementwise_generator import TritonElementwiseGenerator
from .generation.verification_runner import ElementwiseVerificationRunner
from .generation.integrator import KernelIntegrator
from .generation.optimizer_analyzer import OptimizerAnalyzer
from .generation.optimizer_generator import TritonOptimizerFusionGenerator
from .generation.normalization_generator import NormalizationKernelGenerator
from .generation.attention_analyzer import AttentionArchitectureAnalyzer
from .generation.attention_generator import AttentionKernelGenerator

# Phase 3: Paper-sourced
from .generation.paper_kernel_ingestion import KernelPaperIngestion
from .generation.paper_kernel_extraction import KernelPaperExtractor
from .generation.paper_kernel_scoring import KernelScoringExtension

# Phase 4: Discovery & Evolution
from .discovery.opportunity_ranker import KernelOpportunityRanker
from .discovery.autonomous_generator import AutonomousKernelGenerator
from .discovery.loop_orchestrator import DiscoveryLoopOrchestrator
from .evolution.mutator import KernelMutationEngine
from .evolution.selector import EvolutionarySelectionController
from .evolution.convergence import EvolutionConvergenceDetector
from .evolution.scheduler import EvolutionaryRefinementScheduler

# Phase 4.3: Metrics
from .metrics.failure_taxonomy import KernelFailureTaxonomist
from .metrics.constraint_specialist import KernelConstraintSpecialist
from .metrics.kernel_metric_proposer import KernelMetricProposer

# Phase 5: Production
from .validation.extended_divergence import ExtendedDivergenceValidator
from .validation.checkpoint_compat import GradientCheckpointCompatibilityTester
from .validation.mixed_precision import MixedPrecisionStressTester
from .monitoring.runtime_monitor import RuntimeCorrectnessMonitor
from .monitoring.fallback_verifier import FallbackIntegrityVerifier
from .monitoring.auto_recovery import KernelAutoRecoveryAgent
from .monitoring.dashboard import KernelDashboard
from .monitoring.health_auditor import CrossSystemHealthAuditor
from .config.manager import KernelConfigManager


class GPUKernelPipeline:
    """Top-level orchestrator for the GPU Kernel Creation Pipeline."""

    def __init__(
        self,
        train_path: str = "train.py",
        data_dir: str = None,
        kernel_dir: str = None,
    ):
        self.train_path = os.path.abspath(train_path)
        self.data_dir = data_dir or os.path.join(
            os.path.dirname(self.train_path), "data"
        )
        self.kernel_dir = kernel_dir or os.path.join(
            os.path.dirname(self.train_path), "gpu_kernels"
        )
        os.makedirs(self.data_dir, exist_ok=True)

        with open(self.train_path, encoding="utf-8") as f:
            self._base_source = f.read()

        # Phase 1.1: Profiling
        self.profiler = GPUProfilingInstrumenter()
        self.bandwidth_calc = BandwidthUtilizationCalculator()
        self.fuseable_detector = FuseableOperationDetector()
        self.hardware_detector = HardwareCapabilityDetector()

        # Phase 1.2: Reference Catalog
        self.reference_extractor = ReferenceCodeExtractor()
        self.shape_documenter = TensorShapeDocumenter()
        self.tolerance_calibrator = ToleranceBoundCalibrator()
        self.test_generator = TestInputGenerator()

        # Phase 1.3: Verification
        self.correctness_verifier = KernelCorrectnessVerifier()
        self.stability_prober = NumericalStabilityProber()
        self.divergence_detector = TrainingDivergenceDetector()
        self.report_generator = CorrectnessReportGenerator()

        # Phase 1.4: Benchmarking
        self.benchmarker = KernelBenchmarker()
        self.bandwidth_profiler = MemoryBandwidthProfiler()
        self.throughput_estimator = TrainingThroughputEstimator()

        # Phase 2: Generation
        self.elementwise_selector = ElementwiseTargetSelector()
        self.elementwise_generator = TritonElementwiseGenerator()
        self.elementwise_runner = ElementwiseVerificationRunner()
        self.integrator = KernelIntegrator()
        self.optimizer_analyzer = OptimizerAnalyzer()
        self.optimizer_generator = TritonOptimizerFusionGenerator()
        self.normalization_generator = NormalizationKernelGenerator()
        self.attention_analyzer = AttentionArchitectureAnalyzer()
        self.attention_generator = AttentionKernelGenerator()

        # Phase 3: Paper-sourced
        self.paper_ingestion = KernelPaperIngestion()
        self.paper_extractor = KernelPaperExtractor()
        self.paper_scoring = KernelScoringExtension()

        # Phase 4: Discovery & Evolution
        self.opportunity_ranker = KernelOpportunityRanker()
        self.autonomous_generator = AutonomousKernelGenerator()
        self.discovery_orchestrator = DiscoveryLoopOrchestrator()
        self.mutation_engine = KernelMutationEngine()
        self.selection_controller = EvolutionarySelectionController()
        self.convergence_detector = EvolutionConvergenceDetector()
        self.refinement_scheduler = EvolutionaryRefinementScheduler()

        # Phase 4.3: Metrics
        self.failure_taxonomist = KernelFailureTaxonomist()
        self.constraint_specialist = KernelConstraintSpecialist()
        self.metric_proposer = KernelMetricProposer()

        # Phase 5: Production
        self.extended_validator = ExtendedDivergenceValidator()
        self.checkpoint_tester = GradientCheckpointCompatibilityTester()
        self.mixed_precision_tester = MixedPrecisionStressTester()
        self.runtime_monitor = RuntimeCorrectnessMonitor()
        self.config_manager = KernelConfigManager(
            config_dir=self.data_dir
        )
        self.fallback_verifier = FallbackIntegrityVerifier()
        self.auto_recovery = KernelAutoRecoveryAgent()
        self.dashboard = KernelDashboard(data_dir=self.data_dir)
        self.health_auditor = CrossSystemHealthAuditor(data_dir=self.data_dir)

        # State
        self._hardware_profile = None
        self._initialized = False

    def initialize(self):
        """Initialize the pipeline. Detect hardware and load config."""
        try:
            self._hardware_profile = self.hardware_detector.detect()
            print(f"[GPUKernels] Hardware: {self._hardware_profile.gpu_name}")
        except Exception as e:
            print(f"[GPUKernels] Hardware detection failed: {e}")
            self._hardware_profile = HardwareProfile()

        self.config_manager.load()
        self._initialized = True
        print("[GPUKernels] Pipeline initialized")

    def run_profiling_cycle(self, model=None, sample_batch=None) -> dict:
        """Run full profiling cycle to identify kernel optimization targets.

        Returns:
            dict with hardware_profile, operation_profiles, fuseable_groups
        """
        result = {
            "hardware_profile": self._hardware_profile,
            "operation_profiles": [],
            "fuseable_groups": [],
        }

        if model is None or sample_batch is None:
            print("[GPUKernels] No model/batch provided — skipping profiling")
            return result

        # Profile operations
        try:
            profiles = self.profiler.profile_step(model, sample_batch)
            result["operation_profiles"] = profiles

            # Calculate bandwidth utilization
            if self._hardware_profile:
                profiles = self.bandwidth_calc.calculate(profiles, self._hardware_profile)

            # Detect fuseable groups
            groups = self.fuseable_detector.detect(profiles)
            result["fuseable_groups"] = groups

            print(f"[GPUKernels] Profiled {len(profiles)} ops, found {len(groups)} fuseable groups")
        except Exception as e:
            print(f"[GPUKernels] Profiling error: {e}")

        return result

    def run_kernel_generation(self, fuseable_groups: list = None) -> list:
        """Generate, verify, and benchmark kernels for the given targets.

        Returns:
            list of dicts with kernel_id, group_id, speedup, verified, integrated
        """
        results = []
        if not fuseable_groups:
            return results

        # Select top elementwise targets
        targets = self.elementwise_selector.select(fuseable_groups, self._base_source)

        for target in targets[:3]:  # top 3
            target_dict = target if isinstance(target, dict) else target.to_dict() if hasattr(target, "to_dict") else {}
            group_id = target_dict.get("group_id", "unknown")
            print(f"[GPUKernels] Generating kernels for {group_id}")

            try:
                # Generate variants
                variants = self.elementwise_generator.generate(target)
                if not variants:
                    continue

                # Run verification + benchmark, select winner
                winner, variant_results = self.elementwise_runner.run(
                    variants, None, None  # reference and inputs loaded from catalog
                )

                if winner:
                    # Integrate
                    config_entry = self.integrator.integrate(
                        winner, group_id, self._base_source
                    )
                    results.append({
                        "kernel_id": winner.kernel_id if hasattr(winner, "kernel_id") else winner.get("kernel_id", ""),
                        "group_id": group_id,
                        "integrated": True,
                    })
            except Exception as e:
                print(f"[GPUKernels] Generation error for {group_id}: {e}")
                results.append({"group_id": group_id, "error": str(e)})

        return results

    def run_kernel_discovery(self, diagnostics_report=None) -> dict:
        """Run autonomous kernel discovery cycle from diagnostics.

        Returns:
            DiscoveryCycleResult dict
        """
        if diagnostics_report is None:
            return {"status": "no_diagnostics"}

        try:
            kernel_config = self.config_manager.get_active_kernels()
            result = self.discovery_orchestrator.run_cycle(
                diagnostics_report, kernel_config, self._base_source
            )
            result_dict = result.to_dict() if hasattr(result, "to_dict") else result
            print(f"[GPUKernels] Discovery: {result_dict.get('variants_generated', 0)} variants, "
                  f"winner={result_dict.get('winner_kernel_id', 'none')}")
            return result_dict
        except Exception as e:
            print(f"[GPUKernels] Discovery error: {e}")
            return {"error": str(e)}

    def run_evolutionary_refinement(self, kernel_id: str = None) -> list:
        """Run evolutionary refinement on an existing kernel.

        If no kernel_id specified, auto-select the best candidate.

        Returns:
            list of GenerationResult dicts
        """
        # Auto-select if not specified
        if kernel_id is None:
            kernel_config = self.config_manager.get_active_kernels()
            kernel_id = self.refinement_scheduler.select(kernel_config, None)
            if kernel_id is None:
                print("[GPUKernels] No kernel suitable for refinement")
                return []

        print(f"[GPUKernels] Evolving kernel: {kernel_id}")
        generation_results = []

        # Get parent source
        config = self.config_manager.get_active_kernels()
        parent_entry = config.get(kernel_id, {})
        parent_path = parent_entry.get("kernel_path", "")

        if not parent_path or not os.path.exists(parent_path):
            print(f"[GPUKernels] Cannot find kernel source for {kernel_id}")
            return []

        try:
            with open(parent_path) as f:
                parent_source = f.read()
        except Exception:
            return []

        parent_id = kernel_id
        for gen in range(10):  # max 10 generations
            # Mutate
            mutations = self.mutation_engine.mutate(parent_source, parent_id)
            if not mutations:
                break

            # Select
            gen_result = self.selection_controller.run_generation(
                parent_id, mutations, kernel_id
            )
            gen_result_dict = gen_result.to_dict() if hasattr(gen_result, "to_dict") else gen_result
            generation_results.append(gen_result_dict)

            # Check convergence
            stop, reason = self.convergence_detector.should_stop(generation_results)
            if stop:
                print(f"[GPUKernels] Evolution converged: {reason}")
                break

            # Update parent for next generation
            best_id = gen_result_dict.get("best_mutation_id", "")
            if best_id:
                for m in mutations:
                    m_dict = m if isinstance(m, dict) else m.to_dict() if hasattr(m, "to_dict") else {}
                    if m_dict.get("mutation_id") == best_id:
                        parent_source = m_dict.get("kernel_source", parent_source)
                        parent_id = best_id
                        break

        return generation_results

    def run_paper_kernel_pipeline(self, paper_techniques: list) -> list:
        """Process paper-sourced kernel techniques through the full pipeline.

        Returns:
            list of evaluation result dicts
        """
        results = []
        for technique in paper_techniques:
            try:
                # Generate kernel diffs
                diffs = self.paper_extractor.generate_kernel_diffs(
                    technique, self._base_source
                )

                # Score and route
                for diff in diffs:
                    result = self.paper_scoring.route_kernel_candidate(
                        diff, self._base_source
                    )
                    results.append(result)
            except Exception as e:
                results.append({"error": str(e)})

        print(f"[GPUKernels] Paper pipeline: {len(results)} candidates evaluated")
        return results

    def run_extended_validation(self, n_steps: int = 2000) -> dict:
        """Run extended validation on all active kernels.

        Returns:
            dict with per-kernel validation results
        """
        kernel_config = self.config_manager.get_active_kernels()
        if not kernel_config:
            return {"status": "no_active_kernels"}

        print(f"[GPUKernels] Running {n_steps}-step extended validation on {len(kernel_config)} kernels")

        try:
            result = self.extended_validator.validate(kernel_config, n_steps=n_steps)
            result_dict = result.to_dict() if hasattr(result, "to_dict") else result
            passed = result_dict.get("passed", False)
            print(f"[GPUKernels] Extended validation: {'PASS' if passed else 'FAIL'}")
            return result_dict
        except Exception as e:
            print(f"[GPUKernels] Validation error: {e}")
            return {"error": str(e)}

    def check_runtime(self, step: int, kernel_outputs: dict = None, reference_outputs: dict = None) -> list:
        """Runtime correctness check for a training step.

        Call this periodically during training.

        Returns:
            list of RuntimeAlert dicts (empty if no issues)
        """
        if not self.runtime_monitor.should_check(step):
            return []

        alerts = self.runtime_monitor.check_step(
            step, kernel_outputs, reference_outputs, {}
        )

        for alert in alerts:
            alert_dict = alert if isinstance(alert, dict) else alert.to_dict() if hasattr(alert, "to_dict") else {}
            severity = alert_dict.get("severity", "warning")
            kid = alert_dict.get("kernel_id", "?")

            if severity == "critical":
                print(f"[GPUKernels] CRITICAL: kernel {kid} — auto-disabling")
                self.config_manager.disable_kernel(kid, "runtime_divergence")
                # Attempt recovery
                try:
                    recovery = self.auto_recovery.attempt_recovery(kid, "runtime_divergence")
                except Exception as e:
                    logger.exception(e)
            else:
                print(f"[GPUKernels] WARNING: kernel {kid} — divergence detected")

        return alerts

    def get_status(self) -> dict:
        """Get full pipeline status."""
        kernel_config = self.config_manager.get_active_kernels()
        return {
            "initialized": self._initialized,
            "hardware": self._hardware_profile.gpu_name if self._hardware_profile else "unknown",
            "active_kernels": len(kernel_config),
            "kernel_ids": list(kernel_config.keys()),
        }

    def print_dashboard(self):
        """Print CLI dashboard."""
        print(self.dashboard.render_cli())

    def generate_html_dashboard(self, path: str = None) -> str:
        """Generate HTML dashboard."""
        path = path or os.path.join(self.data_dir, "kernel_dashboard.html")
        return self.dashboard.render_html(path)

    def run_health_audit(self) -> dict:
        """Run cross-system health audit."""
        return self.health_auditor.audit()
