"""
Adaptive Bandit with Simulated Annealing Pipeline — Top-level Orchestrator

Wires all bandit components together and exposes high-level methods:
  initialize()       — warm-start from journal history or create fresh state
  run_iteration()    — execute one bandit iteration
  run_warm_start()   — re-run warm-start from journal data
  run_ab_test()      — A/B comparison vs fixed-fraction baseline
  get_status()       — CLI dashboard string
  get_html_status()  — HTML dashboard string
"""

import logging

logger = logging.getLogger(__name__)

import os
import time
import random as _random

from .schemas import (
    BanditState, ArmState, LoopContext, IterationResult,
    DispatchContext, AcceptanceAnalysisReport, BanditAuditReport,
    save_json, load_json,
)

# Phase 1 — taxonomy, state, persistence, logging, warm-start, regime
from .taxonomy import get_all_arms, get_arm, get_arms_by_source_type
from .journal_mapper import JournalArmMapper
from .state import validate_state
from .persistence import AtomicStateManager
from .log import BanditLogWriter
from .warm_start import PosteriorWarmStarter, PosteriorWarmStartValidator
from .regime import RegimeTransitionManager

# Phase 2 — sampling, boosting, dispatch, update, health, visualization
from .sampler import ThompsonSamplerEngine
from .boosting import DiagnosticsArmBooster
from .prompt_router import CategoryPromptRouter
from .dispatch import BanditDispatchRouter
from .queue_bridge import QueueFilteredPopper
from .updater import PosteriorUpdateEngine
from .delayed_corrections import DelayedCorrectionReceiver
from .visualization import PosteriorVisualizer
from .health import PosteriorHealthChecker

# Phase 3 — temperature, calibration, acceptance, reheat, regime change
from .temperature import TemperatureDeriver
from .calibration import TemperatureCalibrator
from .surrogate_bridge import SurrogateModulationCalculator
from .constraint_bridge import ConstraintDensityCalculator
from .acceptance import AnnealingAcceptanceEngine
from .safety import RollbackSafetyNet
from .reheat import AdaptiveReheatEngine
from .regime_change import RegimeChangeDetector
from .ceiling_bridge import CeilingMonitorBanditBridge

# Phase 4 — loop, coordination, fallback, A/B, dashboard, config, tuning
from .loop import BanditLoop
from .coordination import ThreeSystemStateCoordinator
from .fallback import GracefulDegradationHandler
from .ab_test import ABTestOrchestrator
from .dashboard import BanditDashboard
from .config_reload import HotConfigReloader
from .tuning import AutoTuner

# Phase 5 — audit, replay, metrics
from .health_audit import BanditHealthAuditor
from .metrics import BanditMetricProposer, BanditMetricComputer

# Validation
from .validation.selection_validator import SelectionDistributionValidator
from .validation.reheat_validator import ReheatDecayVerifier
from .validation.replay_validator import LogReplayValidator

# Analysis
from .analysis.acceptance_analyzer import AcceptanceAnalyzer
from .analysis.ab_analyzer import ABStatisticalAnalyzer
from .analysis.allocation_comparator import AllocationEfficiencyComparator


class AdaptiveBanditPipeline:
    """Top-level orchestrator for the Adaptive Bandit with Simulated Annealing."""

    def __init__(self, work_dir: str = ".",
                 model_scientist=None, surrogate_triage=None, gpu_kernels=None):
        self.work_dir = work_dir
        # Cross-layer pipeline refs for dispatch routing
        self._model_scientist = model_scientist
        self._surrogate_triage = surrogate_triage
        self._gpu_kernels = gpu_kernels
        self.state_path = os.path.join(work_dir, "strategy_state.json")
        self.log_path = os.path.join(work_dir, "bandit_log.jsonl")
        self.journal_path = os.path.join(work_dir, "hypothesis_journal.jsonl")
        self.queue_path = os.path.join(work_dir, "evaluation_queue.json")
        self.kernel_config_path = os.path.join(work_dir, "kernel_config.json")
        self.overrides_path = os.path.join(work_dir, "bandit_overrides.json")

        # State
        self.state: BanditState = BanditState()
        self.rng = _random.Random(42)

        # Components
        self.state_manager = AtomicStateManager()
        self.log_writer = BanditLogWriter(self.log_path)
        self.warm_starter = PosteriorWarmStarter()
        self.warm_start_validator = PosteriorWarmStartValidator()
        self.regime_manager = RegimeTransitionManager()
        self.journal_mapper = JournalArmMapper()

        self.sampler = ThompsonSamplerEngine()
        self.booster = DiagnosticsArmBooster()
        self.prompt_router = CategoryPromptRouter()
        self.dispatcher = BanditDispatchRouter()
        self.queue_popper = QueueFilteredPopper()
        self.updater = PosteriorUpdateEngine()
        self.delayed_receiver = DelayedCorrectionReceiver()
        self.visualizer = PosteriorVisualizer()
        self.health_checker = PosteriorHealthChecker()

        self.temp_deriver = TemperatureDeriver()
        self.calibrator = TemperatureCalibrator()
        self.surrogate_mod = SurrogateModulationCalculator()
        self.constraint_calc = ConstraintDensityCalculator()
        self.acceptance_engine = AnnealingAcceptanceEngine()
        self.rollback_net = RollbackSafetyNet()
        self.reheat_engine = AdaptiveReheatEngine()
        self.regime_detector = RegimeChangeDetector()
        self.ceiling_bridge = CeilingMonitorBanditBridge()

        self.bandit_loop = BanditLoop(
            sampler=self.sampler,
            booster=self.booster,
            dispatcher=self.dispatcher,
            acceptance=self.acceptance_engine,
            posterior=self.updater,
            reheat=self.reheat_engine,
            rollback=self.rollback_net,
            health=self.health_checker,
            regime=self.regime_manager,
            temp_deriver=self.temp_deriver,
        )
        self.coordinator = ThreeSystemStateCoordinator()
        self.fallback_handler = GracefulDegradationHandler()
        self.ab_orchestrator = ABTestOrchestrator()
        self.dashboard = BanditDashboard()
        self.config_reloader = HotConfigReloader()
        self.auto_tuner = AutoTuner()

        self.health_auditor = BanditHealthAuditor()
        self.metric_proposer = BanditMetricProposer()
        self.metric_computer = BanditMetricComputer()

        self.selection_validator = SelectionDistributionValidator()
        self.reheat_validator = ReheatDecayVerifier()
        self.replay_validator = LogReplayValidator()
        self.acceptance_analyzer = AcceptanceAnalyzer()
        self.ab_analyzer = ABStatisticalAnalyzer()
        self.allocation_comparator = AllocationEfficiencyComparator()

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def initialize(self, force_fresh: bool = False) -> BanditState:
        """Initialize the bandit pipeline.

        If strategy_state.json exists and is valid, loads it.
        Otherwise, warm-starts from hypothesis_journal.jsonl.
        """
        if not force_fresh and os.path.exists(self.state_path):
            try:
                self.state = self.state_manager.load(self.state_path)
                issues = validate_state(self.state)
                if not issues:
                    return self.state
            except Exception as e:
                logger.exception(e)

        # Warm-start from journal
        self.state = self.run_warm_start()
        return self.state

    def run_warm_start(self) -> BanditState:
        """Run warm-start from journal history."""
        arms = get_all_arms()
        self.state = self.warm_starter.warm_start(
            self.journal_path, arms
        )

        # Validate warm-start
        report = self.warm_start_validator.validate_warm_start(
            self.state, self.journal_path
        )

        # Log warm-start
        self.log_writer.log_warm_start(self.state.to_dict())

        # Save state
        self.state_manager.save(self.state, self.state_path)

        return self.state

    # ------------------------------------------------------------------
    # Iteration
    # ------------------------------------------------------------------

    def run_iteration(
        self,
        model_scientist_pipeline=None,
        surrogate_triage_pipeline=None,
        gpu_kernel_pipeline=None,
        queue_manager=None,
        journal_reader=None,
        journal_writer=None,
        diagnostics_report=None,
        base_source: str = "",
    ) -> IterationResult:
        """Execute one bandit iteration.

        Pipeline refs default to those passed at __init__ if not overridden here.
        """
        # Use stored refs as defaults
        model_scientist_pipeline = model_scientist_pipeline or self._model_scientist
        surrogate_triage_pipeline = surrogate_triage_pipeline or self._surrogate_triage
        gpu_kernel_pipeline = gpu_kernel_pipeline or self._gpu_kernels
        # Hot-reload config overrides
        if os.path.exists(self.overrides_path):
            try:
                self.state, changes = self.config_reloader.check_and_reload(
                    self.state, self.overrides_path
                )
                for change in changes:
                    self.log_writer.log_config_change(
                        parameter=change, old_value="", new_value="",
                        reason="operator"
                    )
            except Exception as e:
                logger.exception(e)

        # Build loop context
        context = LoopContext(
            model_scientist_pipeline=model_scientist_pipeline,
            surrogate_triage_pipeline=surrogate_triage_pipeline,
            gpu_kernel_pipeline=gpu_kernel_pipeline,
            queue_manager=queue_manager,
            journal_reader=journal_reader,
            journal_writer=journal_writer,
            bandit_state=self.state,
            log_writer=self.log_writer,
            rng=self.rng,
            diagnostics_report=diagnostics_report,
            base_source=base_source,
        )

        # Run the iteration
        try:
            result = self.bandit_loop.run_iteration(context)
        except Exception as e:
            decision = self.fallback_handler.handle_failure(
                e, self.state, self.state.global_iteration
            )
            result = IterationResult(
                iteration=self.state.global_iteration,
                verdict="error",
                health_alerts=[{
                    "severity": "critical",
                    "message": f"Fallback triggered: {decision.action} — {decision.detail}",
                }],
            )

        # Reload state from disk (loop may have saved updates)
        if os.path.exists(self.state_path):
            try:
                self.state = self.state_manager.load(self.state_path)
            except Exception as e:
                logger.exception(e)

        # Periodic tasks
        if self.state.global_iteration % 10 == 0:
            self._run_periodic_checks()

        return result

    def _run_periodic_checks(self):
        """Run periodic health checks and auto-tuning."""
        # Health check
        try:
            alerts = self.health_checker.check(self.state)
            for alert in alerts:
                if alert.severity == "critical":
                    pass  # Would trigger operator notification
        except Exception as e:
            logger.exception(e)

        # Regime change detection
        try:
            events = self.regime_detector.detect(
                self.state, self.log_path
            )
        except Exception as e:
            logger.exception(e)

        # Auto-tuner (every 100 iterations)
        if self.state.global_iteration % 100 == 0 and self.state.global_iteration > 0:
            try:
                recommendations = self.auto_tuner.recommend(
                    self.state, self.log_path
                )
                if self.state.enable_auto_tuning:
                    for rec in recommendations:
                        if rec.auto_applicable and rec.confidence == "high":
                            setattr(self.state, rec.parameter, rec.recommended_value)
                    self.state_manager.save(self.state, self.state_path)
            except Exception as e:
                logger.exception(e)

        # Compute bandit metrics
        try:
            metrics = self.metric_computer.compute_all(
                self.state, self.log_path
            )
        except Exception as e:
            logger.exception(e)

    # ------------------------------------------------------------------
    # A/B Testing
    # ------------------------------------------------------------------

    def run_ab_test(self, n_iterations: int = 200, n_seeds: int = 3):
        """Run A/B comparison against fixed-fraction baseline."""
        return self.ab_orchestrator.run(n_iterations=n_iterations, n_seeds=n_seeds)

    # ------------------------------------------------------------------
    # Auditing & Validation
    # ------------------------------------------------------------------

    def run_health_audit(self) -> BanditAuditReport:
        """Run comprehensive health audit across all systems."""
        return self.health_auditor.audit(
            self.state, self.journal_path, self.log_path,
            self.queue_path, self.kernel_config_path
        )

    def run_replay_validation(self):
        """Validate state can be reconstructed from log replay."""
        return self.replay_validator.validate(self.state_path, self.log_path)

    def verify_consistency(self):
        """Verify cross-system state consistency."""
        return self.coordinator.verify_consistency(
            self.state, self.journal_path, self.queue_path,
            self.kernel_config_path
        )

    # ------------------------------------------------------------------
    # Dashboard
    # ------------------------------------------------------------------

    def get_status(self) -> str:
        """Get CLI dashboard string."""
        try:
            analysis = self.acceptance_analyzer.analyze(self.log_path)
        except Exception:
            analysis = None
        return self.dashboard.render_cli(self.state, analysis)

    def get_html_status(self) -> str:
        """Get HTML dashboard string."""
        try:
            analysis = self.acceptance_analyzer.analyze(self.log_path)
        except Exception:
            analysis = None
        return self.dashboard.render_html(self.state, analysis)

    def print_status(self):
        """Print CLI dashboard to stdout."""
        print(self.get_status())

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def propose_metrics(self, existing_metrics: list = None):
        """Propose bandit-derived metrics for the MetricRegistry."""
        return self.metric_proposer.propose(
            self.state, existing_metrics or []
        )

    def compute_metrics(self) -> dict:
        """Compute current values of all bandit-derived metrics."""
        return self.metric_computer.compute_all(self.state, self.log_path)

    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------

    def render_posteriors(self) -> str:
        """Render CLI posterior visualization."""
        return self.visualizer.render_cli(self.state)

    # ------------------------------------------------------------------
    # Recovery
    # ------------------------------------------------------------------

    def recover_state(self):
        """Recover state from bandit_log.jsonl replay."""
        self.state = self.state_manager.recover(self.state_path, self.log_path)
        self.state_manager.save(self.state, self.state_path)
        return self.state

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    def calibrate_temperature(self, target_acceptance: float = 0.25,
                               target_delta: float = 0.01) -> float:
        """Calibrate T_base for desired acceptance probability."""
        recommended = self.calibrator.calibrate(target_acceptance, target_delta)
        return recommended

    def sensitivity_analysis(self):
        """Run temperature sensitivity analysis."""
        return self.calibrator.sensitivity_analysis()
