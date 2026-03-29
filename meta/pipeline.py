"""
Meta-Autoresearch Pipeline - Top-level Orchestrator
"""

import logging

logger = logging.getLogger(__name__)

import os
import time

from meta.schemas import (
    MetaBanditState, DimensionState, MetaContext, MetaExperimentResult,
    AggregateIR, ROIData, ConfigDocumentation, ConvergenceStatus,
    save_json, load_json,
)

from meta.inventory.bandit_params import BanditParameterInventorist
from meta.inventory.ms_params import ModelScientistParameterInventorist
from meta.inventory.st_params import SurrogateTriageParameterInventorist
from meta.inventory.gk_params import GPUKernelParameterInventorist
from meta.config_schema import MetaConfigSchemaBuilder
from meta.config_manager import MetaConfigManager
from meta.bridges.bandit_bridge import BanditConfigBridge
from meta.bridges.ms_bridge import ModelScientistConfigBridge
from meta.bridges.st_bridge import SurrogateTriageConfigBridge
from meta.bridges.gk_bridge import GPUKernelConfigBridge
from meta.safety.sandbox import MetaSandboxEnforcer
from meta.safety.recursion_guard import RecursionDepthGuard
from meta.safety.compute_budget import MetaComputeBudgetEnforcer
from meta.safety.eval_guard import EvaluationMetricGuard
from meta.safety.boundary_tester import BoundaryViolationTester
from meta.baseline.runner import BaselineRunOrchestrator
from meta.baseline.improvement_rate import ImprovementRateCalculator
from meta.baseline.power_analysis import MinimumDetectableEffectCalculator
from meta.baseline.experiment_length import MetaExperimentLengthOptimizer
from meta.bandit.meta_bandit import MetaBandit
from meta.bandit.discretizer import MetaVariantDiscretizer
from meta.bandit.meta_state import MetaStateManager
from meta.bandit.meta_updater import MetaPosteriorUpdater
from meta.experiment.runner import MetaExperimentRunner
from meta.experiment.scheduler import MetaExperimentScheduler
from meta.experiment.logger import MetaExperimentLogger
from meta.prompts.variant_generator import PromptVariantGenerator
from meta.prompts.evaluator import PromptABEvaluator
from meta.prompts.evolution import PromptEvolutionController
from meta.context.budget_explorer import ContextBudgetExplorer
from meta.evaluation.protocol_explorer import EvalProtocolExplorer
from meta.evaluation.variance_cost import MetaVarianceCostAnalyzer
from meta.stop.scaffold import STOPScaffold
from meta.stop.safety_checker import StrategySafetyChecker
from meta.stop.executor import StrategyExecutor
from meta.stop.evolution import StrategyEvolutionController
from meta.interactions.detector import InteractionDetector
from meta.interactions.joint_optimizer import JointOptimizer
from meta.budget.optimizer import MetaBudgetOptimizer
from meta.budget.roi_tracker import MetaROITracker
from meta.convergence.detector import MetaConvergenceDetector
from meta.convergence.maintenance import MaintenanceModeManager
from meta.convergence.divergence import DivergenceWatcher
from meta.documentation.config_documenter import MetaConfigDocumenter
from meta.documentation.sensitivity import MetaSensitivityAnalyzer
from meta.knowledge.insight_extractor import InsightExtractor
from meta.knowledge.transfer_validator import TransferValidator
from meta.knowledge.knowledge_base import MetaKnowledgeBaseWriter
from meta.knowledge.updater import KnowledgeBaseUpdater
from meta.knowledge.bootstrapper import NewCampaignBootstrapper
from meta.validation.extended_validator import MetaExtendedValidator
from meta.validation.defaults_comparison import DefaultsVsMetaComparator
from meta.dashboard import MetaDashboard
from meta.monitoring.stability import LongTermStabilityMonitor


class MetaAutoresearchPipeline:
    """Top-level orchestrator for Meta-Autoresearch."""

    def __init__(self, work_dir=".",
                 bandit_pipeline=None, model_scientist_pipeline=None,
                 surrogate_triage_pipeline=None, gpu_kernel_pipeline=None):
        self.work_dir = work_dir
        # Sub-layer pipeline refs for cross-layer integration
        self._bandit_pipeline = bandit_pipeline
        self._model_scientist_pipeline = model_scientist_pipeline
        self._surrogate_triage_pipeline = surrogate_triage_pipeline
        self._gpu_kernel_pipeline = gpu_kernel_pipeline
        self.state_path = os.path.join(work_dir, "meta_state.json")
        self.log_path = os.path.join(work_dir, "meta_log.jsonl")
        self.config_path = os.path.join(work_dir, "meta_config.json")
        self.schema_path = os.path.join(work_dir, "meta_config_schema.json")
        self.state = MetaBanditState()
        self.baseline_ir = AggregateIR()
        self.experiment_history = []

        # Phase 1
        self.bandit_inv = BanditParameterInventorist()
        self.ms_inv = ModelScientistParameterInventorist()
        self.st_inv = SurrogateTriageParameterInventorist()
        self.gk_inv = GPUKernelParameterInventorist()
        self.schema_builder = MetaConfigSchemaBuilder()
        self.config_manager = MetaConfigManager(
            config_path=self.config_path, schema_path=self.schema_path)
        self.bandit_bridge = BanditConfigBridge(
            overrides_path=os.path.join(work_dir, "bandit_overrides.json"))
        self.ms_bridge = ModelScientistConfigBridge(
            overrides_path=os.path.join(work_dir, "ms_overrides.json"))
        self.st_bridge = SurrogateTriageConfigBridge(
            overrides_path=os.path.join(work_dir, "st_overrides.json"))
        self.gk_bridge = GPUKernelConfigBridge(
            overrides_path=os.path.join(work_dir, "gk_overrides.json"))
        self.sandbox = MetaSandboxEnforcer(work_dir=work_dir)
        self.recursion_guard = RecursionDepthGuard()
        self.budget_enforcer = MetaComputeBudgetEnforcer()
        self.eval_guard = EvaluationMetricGuard(
            train_path=os.path.join(work_dir, "train.py"))
        self.boundary_tester = BoundaryViolationTester(work_dir=work_dir)
        self.baseline_runner = BaselineRunOrchestrator(work_dir=work_dir)
        self.ir_calculator = ImprovementRateCalculator()
        self.mdes_calculator = MinimumDetectableEffectCalculator()
        self.length_optimizer = MetaExperimentLengthOptimizer()

        # Phase 2
        self.meta_bandit = MetaBandit()
        self.discretizer = MetaVariantDiscretizer()
        self.state_manager = MetaStateManager()
        self.posterior_updater = MetaPosteriorUpdater()
        self.experiment_runner = MetaExperimentRunner()
        self.experiment_scheduler = MetaExperimentScheduler()
        self.experiment_logger = MetaExperimentLogger(self.log_path)
        self.prompt_generator = PromptVariantGenerator()
        self.prompt_evaluator = PromptABEvaluator()
        self.prompt_evolution = PromptEvolutionController()
        self.context_explorer = ContextBudgetExplorer()
        self.protocol_explorer = EvalProtocolExplorer()
        self.variance_cost_analyzer = MetaVarianceCostAnalyzer()

        # Phase 3
        self.stop_scaffold = STOPScaffold()
        self.strategy_checker = StrategySafetyChecker()
        self.strategy_executor = StrategyExecutor()
        self.strategy_evolution = StrategyEvolutionController()
        self.interaction_detector = InteractionDetector()
        self.joint_optimizer = JointOptimizer()
        self.budget_optimizer = MetaBudgetOptimizer()
        self.roi_tracker = MetaROITracker()

        # Phase 4
        self.convergence_detector = MetaConvergenceDetector()
        self.maintenance_manager = MaintenanceModeManager()
        self.divergence_watcher = DivergenceWatcher()
        self.config_documenter = MetaConfigDocumenter()
        self.sensitivity_analyzer = MetaSensitivityAnalyzer()
        self.insight_extractor = InsightExtractor()
        self.transfer_validator = TransferValidator()
        self.knowledge_writer = MetaKnowledgeBaseWriter()
        self.knowledge_updater = KnowledgeBaseUpdater()
        self.campaign_bootstrapper = NewCampaignBootstrapper()

        # Phase 5
        self.extended_validator = MetaExtendedValidator(self.state, [])
        self.defaults_comparator = DefaultsVsMetaComparator(self.state, {})
        self.dashboard = MetaDashboard()
        self.stability_monitor = LongTermStabilityMonitor()

    def initialize(self):
        """Initialize the meta-optimization pipeline."""
        self.recursion_guard.set_depth(1)
        self.eval_guard.initialize()
        all_params = (self.bandit_inv.inventory() + self.ms_inv.inventory()
                      + self.st_inv.inventory() + self.gk_inv.inventory())
        schema = self.schema_builder.build(all_params)
        save_json(schema, self.schema_path)
        default_config = self.schema_builder.generate_default_config(all_params)
        self.config_manager.save(default_config)
        variants = self.discretizer.discretize_all(all_params)
        self.state = MetaBanditState(
            meta_regime="baseline", current_config=default_config,
            best_config=default_config,
            metadata={"created_at": time.time(), "last_updated": time.time(),
                      "schema_version": "1.0"})
        for p in all_params:
            pv = variants.get(p.param_id, [p.default_value])
            posteriors = {str(v): {"alpha": 1.0, "beta": 1.0} for v in pv}
            self.state.dimensions[p.param_id] = DimensionState(
                param_id=p.param_id, variants=pv,
                variant_posteriors=posteriors, current_best=p.default_value)
        self.state_manager.save(self.state, self.state_path)

        # Update comparator with actual default config
        self.defaults_comparator = DefaultsVsMetaComparator(
            self.state, default_config, all_params
        )

        # Build base context with sub-layer pipeline refs
        self.base_context = MetaContext(
            bandit_pipeline=self._bandit_pipeline,
            model_scientist_pipeline=self._model_scientist_pipeline,
            surrogate_triage_pipeline=self._surrogate_triage_pipeline,
            gpu_kernel_pipeline=self._gpu_kernel_pipeline,
            work_dir=self.work_dir,
        )

        return self.state

    def run_baselines(self, n_runs=3, n_iterations=100):
        """Run baseline measurement campaigns."""
        results = self.baseline_runner.run_baselines(n_runs, n_iterations)
        self.baseline_ir = self.ir_calculator.compute_aggregate(results)
        self.state.meta_regime = "active"
        self.state_manager.save(self.state, self.state_path)
        return self.baseline_ir

    def run_meta_iteration(self, context=None):
        """Execute one meta-level decision cycle."""
        if context is None:
            context = self.base_context
        self.eval_guard.verify_evaluation_unchanged()
        conv = self.convergence_detector.check(self.state, [])
        if conv.recommendation == "enter_maintenance":
            self.state = self.maintenance_manager.enter_maintenance(
                self.state, self.config_manager.load())
        div = self.divergence_watcher.check(self.state, [], self.baseline_ir)
        if div and div.triggered:
            self.state.meta_regime = "active"
            self.state.budget_fraction = 0.2
        should_run = self.experiment_scheduler.should_run_meta_experiment(
            self.state, self.state.global_meta_iteration, self.budget_enforcer)
        result = {"iteration": self.state.global_meta_iteration, "action": "production"}
        if should_run:
            exp = self.run_experiment(context)
            result["action"] = "meta_experiment"
            result["experiment"] = exp.to_dict()
            self.state = self.posterior_updater.update(self.state, exp, self.baseline_ir)
            if self.state.total_meta_experiments % 5 == 0:
                roi = self.roi_tracker.compute_roi(
                    self.state, self.experiment_history, self.baseline_ir)
                rec = self.budget_optimizer.recommend_budget(self.state, roi)
                if self.state.enable_auto_budget:
                    self.state.budget_fraction = rec.recommended_fraction
        self.state.global_meta_iteration += 1
        self.state_manager.save(self.state, self.state_path)
        return result

    def run_experiment(self, context=None):
        """Run a single meta-experiment."""
        if context is None:
            context = self.base_context
        result = self.experiment_runner.run_experiment(self.state, context, 50)
        self.experiment_history.append(result)
        self.state.total_meta_experiments += 1
        self.experiment_logger.log_experiment_completed(
            experiment_id=result.experiment_id,
            improvement_rate=result.improvement_rate,
            compared_to_baseline=getattr(result, "compared_to_baseline", "unknown"),
            meta_iteration=self.state.global_meta_iteration,
            n_iterations=result.n_iterations,
        )
        return result

    def get_status(self):
        """Get CLI dashboard string."""
        roi = self.roi_tracker.compute_roi(
            self.state, self.experiment_history, self.baseline_ir)
        return self.dashboard.render_cli(self.state, roi, self.experiment_history)

    def get_html_status(self):
        """Get HTML dashboard string."""
        roi = self.roi_tracker.compute_roi(
            self.state, self.experiment_history, self.baseline_ir)
        return self.dashboard.render_html(self.state, roi, self.experiment_history)

    def print_status(self):
        """Print CLI dashboard."""
        print(self.get_status())

    def bootstrap_campaign(self, kb_path, campaign_dir):
        """Initialize new campaign from knowledge base."""
        return self.campaign_bootstrapper.bootstrap(kb_path, campaign_dir)

    def _exp_dicts(self):
        """Convert experiment history to list of dicts."""
        return [e.to_dict() if hasattr(e, "to_dict") else e
                for e in self.experiment_history]

    def _baseline_float(self):
        """Extract baseline IR as float."""
        return self.baseline_ir.mean_ir if hasattr(self.baseline_ir, "mean_ir") else float(self.baseline_ir or 0)

    def extract_insights(self):
        """Extract transferable insights."""
        exp_dicts = self._exp_dicts()
        doc = self.config_documenter.document(
            self.state, exp_dicts, self._baseline_float())
        sens = self.sensitivity_analyzer.analyze(self.state, exp_dicts)
        doc_dict = doc.to_dict() if hasattr(doc, "to_dict") else doc
        sens_dict = sens.to_dict() if hasattr(sens, "to_dict") else sens
        return self.insight_extractor.extract(exp_dicts, doc_dict, sens_dict)

    def run_safety_tests(self):
        """Run all boundary violation tests."""
        return self.boundary_tester.run_all_tests()

    def run_health_check(self):
        """Run stability monitoring."""
        return self.stability_monitor.check(self.state, [], []).to_dict()
