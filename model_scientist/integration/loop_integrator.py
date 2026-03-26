"""
LoopIntegrator — wires all Model Scientist Pipeline phases into the
autoresearch experiment loop.

Provides a single ModelScientistLoop class that wraps the standard
propose → train → evaluate → keep/discard cycle with:
  1. Pre-experiment diagnostics capture
  2. Failure constraint injection into the research agent's context
  3. Multi-scale gate before full evaluation
  4. Post-acceptance ablation and component stripping
  5. Journal logging for every experiment
  6. Periodic metric evolution (critic + correlator + promoter)
"""

import logging

logger = logging.getLogger(__name__)

import os
import time
import subprocess
import re
import tempfile

from ..schemas import DiagnosticsReport, JournalEntry, load_jsonl
from ..journal.writer import JournalWriter
from ..journal.reader import JournalReader
from ..failure_mining.extractor import FailureExtractor
from ..failure_mining.clusterer import FailureClusterer
from ..failure_mining.constraints import ConstraintGenerator
from ..failure_mining.validator import ConstraintValidator
from ..scaling.config_deriver import ScaleConfigDeriver
from ..scaling.runner import ScaleRunner
from ..scaling.gate import ScaleGate
from ..scaling.logger import ScaleGateLogger
from ..ablation.orchestrator import AblationOrchestrator
from ..ablation.stripper import ComponentStripper
from ..ablation.journal_writer import AblationJournalWriter
from ..metrics.critic import CriticAgent
from ..metrics.implementer import MetricImplementer
from ..metrics.registry import MetricRegistry
from ..metrics.correlator import MetricCorrelator
from ..metrics.promoter import MetricPromoter
from ..metrics.context_budget import ContextBudgetManager
from .safety_guard import SafetyGuard


_VAL_BPB_RE = re.compile(r"val_bpb:\s+([\d.]+)")


class ModelScientistLoop:
    """Orchestrates the full Model Scientist Pipeline around experiments.

    Usage:
        loop = ModelScientistLoop(base_train_path="train.py")
        loop.initialize(baseline_val_bpb=1.23)

        # For each experiment in the autoresearch loop:
        context = loop.pre_experiment_context()
        # ... agent uses context to propose modification ...
        result = loop.run_experiment(modified_source, hypothesis, predicted_delta)
        # result has .verdict, .val_bpb, .journal_entry, etc.
    """

    def __init__(
        self,
        base_train_path: str = "train.py",
        data_dir: str = None,
        metric_evolution_every_n: int = 10,
        enable_scale_gate: bool = True,
        enable_ablation: bool = True,
        enable_metric_evolution: bool = True,
    ):
        self.base_train_path = os.path.abspath(base_train_path)
        self.data_dir = data_dir or os.path.join(os.path.dirname(self.base_train_path), "data")
        os.makedirs(self.data_dir, exist_ok=True)

        self.metric_evolution_every_n = metric_evolution_every_n
        self.enable_scale_gate = enable_scale_gate
        self.enable_ablation = enable_ablation
        self.enable_metric_evolution = enable_metric_evolution

        # Core components
        self.journal_writer = JournalWriter()
        self.journal_reader = JournalReader()
        self.safety = SafetyGuard()

        # Failure mining
        self.failure_extractor = FailureExtractor()
        self.failure_clusterer = FailureClusterer()
        self.constraint_generator = ConstraintGenerator()
        self.constraint_validator = ConstraintValidator()

        # Scaling
        self.scale_config_deriver = ScaleConfigDeriver()
        self.scale_runner = ScaleRunner()
        self.scale_gate = ScaleGate()
        self.scale_gate_logger = ScaleGateLogger()

        # Ablation
        self.ablation_orchestrator = AblationOrchestrator()
        self.component_stripper = ComponentStripper()
        self.ablation_journal_writer = AblationJournalWriter()

        # Metrics evolution
        self.critic = CriticAgent()
        self.metric_implementer = MetricImplementer()
        self.metric_registry = MetricRegistry(
            path=os.path.join(self.data_dir, "metric_registry.json")
        )
        self.metric_correlator = MetricCorrelator()
        self.metric_promoter = MetricPromoter()
        self.context_budget = ContextBudgetManager()

        # State
        self._experiment_count = 0
        self._baseline_val_bpb = None
        self._current_val_bpb = None
        self._last_diagnostics = None
        self._active_constraints = []
        self._base_source = None

    def initialize(self, baseline_val_bpb: float):
        """Initialize the loop with the baseline val_bpb.

        Call this after the first (unmodified) training run.
        """
        self._baseline_val_bpb = baseline_val_bpb
        self._current_val_bpb = baseline_val_bpb

        with open(self.base_train_path, "r") as f:
            self._base_source = f.read()

        # Load persisted state
        self.metric_registry.load()
        self.safety.load_state()

        # Initial failure mining from existing journal
        self._update_failure_constraints()

    def pre_experiment_context(self, max_tokens: int = 3000) -> str:
        """Generate context for the research agent before it proposes a modification.

        Returns a formatted string containing:
        - Recent experiment history
        - Active failure constraints
        - Diagnostic metrics summary
        - Metric evolution status
        """
        parts = []

        # 1. Recent experiment summary from journal
        self.journal_reader.reload()
        recent_context = self.journal_reader.generate_context(max_tokens=max_tokens // 3)
        if recent_context:
            parts.append("## Recent Experiments\n" + recent_context)

        # 2. Active failure constraints
        if self._active_constraints:
            constraint_text = "\n".join(
                f"- {c.text}" for c in self._active_constraints if c.is_valid
            )
            if constraint_text:
                parts.append("## Known Failure Patterns (AVOID these)\n" + constraint_text)

        # 3. Diagnostic metrics context
        if self._last_diagnostics and self.enable_metric_evolution:
            metrics_context = self.context_budget.allocate(
                self.metric_registry, self._last_diagnostics,
                max_tokens=max_tokens // 3,
            )
            if metrics_context:
                parts.append("## Diagnostic Metrics\n" + metrics_context)

        # 4. Pipeline status
        status = self.safety.status()
        parts.append(
            f"## Pipeline Status\n"
            f"- Experiments run: {self._experiment_count}\n"
            f"- Current val_bpb: {self._current_val_bpb:.6f}\n"
            f"- Baseline val_bpb: {self._baseline_val_bpb:.6f}\n"
            f"- Improvement: {self._baseline_val_bpb - self._current_val_bpb:.6f}"
        )

        return "\n\n".join(parts)

    def run_experiment(
        self,
        modified_source: str,
        hypothesis: str,
        predicted_delta: float,
        modification_diff: str = "",
        tags: list = None,
    ) -> dict:
        """Run a single experiment through the full pipeline.

        Args:
            modified_source: The modified train.py source code.
            hypothesis: What the modification is testing.
            predicted_delta: Predicted change in val_bpb (negative = improvement).
            modification_diff: Git diff text.
            tags: Optional tags for the journal entry.

        Returns:
            dict with keys: verdict, val_bpb, delta, journal_entry,
                           scale_gate_passed, ablation_report, stripped
        """
        self._experiment_count += 1
        self.safety.reset_cycle()
        result = {
            "verdict": "rejected",
            "val_bpb": None,
            "delta": None,
            "journal_entry": None,
            "scale_gate_passed": None,
            "ablation_report": None,
            "stripped": False,
        }

        # --- Phase 2.2-2.3: Scale gate (optional) ---
        if self.enable_scale_gate:
            try:
                scale_passed, scale_reason, scale_results, scale_prediction = (
                    self._run_scale_gate(modified_source)
                )
                result["scale_gate_passed"] = scale_passed
                if not scale_passed:
                    # Log rejection and return early
                    entry = self.journal_writer.log_experiment(
                        hypothesis=hypothesis,
                        predicted_delta=predicted_delta,
                        actual_delta=0.0,
                        modification_diff=modification_diff,
                        verdict="rejected",
                        tags=(tags or []) + ["scale_gate_rejected"],
                        scaling_data={
                            "passed": False,
                            "reason": scale_reason,
                        },
                        scale_gate_passed=False,
                    )
                    result["journal_entry"] = entry
                    return result
            except Exception as e:
                # Scale gate failure is non-fatal — proceed without it
                print(f"WARNING: scale gate error: {e}")
                result["scale_gate_passed"] = None

        # --- Run full training ---
        val_bpb = self._run_training(modified_source)
        if val_bpb is None:
            # Crash
            entry = self.journal_writer.log_experiment(
                hypothesis=hypothesis,
                predicted_delta=predicted_delta,
                actual_delta=0.0,
                modification_diff=modification_diff,
                verdict="crashed",
                tags=(tags or []) + ["crash"],
            )
            result["verdict"] = "crashed"
            result["journal_entry"] = entry
            return result

        delta = val_bpb - self._current_val_bpb
        result["val_bpb"] = val_bpb
        result["delta"] = delta

        # --- Accept/reject decision ---
        if delta < 0:  # improvement (lower bpb is better)
            verdict = "accepted"
        else:
            verdict = "rejected"

        # --- Phase 3: Ablation (only for accepted, multi-component mods) ---
        if verdict == "accepted" and self.enable_ablation:
            try:
                ablation_result = self._run_ablation(
                    modified_source, val_bpb, modification_diff
                )
                if ablation_result:
                    result["ablation_report"] = ablation_result.get("report")
                    if ablation_result.get("stripped_source"):
                        # Re-evaluate stripped version
                        stripped_bpb = self._run_training(ablation_result["stripped_source"])
                        if stripped_bpb is not None and stripped_bpb <= val_bpb:
                            modified_source = ablation_result["stripped_source"]
                            val_bpb = stripped_bpb
                            delta = val_bpb - self._current_val_bpb
                            result["val_bpb"] = val_bpb
                            result["delta"] = delta
                            result["stripped"] = True
            except Exception as e:
                print(f"WARNING: ablation error: {e}")

        # --- Log to journal ---
        entry = self.journal_writer.log_experiment(
            hypothesis=hypothesis,
            predicted_delta=predicted_delta,
            actual_delta=delta,
            modification_diff=modification_diff,
            verdict=verdict,
            tags=tags or [],
        )
        result["verdict"] = verdict
        result["journal_entry"] = entry

        # --- Update state on acceptance ---
        if verdict == "accepted":
            self._current_val_bpb = val_bpb
            self._base_source = modified_source

        # --- Periodic metric evolution ---
        if (self.enable_metric_evolution and
                self._experiment_count % self.metric_evolution_every_n == 0):
            try:
                self._run_metric_evolution()
            except Exception as e:
                print(f"WARNING: metric evolution error: {e}")

        # --- Update failure constraints ---
        if self._experiment_count % 5 == 0:
            try:
                self._update_failure_constraints()
            except Exception as e:
                print(f"WARNING: failure constraint update error: {e}")

        return result

    # --- Internal methods ---

    def _run_training(self, source: str) -> float | None:
        """Run training from source code, return val_bpb or None on crash."""
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".py", delete=False, dir=os.path.dirname(self.base_train_path)
            ) as f:
                f.write(source)
                temp_path = f.name

            cmd = f"uv run {temp_path}"
            proc = subprocess.run(
                cmd, shell=True, capture_output=True, text=True, timeout=600,
                cwd=os.path.dirname(self.base_train_path),
            )

            if proc.returncode != 0:
                return None

            match = _VAL_BPB_RE.search(proc.stdout)
            if match:
                return float(match.group(1))
            return None

        except (subprocess.TimeoutExpired, Exception):
            return None
        finally:
            try:
                os.unlink(temp_path)
            except (OSError, UnboundLocalError):
                pass

    def _run_scale_gate(self, modified_source: str):
        """Run multi-scale testing and evaluate the scale gate."""
        # Check compute budget
        estimated_time = 300 * 0.75  # rough estimate: 75% of TIME_BUDGET for sub-scale runs
        allowed, reason = self.safety.can_run_scale_test(estimated_time)
        if not allowed:
            return True, "budget_skip", [], None  # pass through if budget exhausted

        t0 = time.time()
        results = self.scale_runner.run_at_scales(
            modification_diff="",  # we pass source directly
            modified_source=modified_source,
            scales=[0.25, 0.5],
        )
        self.safety.record_scale_test_compute(time.time() - t0)

        passed, reason, prediction = self.scale_gate.evaluate_from_results(results)
        return passed, reason, results, prediction

    def _run_ablation(self, modified_source: str, val_bpb: float, diff: str) -> dict | None:
        """Run ablation analysis on an accepted modification."""
        base_source = self._base_source
        if not base_source:
            return None

        # Check compute budget
        estimated_time = 300 * 3  # rough: 3 variants * TIME_BUDGET
        allowed, reason = self.safety.can_run_ablation(estimated_time)
        if not allowed:
            return None

        t0 = time.time()
        report = self.ablation_orchestrator.run_ablation(
            base_source=base_source,
            modified_source=modified_source,
            base_val_bpb=self._current_val_bpb,
        )
        self.safety.record_ablation_compute(time.time() - t0)

        if not report or not report.ablation_results:
            return None

        # Strip neutral/negative components
        stripped_source, stripped_ids = self.component_stripper.strip(base_source, report)

        result = {"report": report}
        if stripped_ids:
            result["stripped_source"] = stripped_source
            result["stripped_ids"] = stripped_ids

        return result

    def _update_failure_constraints(self):
        """Re-mine failure patterns and update active constraints."""
        journal_path = self.journal_writer.path
        if not os.path.exists(journal_path):
            return

        failures = self.failure_extractor.extract(journal_path)
        if len(failures) < 3:
            return

        patterns = self.failure_clusterer.cluster(failures)
        if not patterns:
            return

        entries = load_jsonl(journal_path)
        constraints = self.constraint_generator.generate(patterns, entries)
        validated = self.constraint_validator.validate(constraints, entries)
        self._active_constraints = [c for c in validated if c.is_valid]

    def _run_metric_evolution(self):
        """Run one cycle of metric evolution: critic → implement → correlate → promote."""
        # 1. Critic proposes new metrics
        can_propose, max_n, reason = self.safety.can_propose_metrics()
        if can_propose and self._last_diagnostics:
            existing = self.metric_registry.get_active() + self.metric_registry.get_candidates()
            proposals = self.critic.propose_metrics(self._last_diagnostics, existing)
            proposals = proposals[:max_n]

            for metric in proposals:
                # Check if needs human review
                if self.safety.needs_review(metric.computation_method):
                    self.safety.queue_for_review(metric.name, metric.computation_method)
                    continue

                # Try to implement and validate
                success, error, values = self.metric_implementer.implement(
                    metric, [self._last_diagnostics]
                )
                if success:
                    self.metric_registry.add(metric)

            self.safety.record_proposals(len(proposals))

        # 2. Correlate metrics with experiment outcomes
        self.journal_reader.reload()
        entries = self.journal_reader._entries
        all_metrics = self.metric_registry.get_active() + self.metric_registry.get_candidates()

        if len(entries) >= 5:
            correlations = self.metric_correlator.compute_correlations(
                all_metrics, entries, self._diagnostics_history
            )

            # 3. Promote/retire
            result = self.metric_promoter.run_cycle(
                self.metric_registry, correlations
            )

        # 4. Save state
        self.metric_registry.save()
        self.safety.save_state()

    def update_diagnostics(self, report: DiagnosticsReport):
        """Update the latest diagnostics report (call after each training run)."""
        self._last_diagnostics = report
        self._diagnostics_history.append(report)
