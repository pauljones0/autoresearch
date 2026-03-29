"""
Model Scientist Pipeline — Full pipeline orchestrator.

Chains all four phases together for the end-to-end flow:
  1. Training run → diagnostics report (Phase 1)
  2. Agent reads diagnostics + evolved metrics + failure constraints → proposes modification
  3. Modification tested at multiple scales → passes scale gate (Phase 2)
  4. Accepted modification ablated → neutral components stripped (Phase 3)
  5. Full record written to hypothesis journal (Phase 1)
  6. Failure patterns updated (Phase 2)
  7. Metric evolution cycle runs (Phase 4)

Usage:
    from model_scientist.pipeline import ModelScientistPipeline

    pipeline = ModelScientistPipeline(train_path="train.py")
    pipeline.initialize(baseline_val_bpb=1.23)

    # Each experiment cycle:
    context = pipeline.get_research_context()
    # ... agent proposes modification using context ...
    result = pipeline.evaluate_modification(
        modified_source=new_code,
        hypothesis="Added RMSNorm pre-attention",
        predicted_delta=-0.01,
        modification_diff=diff_text,
    )
    print(f"Verdict: {result['verdict']}, val_bpb: {result['val_bpb']}")
"""

import os
import time

from .integration.loop_integrator import ModelScientistLoop
from .integration.safety_guard import SafetyGuard
from .integration.monitor import PipelineMonitor
from .diagnostics.instrumenter import DiagnosticsInstrumenter
from .diagnostics.attention_analyzer import AttentionAnalyzer
from .diagnostics.loss_decomposer import LossDecomposer
from .diagnostics.validator import validate_diagnostics_report
from .diagnostics.probes import ProbeTrainer
from .diagnostics.cka import CKASimilarity
from .diagnostics.head_clustering import HeadClusterer
from .journal.writer import JournalWriter
from .journal.reader import JournalReader
from .schemas import DiagnosticsReport


class ModelScientistPipeline:
    """Top-level orchestrator for the complete Model Scientist Pipeline.

    Wraps ModelScientistLoop with the full diagnostics capture layer
    and provides a clean API for the autoresearch experiment loop.
    """

    def __init__(
        self,
        train_path: str = "train.py",
        data_dir: str = None,
        enable_scale_gate: bool = True,
        enable_ablation: bool = True,
        enable_metric_evolution: bool = True,
        enable_probes: bool = True,
        metric_evolution_every_n: int = 10,
        diagnostics_capture_interval: int = 50,
    ):
        self.train_path = os.path.abspath(train_path)
        self.data_dir = data_dir or os.path.join(
            os.path.dirname(self.train_path), "data"
        )
        os.makedirs(self.data_dir, exist_ok=True)

        # Core loop with all phases
        self.loop = ModelScientistLoop(
            base_train_path=self.train_path,
            data_dir=self.data_dir,
            metric_evolution_every_n=metric_evolution_every_n,
            enable_scale_gate=enable_scale_gate,
            enable_ablation=enable_ablation,
            enable_metric_evolution=enable_metric_evolution,
        )

        # Diagnostics Phase 1.1
        self.instrumenter = DiagnosticsInstrumenter(
            capture_every_n_steps=diagnostics_capture_interval
        )

        # Diagnostics Phase 1.2
        self.enable_probes = enable_probes
        self.probe_trainer = ProbeTrainer()
        self.cka = CKASimilarity()
        self.head_clusterer = HeadClusterer()

        # Monitoring
        self.monitor = PipelineMonitor(data_dir=self.data_dir)

        # State
        self._initialized = False
        self._experiment_count = 0

    def initialize(self, baseline_val_bpb: float):
        """Initialize the pipeline with the baseline performance.

        Call after the first unmodified training run completes.
        """
        self.loop.initialize(baseline_val_bpb)
        self._initialized = True
        print(f"[ModelScientist] Initialized with baseline val_bpb={baseline_val_bpb:.6f}")

    def get_research_context(self, max_tokens: int = 3000) -> str:
        """Get diagnostic context for the research agent.

        Returns formatted text containing recent experiment history,
        failure constraints, diagnostic metrics, and pipeline status.
        Inject this into the research agent's prompt.
        """
        if not self._initialized:
            return "Pipeline not initialized. Run baseline first."
        return self.loop.pre_experiment_context(max_tokens=max_tokens)

    def evaluate_modification(
        self,
        modified_source: str,
        hypothesis: str,
        predicted_delta: float,
        modification_diff: str = "",
        tags: list = None,
    ) -> dict:
        """Evaluate a proposed modification through the full pipeline.

        Steps:
        1. Scale gate (if enabled) — test at smaller scales first
        2. Full training run
        3. Accept/reject based on val_bpb improvement
        4. Ablation (if enabled and accepted) — strip neutral components
        5. Log everything to the hypothesis journal
        6. Update failure patterns and metric evolution periodically

        Args:
            modified_source: Modified train.py source code.
            hypothesis: What this modification tests.
            predicted_delta: Predicted val_bpb change (negative = improvement).
            modification_diff: Git diff of the change.
            tags: Optional experiment tags.

        Returns:
            dict with: verdict, val_bpb, delta, journal_entry,
                      scale_gate_passed, ablation_report, stripped
        """
        if not self._initialized:
            raise RuntimeError("Pipeline not initialized. Call initialize() first.")

        self._experiment_count += 1
        print(f"\n[ModelScientist] Experiment #{self._experiment_count}: {hypothesis[:80]}")

        result = self.loop.run_experiment(
            modified_source=modified_source,
            hypothesis=hypothesis,
            predicted_delta=predicted_delta,
            modification_diff=modification_diff,
            tags=tags,
        )

        # Log result
        verdict = result["verdict"]
        val_bpb = result.get("val_bpb")
        delta = result.get("delta")

        if verdict == "accepted":
            print(f"[ModelScientist] ACCEPTED: val_bpb={val_bpb:.6f} (delta={delta:+.6f})")
            if result.get("stripped"):
                print(f"[ModelScientist] Components stripped via ablation")
        elif verdict == "rejected":
            if result.get("scale_gate_passed") is False:
                print(f"[ModelScientist] REJECTED by scale gate")
            elif val_bpb is not None:
                print(f"[ModelScientist] REJECTED: val_bpb={val_bpb:.6f} (delta={delta:+.6f})")
            else:
                print(f"[ModelScientist] REJECTED")
        else:
            print(f"[ModelScientist] CRASHED")

        return result

    def capture_diagnostics(self, model, dataloader=None, num_probe_batches: int = 5):
        """Run full diagnostics capture on a trained model.

        Call this after a training run completes (before evaluate_modification
        for the next experiment, or after the initial baseline).

        Args:
            model: The trained GPT model (raw or compiled).
            dataloader: Optional dataloader for probe tasks.
            num_probe_batches: Batches for probe training.

        Returns:
            DiagnosticsReport with all Phase 1 data.
        """
        report = DiagnosticsReport(timestamp=time.time())

        # Phase 1.1: Gradient/activation stats from instrumenter
        try:
            existing_report = self.instrumenter.generate_report()
            if existing_report:
                report.gradient_stats = existing_report.gradient_stats
                report.activation_stats = existing_report.activation_stats
        except Exception as e:
            print(f"[ModelScientist] Instrumenter error: {e}")

        # Phase 1.1: Attention analysis
        try:
            if dataloader:
                analyzer = AttentionAnalyzer()
                batch_x, batch_y, _ = next(iter(dataloader))
                attention_stats = analyzer.analyze(model, batch_x)
                report.attention_stats = [
                    s.to_dict() if hasattr(s, "to_dict") else s
                    for s in (attention_stats or [])
                ]
        except Exception as e:
            print(f"[ModelScientist] Attention analysis error: {e}")

        # Phase 1.2: CKA similarity
        try:
            if dataloader:
                batch_x, _, _ = next(iter(dataloader))
                similarity = self.cka.compute_similarity_matrix(model, batch_x)
                report.layer_similarity_matrix = [
                    s if isinstance(s, dict) else (s.to_dict() if hasattr(s, "to_dict") else s)
                    for s in (similarity or [])
                ]
        except Exception as e:
            print(f"[ModelScientist] CKA error: {e}")

        # Phase 1.2: Head clustering
        try:
            if dataloader:
                batch_x, _, _ = next(iter(dataloader))
                clusters = self.head_clusterer.cluster_heads(model, batch_x)
                report.head_clusters = [
                    c if isinstance(c, dict) else (c.to_dict() if hasattr(c, "to_dict") else c)
                    for c in (clusters or [])
                ]
        except Exception as e:
            print(f"[ModelScientist] Head clustering error: {e}")

        # Phase 1.2: Probes
        if self.enable_probes and dataloader:
            try:
                probe_results = self.probe_trainer.train_probes(
                    model, dataloader, num_batches=num_probe_batches
                )
                report.probe_results = [
                    p if isinstance(p, dict) else (p.to_dict() if hasattr(p, "to_dict") else p)
                    for p in (probe_results or [])
                ]
            except Exception as e:
                print(f"[ModelScientist] Probe training error: {e}")

        # Validate
        is_valid, errors, warnings = validate_diagnostics_report(report.to_dict())
        if not is_valid:
            print(f"[ModelScientist] Diagnostics validation errors: {errors}")
        if warnings:
            print(f"[ModelScientist] Diagnostics warnings: {warnings}")

        # Save report
        report_path = os.path.join(self.data_dir, "diagnostics_report.json")
        report.to_json(report_path)

        # Feed to loop
        self.loop.update_diagnostics(report)

        return report

    def print_status(self):
        """Print pipeline status dashboard to CLI."""
        self.monitor.print_status()

    def reload_overrides(self, path: str) -> bool:
        """Reload configuration overrides from a JSON file.

        Used by meta-optimization layer to propagate config changes.
        Returns True if overrides were applied.
        """
        import json
        try:
            if not os.path.exists(path):
                return False
            mtime = os.path.getmtime(path)
            if mtime <= getattr(self, "_last_override_mtime", 0):
                return False
            with open(path) as f:
                overrides = json.load(f)
            for k, v in overrides.items():
                if hasattr(self, k):
                    setattr(self, k, v)
            self._last_override_mtime = mtime
            return True
        except Exception as e:
            logger.exception(e)
            return False

    def generate_html_dashboard(self, path: str = None) -> str:
        """Generate HTML dashboard.

        Args:
            path: Output file path. Defaults to data/dashboard.html.

        Returns:
            HTML string.
        """
        path = path or os.path.join(self.data_dir, "dashboard.html")
        return self.monitor.generate_html(path)
