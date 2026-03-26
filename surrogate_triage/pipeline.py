"""
Surrogate Triage Pipeline — Full orchestrator.

Chains all five phases together:
  1. Daily: ArXiv papers fetched, filtered, techniques extracted, diffs generated
  2. Daily: Diffs enriched, scored by surrogate, queued
  3. Per-iteration: Top candidates routed through ModelScientistPipeline
  4. Per-evaluation: Results feed back to surrogate, extraction tracker, source tracker
  5. Weekly: Ingestion bias updated, surrogate retrained, reports generated

Usage:
    from surrogate_triage.pipeline import SurrogateTriagePipeline

    pipeline = SurrogateTriagePipeline(
        train_path="train.py",
        model_scientist_pipeline=model_scientist_pipeline,
    )
    pipeline.initialize()

    # Daily cycle:
    pipeline.run_daily_ingestion()

    # Per-iteration (called from experiment loop):
    result = pipeline.evaluate_next_paper_candidate()

    # Weekly:
    report = pipeline.run_weekly_cycle()
"""

import os
import time

from .schemas import load_jsonl, save_jsonl

# Phase 1: Ingestion
from .ingestion.arxiv_fetcher import ArxivFetcher
from .ingestion.paper_filter import PaperFilter
from .ingestion.pdf_downloader import PDFDownloader

# Phase 1.2-1.3: Extraction
from .extraction.paper_reader import PaperReader
from .extraction.extraction_validator import ExtractionValidator
from .extraction.deduplicator import TechniqueDeduplicator
from .extraction.diff_generator import DiffGenerator
from .extraction.diff_checker import DiffApplicabilityChecker
from .extraction.constraint_filter import ConstraintPreFilter

# Phase 2: Surrogate
from .surrogate.diff_embedder import DiffEmbedder
from .surrogate.feature_enricher import FeatureEnricher
from .surrogate.journal_data_extractor import JournalDataExtractor
from .surrogate.trainer import SurrogateTrainer
from .surrogate.evaluator import SurrogateEvaluator
from .surrogate.cold_start import ColdStartManager
from .surrogate.retrainer import SurrogateRetrainer
from .surrogate.drift_detector import FeatureDriftDetector

# Phase 3: Funnel
from .funnel.scoring_pipeline import SurrogateScoringPipeline
from .funnel.queue_manager import QueueManager
from .funnel.evaluation_scheduler import EvaluationScheduler
from .funnel.candidate_router import PaperCandidateRouter
from .funnel.journal_enricher import PaperJournalEnricher
from .funnel.feedback_loop import SurrogateFeedbackLoop
from .funnel.extraction_tracker import ExtractionQualityTracker
from .funnel.source_tracker import PaperSourceTracker
from .funnel.failure_bridge import FailureMiningBridge

# Phase 4: Intelligence
from .intelligence.diagnostics_linker import DiagnosticsIngestionLinker
from .intelligence.bottleneck_mapper import BottleneckMapper
from .intelligence.ingestion_bias import IngestionBiasAgent
from .intelligence.diversity_enforcer import SourceDiversityEnforcer
from .intelligence.source_scout import NewSourceScout

# Phase 5: Steady-state
from .steady_state.dual_calibration import DualCalibrationManager
from .steady_state.novelty_classifier import NoveltyClassifier
from .steady_state.impact_tracker import ImpactTracker
from .steady_state.ceiling_monitor import KnowledgeCeilingMonitor

# Reporting
from .reporting.meta_monitor import MetaLearningMonitor
from .reporting.reporter import PipelineReporter
from .reporting.metric_proposer import SurrogateMetricProposer


class SurrogateTriagePipeline:
    """Top-level orchestrator for the Literature-Informed Surrogate Triage Pipeline."""

    def __init__(
        self,
        train_path: str = "train.py",
        data_dir: str = None,
        model_scientist_pipeline=None,
        paper_evaluation_fraction: float = 0.3,
    ):
        self.train_path = os.path.abspath(train_path)
        self.data_dir = data_dir or os.path.join(
            os.path.dirname(self.train_path), "data"
        )
        os.makedirs(self.data_dir, exist_ok=True)
        self.model_scientist = model_scientist_pipeline
        self.paper_evaluation_fraction = paper_evaluation_fraction

        # Load base source
        with open(self.train_path) as f:
            self._base_source = f.read()

        # Phase 1.1: Ingestion
        self.fetcher = ArxivFetcher(
            index_path=os.path.join(self.data_dir, "papers_index.jsonl")
        )
        self.paper_filter = PaperFilter()
        self.pdf_downloader = PDFDownloader()

        # Phase 1.2-1.3: Extraction
        self.paper_reader = PaperReader()
        self.extraction_validator = ExtractionValidator()
        self.deduplicator = TechniqueDeduplicator()
        self.diff_generator = DiffGenerator()
        self.diff_checker = DiffApplicabilityChecker()
        self.constraint_filter = ConstraintPreFilter()

        # Phase 2: Surrogate
        self.embedder = DiffEmbedder()
        self.enricher = FeatureEnricher()
        self.journal_extractor = JournalDataExtractor()
        self.surrogate_trainer = SurrogateTrainer()
        self.surrogate_evaluator = SurrogateEvaluator()
        self.cold_start = ColdStartManager()
        self.retrainer = SurrogateRetrainer()
        self.drift_detector = FeatureDriftDetector()

        # Phase 3: Funnel
        self.scoring_pipeline = SurrogateScoringPipeline()
        self.queue_manager = QueueManager(
            queue_path=os.path.join(self.data_dir, "evaluation_queue.json")
        )
        self.scheduler = EvaluationScheduler(
            paper_fraction=paper_evaluation_fraction
        )
        self.router = PaperCandidateRouter()
        self.journal_enricher = PaperJournalEnricher()
        self.feedback_loop = SurrogateFeedbackLoop(
            data_dir=self.data_dir
        )
        self.extraction_tracker = ExtractionQualityTracker(
            data_dir=self.data_dir
        )
        self.source_tracker = PaperSourceTracker(
            path=os.path.join(self.data_dir, "paper_source_quality.json")
        )
        self.failure_bridge = FailureMiningBridge()

        # Phase 4: Intelligence
        self.diagnostics_linker = DiagnosticsIngestionLinker()
        self.bottleneck_mapper = BottleneckMapper()
        self.ingestion_bias = IngestionBiasAgent()
        self.diversity_enforcer = SourceDiversityEnforcer()
        self.source_scout = NewSourceScout(
            known_sources_path=os.path.join(self.data_dir, "known_sources.json")
        )

        # Phase 5: Steady-state
        self.calibration = DualCalibrationManager()
        self.novelty_classifier = NoveltyClassifier()
        self.impact_tracker = ImpactTracker(data_dir=self.data_dir)
        self.ceiling_monitor = KnowledgeCeilingMonitor(
            data_dir=self.data_dir
        )

        # Reporting
        self.meta_monitor = MetaLearningMonitor(data_dir=self.data_dir)
        self.reporter = PipelineReporter(data_dir=self.data_dir)
        self.metric_proposer = SurrogateMetricProposer()

        # State
        self._initialized = False
        self._surrogate_trained = False
        self._iteration_count = 0

    def initialize(self):
        """Initialize the pipeline. Call once at startup."""
        self.queue_manager.load()
        self.source_tracker.load()

        # Check if surrogate can skip cold-start
        journal_path = os.path.join(self.data_dir, "..", "hypothesis_journal.jsonl")
        n_entries = len(load_jsonl(journal_path))
        regime_name, filter_frac, use_surrogate = self.cold_start.get_regime(n_entries)
        print(f"[SurrogateTriage] Initialized. Journal has {n_entries} entries → regime: {regime_name}")

        if use_surrogate and n_entries >= 50:
            try:
                self._train_surrogate(journal_path)
            except Exception as e:
                print(f"[SurrogateTriage] Surrogate training failed: {e}")

        self._initialized = True

    def run_daily_ingestion(self, days_back: int = 1, max_results: int = 100) -> dict:
        """Run the daily paper ingestion cycle.

        Fetches new papers, filters, extracts techniques, generates diffs,
        scores with surrogate, and queues top candidates.

        Returns:
            Summary dict with counts at each stage.
        """
        stats = {
            "papers_fetched": 0,
            "papers_relevant": 0,
            "techniques_extracted": 0,
            "diffs_generated": 0,
            "diffs_valid": 0,
            "candidates_queued": 0,
        }

        # 1. Fetch papers
        try:
            new_papers = self.fetcher.fetch_recent(days_back=days_back, max_results=max_results)
            stats["papers_fetched"] = len(new_papers)
        except Exception as e:
            print(f"[SurrogateTriage] Fetch error: {e}")
            return stats

        if not new_papers:
            return stats

        # 2. Get current diagnostics for filtering
        diagnostics = None
        if self.model_scientist and hasattr(self.model_scientist, "loop"):
            diagnostics = getattr(self.model_scientist.loop, "_last_diagnostics", None)

        # 3. Filter papers
        diagnostics_dict = diagnostics.to_dict() if diagnostics and hasattr(diagnostics, "to_dict") else diagnostics
        relevant, filtered = self.paper_filter.filter_papers(
            [p if isinstance(p, dict) else p.to_dict() for p in new_papers],
            diagnostics=diagnostics_dict,
        )
        stats["papers_relevant"] = len(relevant)

        # 4. Apply source quality bias
        try:
            relevant = self.ingestion_bias.apply_bias(
                relevant,
                source_quality_path=os.path.join(self.data_dir, "paper_source_quality.json"),
            )
        except Exception:
            pass

        # 5. Extract techniques
        all_techniques = []
        for paper in relevant:
            try:
                techniques = self.paper_reader.extract_from_abstract(paper)
                all_techniques.extend(techniques)
            except Exception:
                continue
        stats["techniques_extracted"] = len(all_techniques)

        # 6. Deduplicate
        all_techniques = self.deduplicator.deduplicate(all_techniques)
        journal_path = os.path.join(self.data_dir, "..", "hypothesis_journal.jsonl")
        if os.path.exists(journal_path):
            all_techniques = self.deduplicator.check_against_journal(
                all_techniques, journal_path
            )

        # Filter to non-duplicate, non-explored
        active_techniques = [
            t for t in all_techniques
            if not (t.get("deduplicated") if isinstance(t, dict) else getattr(t, "deduplicated", False))
            and not (t.get("already_explored") if isinstance(t, dict) else getattr(t, "already_explored", False))
        ]

        # 7. Generate diffs
        all_diffs = []
        for technique in active_techniques:
            try:
                diffs = self.diff_generator.generate_diffs(technique, self._base_source)
                all_diffs.extend(diffs)
            except Exception:
                continue
        stats["diffs_generated"] = len(all_diffs)

        # 8. Check applicability
        valid_diffs = []
        for diff in all_diffs:
            try:
                checked = self.diff_checker.check(diff, self._base_source)
                if checked.applies_cleanly if hasattr(checked, "applies_cleanly") else checked.get("applies_cleanly"):
                    valid_diffs.append(checked)
            except Exception:
                continue
        stats["diffs_valid"] = len(valid_diffs)

        # 9. Score and queue (if surrogate is trained)
        if self._surrogate_trained and valid_diffs:
            try:
                scored = self.scoring_pipeline.score_and_rank(
                    valid_diffs, self.surrogate_trainer, self.enricher,
                )
                # Queue top candidates
                for pred in scored[:10]:  # top 10
                    diff = pred if isinstance(pred, dict) else pred
                    diff_id = diff.get("diff_id", "") if isinstance(diff, dict) else getattr(diff, "diff_id", "")
                    from .schemas import QueueEntry
                    entry = QueueEntry(
                        queue_id=f"q_{diff_id}",
                        diff_id=diff_id,
                        surrogate_score=diff.get("predicted_delta", 0) if isinstance(diff, dict) else getattr(diff, "predicted_delta", 0),
                        adjusted_score=diff.get("adjusted_score", 0) if isinstance(diff, dict) else getattr(diff, "adjusted_score", 0),
                        diff_text=diff.get("diff_text", "") if isinstance(diff, dict) else getattr(diff, "diff_text", ""),
                        status="pending",
                    )
                    self.queue_manager.add(entry)
                stats["candidates_queued"] = min(len(scored), 10)
                self.queue_manager.save()
            except Exception as e:
                print(f"[SurrogateTriage] Scoring error: {e}")
        elif valid_diffs:
            # Cold-start: queue all valid diffs without scoring
            from .schemas import QueueEntry
            for diff in valid_diffs[:10]:
                diff_dict = diff if isinstance(diff, dict) else diff.to_dict() if hasattr(diff, "to_dict") else {}
                entry = QueueEntry(
                    queue_id=f"q_{diff_dict.get('diff_id', '')}",
                    diff_id=diff_dict.get("diff_id", ""),
                    diff_text=diff_dict.get("diff_text", ""),
                    status="pending",
                )
                self.queue_manager.add(entry)
            stats["candidates_queued"] = min(len(valid_diffs), 10)
            self.queue_manager.save()

        print(f"[SurrogateTriage] Daily ingestion: {stats}")
        return stats

    def evaluate_next_paper_candidate(self, iteration: int = 0) -> dict | None:
        """Evaluate the next paper candidate if the scheduler says it's time.

        Call this from the main experiment loop alongside internal modifications.

        Returns:
            Evaluation result dict, or None if no paper candidate should be evaluated.
        """
        self._iteration_count += 1

        # Check if we should evaluate a paper candidate this iteration
        queue_size = len(self.queue_manager.get_all())
        if not self.scheduler.should_evaluate_paper(queue_size, self._iteration_count):
            return None

        # Get next candidate
        candidate = self.scheduler.get_next_candidate(self.queue_manager)
        if candidate is None:
            return None

        candidate_dict = candidate if isinstance(candidate, dict) else candidate.to_dict() if hasattr(candidate, "to_dict") else {}
        print(f"[SurrogateTriage] Evaluating paper candidate: {candidate_dict.get('technique_name', candidate_dict.get('queue_id', '?'))}")

        # Route through Model Scientist Pipeline
        if self.model_scientist:
            try:
                result = self.router.route(candidate, self._base_source, self.model_scientist)
            except Exception as e:
                print(f"[SurrogateTriage] Evaluation error: {e}")
                result = {"verdict": "crashed", "val_bpb": None, "delta": None}
        else:
            print("[SurrogateTriage] No Model Scientist pipeline — skipping evaluation")
            return None

        # Record feedback
        actual_delta = result.get("delta", 0) or 0
        predicted_delta = candidate_dict.get("surrogate_score", 0)
        diff_id = candidate_dict.get("diff_id", "")

        self.feedback_loop.record_result(
            diff_id=diff_id,
            predicted_delta=predicted_delta,
            actual_delta=actual_delta,
            source="paper",
        )

        # Track extraction quality
        paper_id = candidate_dict.get("paper_id", "")
        technique_id = candidate_dict.get("technique_id", "")
        if paper_id:
            self.extraction_tracker.record(
                paper_id=paper_id,
                technique_id=technique_id,
                predicted=predicted_delta,
                actual=actual_delta,
            )

        # Track source quality
        self.source_tracker.record_evaluation(
            paper_metadata=candidate_dict,
            technique_category=candidate_dict.get("modification_category", "other"),
            verdict=result.get("verdict", "rejected"),
            delta=actual_delta,
        )
        self.source_tracker.save()

        # Feed rejections to failure mining
        if result.get("verdict") == "rejected":
            try:
                self.failure_bridge.feed_rejection(
                    journal_entry=result.get("journal_entry", {}),
                    paper_metadata=candidate_dict,
                    journal_path=self.journal_path,
                )
            except Exception:
                pass

        # Update base source if accepted
        if result.get("verdict") == "accepted":
            try:
                with open(self.train_path) as f:
                    self._base_source = f.read()
            except Exception:
                pass

        # Update ceiling monitor
        try:
            journal_path = os.path.join(self.data_dir, "..", "hypothesis_journal.jsonl")
            entries = load_jsonl(journal_path)
            self.ceiling_monitor.update(entries)
        except Exception:
            pass

        return result

    def run_weekly_cycle(self) -> str:
        """Run the weekly maintenance cycle.

        - Retrain surrogate
        - Update ingestion bias
        - Generate reports
        - Propose metrics

        Returns:
            Weekly report as markdown string.
        """
        print("[SurrogateTriage] Running weekly cycle...")

        # 1. Retrain surrogate
        journal_path = os.path.join(self.data_dir, "..", "hypothesis_journal.jsonl")
        model_path = os.path.join(self.data_dir, "surrogate_model.pt")
        try:
            retrained, metrics = self.retrainer.check_and_retrain(
                journal_path, model_path
            )
            if retrained:
                print(f"[SurrogateTriage] Surrogate retrained: {metrics}")
                self._surrogate_trained = True
        except Exception as e:
            print(f"[SurrogateTriage] Retraining error: {e}")

        # 2. Update calibration
        try:
            feedback = self.feedback_loop.get_prediction_errors()
            if len(feedback) >= 10:
                predictions = [f[0] for f in feedback]
                actuals = [f[1] for f in feedback]
                sources = ["paper"] * len(feedback)  # TODO: track actual sources
                self.calibration.calibrate(predictions, actuals, sources)
        except Exception:
            pass

        # 3. Propose new metrics to Model Scientist
        if self.model_scientist:
            try:
                triage_stats = {
                    "n_paper_evaluations": len(self.feedback_loop.get_prediction_errors()),
                    "total_papers_ingested": len(load_jsonl(
                        os.path.join(self.data_dir, "papers_index.jsonl")
                    )),
                }
                proposals = self.metric_proposer.propose_metrics(triage_stats)
                if proposals:
                    print(f"[SurrogateTriage] Proposed {len(proposals)} new metrics")
            except Exception:
                pass

        # 4. Collect meta-learning snapshot
        try:
            journal_reader = None
            try:
                from model_scientist.journal.reader import JournalReader
                journal_reader = JournalReader()
            except Exception:
                pass

            snapshot = self.meta_monitor.collect_snapshot(
                journal_reader=journal_reader,
            )
            self.meta_monitor.record_snapshot(snapshot)
        except Exception:
            pass

        # 5. Generate weekly report
        try:
            report = self.reporter.generate_weekly_report(
                source_tracker=self.source_tracker,
                extraction_tracker=self.extraction_tracker,
                queue_manager=self.queue_manager,
                ceiling_monitor=self.ceiling_monitor,
                meta_monitor=self.meta_monitor,
            )
            print(f"[SurrogateTriage] Weekly report generated at {self.data_dir}/weekly_report.md")
            return report
        except Exception as e:
            print(f"[SurrogateTriage] Report generation error: {e}")
            return ""

    def _train_surrogate(self, journal_path: str):
        """Train the surrogate model from journal data."""
        examples = self.journal_extractor.extract(journal_path, self.enricher)
        if len(examples) < 10:
            print(f"[SurrogateTriage] Not enough training data ({len(examples)} examples)")
            return

        metrics = self.surrogate_trainer.train(examples)
        model_path = os.path.join(self.data_dir, "surrogate_model.pt")
        self.surrogate_trainer.save(model_path)
        self._surrogate_trained = True
        print(f"[SurrogateTriage] Surrogate trained on {len(examples)} examples: {metrics}")

    def get_status(self) -> dict:
        """Get current pipeline status."""
        return {
            "initialized": self._initialized,
            "surrogate_trained": self._surrogate_trained,
            "iteration_count": self._iteration_count,
            "queue_size": len(self.queue_manager.get_all()),
            "papers_indexed": len(load_jsonl(
                os.path.join(self.data_dir, "papers_index.jsonl")
            )),
        }
"papers_index.jsonl")
            )),
        }
