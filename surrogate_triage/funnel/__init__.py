"""
Phase 3 — Funnel Integration: routes paper-sourced candidates through the
Model Scientist Pipeline for evaluation, tracking, and feedback.
"""

from .scoring_pipeline import SurrogateScoringPipeline
from .queue_manager import QueueManager
from .evaluation_scheduler import EvaluationScheduler
from .candidate_router import PaperCandidateRouter
from .journal_enricher import PaperJournalEnricher
from .ablation_insights import AblationInsightExtractor
from .feedback_loop import SurrogateFeedbackLoop
from .extraction_tracker import ExtractionQualityTracker
from .source_tracker import PaperSourceTracker
from .failure_bridge import FailureMiningBridge

__all__ = [
    "SurrogateScoringPipeline",
    "QueueManager",
    "EvaluationScheduler",
    "PaperCandidateRouter",
    "PaperJournalEnricher",
    "AblationInsightExtractor",
    "SurrogateFeedbackLoop",
    "ExtractionQualityTracker",
    "PaperSourceTracker",
    "FailureMiningBridge",
]
