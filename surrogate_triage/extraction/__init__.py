"""Surrogate Triage -- Extraction & Diff Generation (Phase 1.2-1.3)."""

from surrogate_triage.extraction.paper_reader import PaperReader
from surrogate_triage.extraction.extraction_validator import ExtractionValidator
from surrogate_triage.extraction.deduplicator import TechniqueDeduplicator
from surrogate_triage.extraction.diff_generator import DiffGenerator
from surrogate_triage.extraction.diff_checker import DiffApplicabilityChecker
from surrogate_triage.extraction.constraint_filter import ConstraintPreFilter

__all__ = [
    "PaperReader",
    "ExtractionValidator",
    "TechniqueDeduplicator",
    "DiffGenerator",
    "DiffApplicabilityChecker",
    "ConstraintPreFilter",
]
