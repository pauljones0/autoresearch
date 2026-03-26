"""
Arm taxonomy for the Adaptive Bandit pipeline.

Defines 9 canonical arms (7 model-scientist categories + 2 kernel arms)
and provides lookup/filtering utilities.
"""

import json
import random

from bandit.schemas import ArmDefinition, ValidationReport


# ---------------------------------------------------------------------------
# Canonical arm definitions
# ---------------------------------------------------------------------------

_ARM_DEFINITIONS: list[ArmDefinition] = [
    # 7 Python-level model-scientist arms (from failure_mining extractor)
    ArmDefinition(
        arm_id="architecture",
        display_name="Architecture",
        source_type="internal",
        dispatch_target="model_scientist.pipeline.run_architecture",
        prompt_template_key="architecture",
        can_have_paper_variant=True,
    ),
    ArmDefinition(
        arm_id="optimizer",
        display_name="Optimizer",
        source_type="internal",
        dispatch_target="model_scientist.pipeline.run_optimizer",
        prompt_template_key="optimizer",
        can_have_paper_variant=True,
    ),
    ArmDefinition(
        arm_id="hyperparameter",
        display_name="Hyperparameter",
        source_type="internal",
        dispatch_target="model_scientist.pipeline.run_hyperparameter",
        prompt_template_key="hyperparameter",
        can_have_paper_variant=True,
    ),
    ArmDefinition(
        arm_id="activation",
        display_name="Activation",
        source_type="internal",
        dispatch_target="model_scientist.pipeline.run_activation",
        prompt_template_key="activation",
        can_have_paper_variant=True,
    ),
    ArmDefinition(
        arm_id="initialization",
        display_name="Initialization",
        source_type="internal",
        dispatch_target="model_scientist.pipeline.run_initialization",
        prompt_template_key="initialization",
        can_have_paper_variant=True,
    ),
    ArmDefinition(
        arm_id="regularization",
        display_name="Regularization",
        source_type="internal",
        dispatch_target="model_scientist.pipeline.run_regularization",
        prompt_template_key="regularization",
        can_have_paper_variant=True,
    ),
    ArmDefinition(
        arm_id="scheduling",
        display_name="Scheduling",
        source_type="internal",
        dispatch_target="model_scientist.pipeline.run_scheduling",
        prompt_template_key="scheduling",
        can_have_paper_variant=True,
    ),
    # 2 kernel arms
    ArmDefinition(
        arm_id="kernel_discovery",
        display_name="Kernel Discovery",
        source_type="kernel",
        dispatch_target="gpu_kernel.pipeline.run_kernel_discovery",
        queue_filter={"source": "kernel", "subtype": "discovery"},
    ),
    ArmDefinition(
        arm_id="kernel_evolution",
        display_name="Kernel Evolution",
        source_type="kernel",
        dispatch_target="gpu_kernel.pipeline.run_kernel_evolution",
        queue_filter={"source": "kernel", "subtype": "evolution"},
    ),
]

# Build lookup maps
_ARM_BY_ID: dict[str, ArmDefinition] = {a.arm_id: a for a in _ARM_DEFINITIONS}
_ARMS_BY_SOURCE: dict[str, list[ArmDefinition]] = {}
for _arm in _ARM_DEFINITIONS:
    _ARMS_BY_SOURCE.setdefault(_arm.source_type, []).append(_arm)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_all_arms() -> list[ArmDefinition]:
    """Return all canonical arm definitions."""
    return list(_ARM_DEFINITIONS)


def get_arm(arm_id: str) -> ArmDefinition:
    """Return a single arm by ID. Raises KeyError if not found."""
    return _ARM_BY_ID[arm_id]


def get_arms_by_source_type(source_type: str) -> list[ArmDefinition]:
    """Return arms filtered by source_type ('internal', 'paper', 'kernel')."""
    return list(_ARMS_BY_SOURCE.get(source_type, []))


# ---------------------------------------------------------------------------
# Taxonomy Validation
# ---------------------------------------------------------------------------

def validate_taxonomy(journal_path: str) -> ValidationReport:
    """Validate taxonomy coverage against a journal file.

    Checks for:
    - Duplicate arm IDs
    - Orphan categories in journal not covered by any arm
    - Mapping rate of journal entries to arms
    """
    report = ValidationReport()
    report.arm_count = len(_ARM_DEFINITIONS)

    # Check for duplicate arm IDs
    seen_ids: set[str] = set()
    for arm in _ARM_DEFINITIONS:
        if arm.arm_id in seen_ids:
            report.duplicate_arms.append(arm.arm_id)
        seen_ids.add(arm.arm_id)

    # Load journal and check mapping
    try:
        with open(journal_path) as f:
            entries = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        report.issues.append(f"Cannot read journal at {journal_path}")
        return report

    if not isinstance(entries, list):
        entries = []

    arm_ids = set(_ARM_BY_ID.keys())
    mapped = 0
    unmapped = 0
    orphan_categories: set[str] = set()

    for entry in entries:
        if not isinstance(entry, dict):
            continue
        cat = entry.get("bandit_arm") or entry.get("modification_category", "")
        source = entry.get("source", "")

        if cat in arm_ids:
            mapped += 1
        elif source == "kernel":
            mapped += 1
        elif cat:
            unmapped += 1
            if cat not in arm_ids:
                orphan_categories.add(cat)
        else:
            unmapped += 1

    report.mapped_entries = mapped
    report.unmapped_entries = unmapped
    report.orphan_categories = sorted(orphan_categories)
    report.valid = len(report.duplicate_arms) == 0 and len(report.issues) == 0
    return report


# ---------------------------------------------------------------------------
# Paper-Arm Splitter
# ---------------------------------------------------------------------------

class PaperArmSplitter:
    """Decides whether a given arm selection should use the paper-based
    variant (Surrogate Triage queue) instead of the internal prompt path."""

    def should_use_paper(
        self,
        arm_id: str,
        queue_manager: object,
        paper_preference_ratio: float,
        rng: random.Random,
    ) -> bool:
        """Return True if this arm selection should dispatch via a paper queue entry.

        Conditions:
        1. The arm must support paper variants (can_have_paper_variant=True).
        2. The queue_manager must have at least one matching entry.
        3. A random draw must fall below paper_preference_ratio.
        """
        arm_def = _ARM_BY_ID.get(arm_id)
        if arm_def is None or not arm_def.can_have_paper_variant:
            return False

        # Check queue availability via duck-typed queue_manager
        has_entries = False
        if queue_manager is not None:
            get_pending = getattr(queue_manager, "get_pending_for_category", None)
            if callable(get_pending):
                pending = get_pending(arm_id)
                has_entries = bool(pending)

        if not has_entries:
            return False

        return rng.random() < paper_preference_ratio
