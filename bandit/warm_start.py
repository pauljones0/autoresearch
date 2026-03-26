"""
Posterior warm-start from journal history.

Bootstraps BanditState alpha/beta posteriors and initial temperatures
from historical journal entries so the bandit begins with informed priors.
"""

import json
import math
import time

from bandit.schemas import BanditState, ArmState, WarmStartValidationReport
from bandit.journal_mapper import JournalArmMapper
from bandit.taxonomy import get_all_arms


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _compute_temperature(alpha: float, beta: float, T_base: float) -> float:
    """Compute initial temperature from Beta distribution std dev.

    T = T_base * sqrt(a*b / ((a+b)^2 * (a+b+1)))
    """
    ab = alpha + beta
    variance = (alpha * beta) / (ab * ab * (ab + 1))
    return T_base * math.sqrt(variance)


def _determine_regime(entry_count: int) -> str:
    """Map total journal entry count to a regime string."""
    if entry_count < 30:
        return "no_bandit"
    if entry_count < 100:
        return "conservative_bandit"
    return "full_bandit"


# ---------------------------------------------------------------------------
# PosteriorWarmStarter
# ---------------------------------------------------------------------------

class PosteriorWarmStarter:
    """Bootstrap bandit posteriors from an existing experiment journal."""

    def __init__(self):
        self._mapper = JournalArmMapper()

    def warm_start(self, journal_path: str, taxonomy=None,
                   T_base: float = 0.025) -> BanditState:
        """Load journal, map entries to arms, and build initial BanditState.

        For each arm:
          alpha = 1 + successes
          beta  = 1 + failures
          temperature = T_base * sqrt(a*b / ((a+b)^2*(a+b+1)))

        Regime is set by total entry count:
          <30  -> no_bandit
          30-99 -> conservative_bandit
          >=100 -> full_bandit
        """
        groups = self._mapper.map_all(journal_path)
        meta = groups.pop("_meta", {})
        total_entries = meta.get("total", 0)

        # Count successes/failures per arm
        arm_stats: dict[str, tuple[int, int]] = {}  # arm_id -> (successes, failures)
        for arm_id, entries in groups.items():
            if arm_id == "unknown":
                continue
            successes = 0
            failures = 0
            for entry in entries:
                verdict = (entry.get("verdict") or "").lower()
                if verdict in ("accepted", "improved", "improvement"):
                    successes += 1
                else:
                    failures += 1
            arm_stats[arm_id] = (successes, failures)

        # Build state
        state = BanditState()
        state.T_base = T_base

        # Ensure all canonical arms exist
        all_arms = taxonomy or get_all_arms()
        if hasattr(all_arms, '__iter__'):
            for arm_def in all_arms:
                arm_id = arm_def.arm_id if hasattr(arm_def, 'arm_id') else str(arm_def)
                successes, failures = arm_stats.get(arm_id, (0, 0))
                alpha = 1.0 + successes
                beta = 1.0 + failures
                temp = _compute_temperature(alpha, beta, T_base)

                arm_state = ArmState(
                    alpha=alpha,
                    beta=beta,
                    temperature=temp,
                    total_attempts=successes + failures,
                    total_successes=successes,
                    source_type=getattr(arm_def, 'source_type', 'internal'),
                )
                state.arms[arm_id] = arm_state

        state.regime = _determine_regime(total_entries)
        state.global_iteration = total_entries
        now = time.time()
        state.metadata["created_at"] = now
        state.metadata["last_updated"] = now
        state.metadata["warm_start_source"] = journal_path
        state.metadata["warm_start_entries"] = total_entries
        return state

    def warm_start_from_source_tracker(
        self, source_tracker_path: str, taxonomy=None,
    ) -> dict[str, tuple[float, float]]:
        """Extract alpha/beta pairs from a source_tracker.json file.

        Returns ``{arm_id: (alpha, beta)}`` for each arm found.
        """
        try:
            with open(source_tracker_path) as f:
                tracker = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {}

        results: dict[str, tuple[float, float]] = {}
        if not isinstance(tracker, dict):
            return results

        for arm_id, info in tracker.items():
            if not isinstance(info, dict):
                continue
            successes = info.get("successes", 0)
            failures = info.get("failures", 0)
            alpha = 1.0 + successes
            beta = 1.0 + failures
            results[arm_id] = (alpha, beta)
        return results


# ---------------------------------------------------------------------------
# Kernel arm warm-start
# ---------------------------------------------------------------------------

def warm_start_kernel_arms(
    journal_path: str,
    kernel_config_path: str,
    disable_log_path: str,
) -> dict[str, tuple[float, float]]:
    """Warm-start kernel arms from journal and kernel config.

    Reads disabled kernels from *disable_log_path* to exclude them,
    then counts kernel successes/failures from the journal.

    Returns ``{arm_id: (alpha, beta)}``.
    """
    # Load disabled kernels
    disabled: set[str] = set()
    try:
        with open(disable_log_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        entry = json.loads(line)
                        kid = entry.get("kernel_id", "")
                        if kid:
                            disabled.add(kid)
                    except json.JSONDecodeError:
                        continue
    except FileNotFoundError:
        pass

    # Map journal entries
    mapper = JournalArmMapper()
    groups = mapper.map_all(journal_path)
    groups.pop("_meta", None)

    results: dict[str, tuple[float, float]] = {}
    for arm_id in ("kernel_discovery", "kernel_evolution"):
        entries = groups.get(arm_id, [])
        successes = 0
        failures = 0
        for entry in entries:
            kernel_id = entry.get("kernel_id", "")
            if kernel_id in disabled:
                continue
            verdict = (entry.get("verdict") or "").lower()
            if verdict in ("accepted", "improved", "improvement"):
                successes += 1
            else:
                failures += 1
        results[arm_id] = (1.0 + successes, 1.0 + failures)
    return results


# ---------------------------------------------------------------------------
# Warm-start validation
# ---------------------------------------------------------------------------

class PosteriorWarmStartValidator:
    """Validate that warm-started posteriors are consistent with the journal."""

    def validate_warm_start(
        self, state: BanditState, journal_path: str,
    ) -> WarmStartValidationReport:
        """Check posterior means within +/-0.05 of empirical rates,
        evidence conservation, and temperature ordering."""
        report = WarmStartValidationReport()
        report.valid = True

        mapper = JournalArmMapper()
        groups = mapper.map_all(journal_path)
        groups.pop("_meta", None)

        total_journal_evidence = 0
        total_state_evidence = 0

        for arm_id, arm_state in state.arms.items():
            if not isinstance(arm_state, ArmState):
                continue

            entries = groups.get(arm_id, [])
            successes = sum(
                1 for e in entries
                if (e.get("verdict") or "").lower()
                in ("accepted", "improved", "improvement")
            )
            total = len(entries)
            failures = total - successes
            empirical_rate = successes / total if total > 0 else 0.5

            # Posterior mean of Beta(alpha, beta)
            posterior_mean = arm_state.alpha / (arm_state.alpha + arm_state.beta)

            deviation = abs(posterior_mean - empirical_rate)
            check = {
                "empirical_rate": empirical_rate,
                "posterior_mean": posterior_mean,
                "deviation": deviation,
                "ok": deviation <= 0.05 or total == 0,
            }
            report.per_arm_checks[arm_id] = check

            if not check["ok"]:
                report.valid = False
                report.issues.append(
                    f"Arm {arm_id}: posterior mean {posterior_mean:.3f} deviates "
                    f"from empirical {empirical_rate:.3f} by {deviation:.3f}")

            # Evidence accounting
            total_journal_evidence += total
            total_state_evidence += (arm_state.alpha - 1) + (arm_state.beta - 1)

        # Evidence conservation
        report.evidence_conservation = {
            "journal_evidence": total_journal_evidence,
            "state_evidence": total_state_evidence,
            "conserved": abs(total_journal_evidence - total_state_evidence) < 1e-6,
        }
        if not report.evidence_conservation["conserved"]:
            report.valid = False
            report.issues.append(
                f"Evidence not conserved: journal={total_journal_evidence}, "
                f"state={total_state_evidence}")

        # Temperature ordering: arms with more evidence should have
        # lower or equal temperatures
        arm_items = [
            (aid, a) for aid, a in state.arms.items()
            if isinstance(a, ArmState)
        ]
        report.temperature_ordering_correct = True
        for i, (aid_a, a) in enumerate(arm_items):
            for aid_b, b in arm_items[i + 1:]:
                evidence_a = a.alpha + a.beta
                evidence_b = b.alpha + b.beta
                if evidence_a > evidence_b and a.temperature > b.temperature + 1e-9:
                    report.temperature_ordering_correct = False
                elif evidence_b > evidence_a and b.temperature > a.temperature + 1e-9:
                    report.temperature_ordering_correct = False

        if not report.temperature_ordering_correct:
            report.issues.append("Temperature ordering violated: "
                                 "more evidence should yield lower temperature")

        return report
