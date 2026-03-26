"""
Rollback safety net for annealing-accepted regressions.
"""

import time
from bandit.schemas import BanditState, ArmState, RollbackResult, load_jsonl


class RollbackSafetyNet:
    """Detects and triggers rollback when annealing acceptance leads to cascading regressions."""

    def check_and_rollback(
        self,
        state: BanditState,
        journal_reader,
    ) -> "RollbackResult | None":
        """Check if a rollback is needed and perform it.

        Trigger condition: The last accepted entry was via annealing AND the next 3
        entries are all regressions (delta > 0 and rejected or further annealing-accepted
        regressions).

        Args:
            state: Current bandit state.
            journal_reader: Object with .read_recent(n) -> list of entry dicts.

        Returns:
            RollbackResult if rollback triggered, None otherwise.
        """
        if not state.enable_rollback_safety:
            return None

        # Read recent journal entries (need at least 4: 1 annealing + 3 subsequent)
        recent = journal_reader.read_recent(10)
        if len(recent) < 4:
            return None

        # Find the most recent annealing-accepted entry
        annealing_idx = None
        for i in range(len(recent) - 1, -1, -1):
            if recent[i].get("accepted_by") == "annealing":
                annealing_idx = i
                break

        if annealing_idx is None:
            return None

        # Check if there are at least 3 subsequent entries
        subsequent = recent[annealing_idx + 1:]
        if len(subsequent) < 3:
            return None

        # Check if all 3 subsequent entries are regressions
        all_regressions = all(
            entry.get("delta", 0) > 0 for entry in subsequent[:3]
        )

        if not all_regressions:
            return None

        annealing_entry = recent[annealing_idx]
        arm_id = annealing_entry.get("arm_id", "")
        source_hash = annealing_entry.get("source_hash_before", "")

        # Perform rollback: revert the arm's posterior to undo the annealing acceptance
        if arm_id in state.arms:
            arm = state.arms[arm_id]
            if isinstance(arm, ArmState):
                # Undo the annealing acceptance by incrementing beta (it was a regression)
                arm.beta += 1.0
                arm.consecutive_failures += 1

        return RollbackResult(
            rolled_back=True,
            arm_id=arm_id,
            annealing_entry_id=annealing_entry.get("entry_id", ""),
            subsequent_entries=[e.get("entry_id", "") for e in subsequent[:3]],
            reverted_to_source_hash=source_hash,
        )
