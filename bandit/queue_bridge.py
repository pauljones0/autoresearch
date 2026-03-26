"""
Queue bridge — filters and pops queue entries for paper-sourced arms.
"""


class QueueFilteredPopper:
    """Filters and pops queue entries matching arm criteria."""

    def pop_matching(self, queue_manager, arm_id: str, source_type: str = "paper"):
        """Filter queue for matching entries, pop best (lowest adjusted_score).

        Args:
            queue_manager: Queue manager object with get_entries/pop methods.
            arm_id: Arm identifier to filter by.
            source_type: Source type filter (default "paper").

        Returns:
            dict with entry data, or None if no match.
        """
        if queue_manager is None:
            return None

        # Get all entries from queue
        entries = []
        if hasattr(queue_manager, "get_entries"):
            entries = queue_manager.get_entries()
        elif hasattr(queue_manager, "entries"):
            entries = queue_manager.entries
        else:
            return None

        # Filter for matching arm and source type
        matching = []
        for entry in entries:
            entry_cat = entry.get("category", "") if isinstance(entry, dict) else getattr(entry, "category", "")
            entry_src = entry.get("source_type", "") if isinstance(entry, dict) else getattr(entry, "source_type", "")

            if self._matches_arm(entry_cat, arm_id) and (not source_type or entry_src == source_type):
                matching.append(entry)

        if not matching:
            return None

        # Select best: lowest adjusted_score
        def score_key(e):
            if isinstance(e, dict):
                return e.get("adjusted_score", e.get("score", float("inf")))
            return getattr(e, "adjusted_score", getattr(e, "score", float("inf")))

        best = min(matching, key=score_key)

        # Pop from queue
        entry_id = best.get("id", "") if isinstance(best, dict) else getattr(best, "id", "")
        if hasattr(queue_manager, "pop"):
            queue_manager.pop(entry_id)
        elif hasattr(queue_manager, "remove"):
            queue_manager.remove(entry_id)

        return best

    def queue_availability(self, queue_manager, taxonomy) -> dict:
        """Per-arm counts of available queue entries.

        Args:
            queue_manager: Queue manager object.
            taxonomy: Arm taxonomy for arm_id list.

        Returns:
            dict mapping arm_id -> count of available entries.
        """
        counts = {}

        # Get arm IDs from taxonomy
        arm_ids = []
        if isinstance(taxonomy, dict):
            arm_ids = list(taxonomy.keys())
        elif hasattr(taxonomy, "arms"):
            arm_ids = list(taxonomy.arms.keys()) if isinstance(taxonomy.arms, dict) else []

        for arm_id in arm_ids:
            counts[arm_id] = 0

        if queue_manager is None:
            return counts

        entries = []
        if hasattr(queue_manager, "get_entries"):
            entries = queue_manager.get_entries()
        elif hasattr(queue_manager, "entries"):
            entries = queue_manager.entries

        for entry in entries:
            entry_cat = entry.get("category", "") if isinstance(entry, dict) else getattr(entry, "category", "")
            for arm_id in arm_ids:
                if self._matches_arm(entry_cat, arm_id):
                    counts[arm_id] = counts.get(arm_id, 0) + 1

        return counts

    def _matches_arm(self, category: str, arm_id: str) -> bool:
        """Check if a queue entry category matches an arm_id."""
        if not category or not arm_id:
            return False
        # Direct match or substring containment
        cat_lower = category.lower()
        arm_lower = arm_id.lower()
        return cat_lower == arm_lower or arm_lower in cat_lower or cat_lower in arm_lower
