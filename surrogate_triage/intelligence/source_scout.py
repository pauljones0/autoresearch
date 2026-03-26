"""
NewSourceScout: allocates a fraction of extraction slots to papers
from previously unseen sources to maintain exploration.
"""

import json
import os
import time


class NewSourceScout:
    """Scouts for papers from new/unknown sources."""

    def __init__(self, known_sources_path: str = None):
        self.known_sources_path = known_sources_path

    DEFAULT_EXPLORATION_FRACTION = 0.15

    def identify_new_sources(self, papers: list, known_sources_path: str) -> list:
        """Identify papers from authors or venues not in the known sources database.

        Args:
            papers: list of paper dicts with 'authors', 'categories'.
            known_sources_path: path to known_sources.json.

        Returns:
            list of papers that come from at least one unknown source.
        """
        known = self._load_known_sources(known_sources_path)
        known_authors = set(known.get("authors", []))
        known_venues = set(known.get("venues", []))

        new_source_papers = []
        for p in papers:
            is_new = False
            for author in p.get("authors", []):
                if isinstance(author, str) and author not in known_authors:
                    is_new = True
                    break
            if not is_new:
                venue = p.get("venue", "")
                if venue and venue not in known_venues:
                    is_new = True
            if is_new:
                new_source_papers.append(p)

        return new_source_papers

    def allocate_exploration_slots(
        self,
        papers: list,
        total_slots: int,
        exploration_fraction: float = 0.15,
    ) -> tuple:
        """Split papers into exploration (new sources) and exploitation lists.

        The exploration list is capped at exploration_fraction * total_slots.
        Remaining slots go to exploitation (papers not in exploration set).

        Args:
            papers: list of paper dicts, assumed pre-tagged. Papers with
                '_is_new_source' == True go to exploration pool.
            total_slots: total number of extraction slots available.
            exploration_fraction: fraction of slots reserved for exploration.

        Returns:
            (exploration_papers, exploitation_papers) tuple of lists.
        """
        if total_slots <= 0:
            return [], []

        explore_slots = max(1, int(total_slots * exploration_fraction))
        exploit_slots = total_slots - explore_slots

        explore_pool = [p for p in papers if p.get("_is_new_source", False)]
        exploit_pool = [p for p in papers if not p.get("_is_new_source", False)]

        # Sort exploration by relevance_score descending (best new-source papers first)
        explore_pool.sort(key=lambda x: x.get("relevance_score", 0.0), reverse=True)
        exploit_pool.sort(key=lambda x: x.get("relevance_score", 0.0), reverse=True)

        exploration = explore_pool[:explore_slots]
        exploitation = exploit_pool[:exploit_slots]

        return exploration, exploitation

    def update_known_sources(self, papers: list, known_sources_path: str):
        """Add authors and venues from processed papers to known sources.

        Call this after each ingestion cycle to keep the database current.

        Args:
            papers: list of paper dicts that were processed.
            known_sources_path: path to known_sources.json.
        """
        known = self._load_known_sources(known_sources_path)
        authors = set(known.get("authors", []))
        venues = set(known.get("venues", []))

        for p in papers:
            for author in p.get("authors", []):
                if isinstance(author, str) and author:
                    authors.add(author)
            venue = p.get("venue", "")
            if venue:
                venues.add(venue)

        known["authors"] = sorted(authors)
        known["venues"] = sorted(venues)
        known["last_updated"] = time.time()

        try:
            parent = os.path.dirname(known_sources_path)
            if parent:
                os.makedirs(parent, exist_ok=True)
            with open(known_sources_path, "w") as f:
                json.dump(known, f, indent=2)
        except OSError:
            pass

    @staticmethod
    def _load_known_sources(path: str) -> dict:
        """Load known sources database."""
        try:
            with open(path) as f:
                data = json.load(f)
            if isinstance(data, dict):
                return data
        except (FileNotFoundError, json.JSONDecodeError):
            pass
        return {"authors": [], "venues": [], "last_updated": 0.0}
