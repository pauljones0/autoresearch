"""
IngestionBiasAgent: uses source quality data to apply an additive
relevance boost to papers during ingestion filtering.
"""

import json
import os


class IngestionBiasAgent:
    """Biases paper ingestion based on historical source quality."""

    MAX_BOOST = 0.3
    MIN_RELEVANCE_FRACTION = 0.10  # minimum relevance as fraction of max score

    def compute_source_boost(self, paper: dict, source_quality: dict) -> float:
        """Compute an additive relevance boost from source quality data.

        Args:
            paper: dict with keys authors, categories (and optionally first_author).
            source_quality: dict keyed by dimension (author, category, venue),
                each value is a dict mapping specific values to quality records
                with at least a 'success_rate' field.

        Returns:
            float boost in [0, MAX_BOOST].
        """
        boosts = []

        # Check first author
        authors = paper.get("authors", [])
        first_author = paper.get("first_author", "")
        if not first_author and authors:
            first_author = authors[0] if isinstance(authors[0], str) else ""

        author_quality = source_quality.get("author", {})
        if first_author and first_author in author_quality:
            rate = author_quality[first_author].get("success_rate", 0.0)
            boosts.append(rate)

        # Check all authors
        for author in authors:
            if isinstance(author, str) and author in author_quality:
                rate = author_quality[author].get("success_rate", 0.0)
                boosts.append(rate)

        # Check categories
        category_quality = source_quality.get("category", {})
        for cat in paper.get("categories", []):
            if isinstance(cat, str) and cat in category_quality:
                rate = category_quality[cat].get("success_rate", 0.0)
                boosts.append(rate)

        # Check venue
        venue_quality = source_quality.get("venue", {})
        venue = paper.get("venue", "")
        if venue and venue in venue_quality:
            rate = venue_quality[venue].get("success_rate", 0.0)
            boosts.append(rate)

        if not boosts:
            return 0.0

        # Average success rate, scaled to max boost
        avg_rate = sum(boosts) / len(boosts)
        return min(avg_rate * self.MAX_BOOST, self.MAX_BOOST)

    def apply_bias(self, papers: list, source_quality_path: str) -> list:
        """Apply source quality bias to a list of paper dicts.

        Loads source quality from JSON file, computes boost for each paper,
        and ensures minimum relevance floor.

        Args:
            papers: list of paper dicts (with relevance_score, authors, categories).
            source_quality_path: path to paper_source_quality.json.

        Returns:
            list of papers with source_quality_boost filled in.
        """
        source_quality = self._load_source_quality(source_quality_path)

        # Determine max score for minimum relevance floor
        max_score = 0.0
        for p in papers:
            score = p.get("relevance_score", 0.0) + p.get("keyword_score", 0.0)
            if score > max_score:
                max_score = score

        min_relevance = max_score * self.MIN_RELEVANCE_FRACTION if max_score > 0 else 0.0

        result = []
        for p in papers:
            paper = dict(p)
            boost = self.compute_source_boost(paper, source_quality)
            paper["source_quality_boost"] = boost

            # Apply minimum relevance floor
            current_score = paper.get("relevance_score", 0.0)
            if current_score + boost < min_relevance and max_score > 0:
                paper["source_quality_boost"] = max(boost, min_relevance - current_score)

            result.append(paper)

        return result

    @staticmethod
    def _load_source_quality(path: str) -> dict:
        """Load source quality data from JSON."""
        try:
            with open(path) as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {}
