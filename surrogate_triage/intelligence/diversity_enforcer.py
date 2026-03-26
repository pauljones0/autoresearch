"""
SourceDiversityEnforcer: prevents source bias from collapsing paper
diversity by throttling over-represented sources.
"""

import math
import time


class SourceDiversityEnforcer:
    """Enforces diversity in paper source distribution."""

    DEFAULT_MAX_FRACTION = 0.30
    DEFAULT_WINDOW_DAYS = 7
    # Shannon entropy threshold: for N categories, max entropy = ln(N).
    # Alert if entropy drops below 50% of max possible.
    ENTROPY_ALERT_FRACTION = 0.50

    def check_diversity(self, recent_papers: list, window_days: int = 7) -> dict:
        """Check distribution of papers across source dimensions.

        Args:
            recent_papers: list of paper dicts with 'authors', 'categories',
                and optionally 'fetched_at' (epoch timestamp).
            window_days: only consider papers fetched within this window.

        Returns:
            dict mapping source key (e.g. "author:Jane", "category:cs.LG")
            to its fraction of total papers.
        """
        cutoff = time.time() - (window_days * 86400)
        filtered = []
        for p in recent_papers:
            fetched = p.get("fetched_at", 0.0)
            if fetched >= cutoff or fetched == 0.0:
                filtered.append(p)

        if not filtered:
            return {}

        counts = {}
        total_entries = 0

        for p in filtered:
            # Count authors
            for author in p.get("authors", []):
                if isinstance(author, str) and author:
                    key = "author:" + author
                    counts[key] = counts.get(key, 0) + 1
                    total_entries += 1
            # Count categories
            for cat in p.get("categories", []):
                if isinstance(cat, str) and cat:
                    key = "category:" + cat
                    counts[key] = counts.get(key, 0) + 1
                    total_entries += 1
            # Count venue
            venue = p.get("venue", "")
            if venue:
                key = "venue:" + venue
                counts[key] = counts.get(key, 0) + 1
                total_entries += 1

        if total_entries == 0:
            return {}

        return {k: v / total_entries for k, v in counts.items()}

    def apply_throttle(
        self, papers: list, diversity_stats: dict, max_fraction: float = 0.30
    ) -> list:
        """Throttle source_quality_boost for over-represented sources.

        For any source exceeding max_fraction, scale down its papers' boost
        proportionally.

        Args:
            papers: list of paper dicts with source_quality_boost.
            diversity_stats: output from check_diversity().
            max_fraction: maximum allowed fraction for any single source.

        Returns:
            list of papers with throttled boosts.
        """
        # Identify over-represented sources
        over_represented = {}
        for source_key, fraction in diversity_stats.items():
            if fraction > max_fraction:
                # Throttle factor: how much to reduce boost
                over_represented[source_key] = max_fraction / fraction

        if not over_represented:
            return papers

        result = []
        for p in papers:
            paper = dict(p)
            throttle_factor = 1.0

            # Check if this paper belongs to any over-represented source
            for author in paper.get("authors", []):
                key = "author:" + str(author)
                if key in over_represented:
                    throttle_factor = min(throttle_factor, over_represented[key])
            for cat in paper.get("categories", []):
                key = "category:" + str(cat)
                if key in over_represented:
                    throttle_factor = min(throttle_factor, over_represented[key])
            venue = paper.get("venue", "")
            if venue:
                key = "venue:" + venue
                if key in over_represented:
                    throttle_factor = min(throttle_factor, over_represented[key])

            if throttle_factor < 1.0:
                current_boost = paper.get("source_quality_boost", 0.0)
                paper["source_quality_boost"] = current_boost * throttle_factor

            result.append(paper)

        return result

    def compute_shannon_entropy(self, diversity_stats: dict) -> float:
        """Compute Shannon entropy of the source distribution.

        Returns entropy in nats. Higher = more diverse.
        """
        fractions = [v for v in diversity_stats.values() if v > 0]
        if not fractions:
            return 0.0
        total = sum(fractions)
        entropy = 0.0
        for f in fractions:
            p = f / total
            if p > 0:
                entropy -= p * math.log(p)
        return entropy

    def check_entropy_alert(self, diversity_stats: dict) -> dict:
        """Check if diversity has dropped below acceptable threshold.

        Returns a dict with 'alert' (bool), 'entropy', 'max_entropy', and 'threshold'.
        """
        n_sources = len([v for v in diversity_stats.values() if v > 0])
        if n_sources <= 1:
            return {"alert": False, "entropy": 0.0, "max_entropy": 0.0, "threshold": 0.0}

        entropy = self.compute_shannon_entropy(diversity_stats)
        max_entropy = math.log(n_sources)
        threshold = max_entropy * self.ENTROPY_ALERT_FRACTION

        return {
            "alert": entropy < threshold,
            "entropy": entropy,
            "max_entropy": max_entropy,
            "threshold": threshold,
        }
