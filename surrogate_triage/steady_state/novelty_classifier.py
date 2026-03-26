"""
Phase 5 — NoveltyClassifier: classify paper techniques as genuinely novel
vs LLM-proposable by measuring distance to internally-generated diffs.
"""

import math

from surrogate_triage.surrogate.diff_embedder import DiffEmbedder


def _cosine_distance(a: list, b: list) -> float:
    """Compute cosine distance (1 - cosine_similarity) between two vectors.

    Returns 1.0 (maximum distance) if either vector is zero.
    """
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a < 1e-12 or norm_b < 1e-12:
        return 1.0
    similarity = dot / (norm_a * norm_b)
    # Clamp for floating-point safety
    similarity = max(-1.0, min(1.0, similarity))
    return 1.0 - similarity


class NoveltyClassifier:
    """Classify paper techniques as genuinely novel vs LLM-proposable.

    Embeds the paper technique's diff and computes nearest-neighbor distance
    to all internally-generated diffs in the hypothesis journal.  High minimum
    distance indicates a genuinely novel technique.
    """

    def __init__(self, novelty_threshold: float = 0.7, embed_dim: int = 256):
        self.novelty_threshold = novelty_threshold
        self.embed_dim = embed_dim

    def classify(
        self,
        diff_text: str,
        journal_diffs: list,
        embedder: DiffEmbedder | None = None,
    ) -> dict:
        """Classify a paper technique by novelty.

        Args:
            diff_text: The paper technique's diff text.
            journal_diffs: list of dicts with at least 'modification_diff' and 'id'
                from hypothesis_journal.jsonl (internally-generated experiments).
            embedder: Optional DiffEmbedder instance; created if not provided.

        Returns:
            dict with novelty_score, is_novel, nearest_distance, nearest_journal_id.
        """
        if embedder is None:
            embedder = DiffEmbedder()

        if not diff_text or not diff_text.strip():
            return {
                "novelty_score": 1.0,
                "is_novel": True,
                "nearest_distance": 1.0,
                "nearest_journal_id": "",
            }

        paper_vec = embedder.embed(diff_text, dim=self.embed_dim)

        if not journal_diffs:
            return {
                "novelty_score": 1.0,
                "is_novel": True,
                "nearest_distance": 1.0,
                "nearest_journal_id": "",
            }

        nearest_dist = float("inf")
        nearest_id = ""

        for entry in journal_diffs:
            jdiff = entry.get("modification_diff", "")
            if not jdiff or not jdiff.strip():
                continue
            jvec = embedder.embed(jdiff, dim=self.embed_dim)
            dist = _cosine_distance(paper_vec, jvec)
            if dist < nearest_dist:
                nearest_dist = dist
                nearest_id = entry.get("id", "")

        # If no valid journal diffs, treat as fully novel
        if nearest_dist == float("inf"):
            nearest_dist = 1.0

        # Novelty score is the nearest-neighbor distance (0..1 for cosine)
        novelty_score = min(nearest_dist, 1.0)

        return {
            "novelty_score": novelty_score,
            "is_novel": novelty_score > self.novelty_threshold,
            "nearest_distance": nearest_dist,
            "nearest_journal_id": nearest_id,
        }
