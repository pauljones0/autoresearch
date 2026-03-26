"""
Phase 2.1 — FailureClusterer: cluster FailureFeatures by similarity
using k-means implemented from scratch with PyTorch.
"""

import json
import math
from dataclasses import asdict

import torch

from model_scientist.schemas import FailureFeatures, FailurePattern
from model_scientist.failure_mining.extractor import FailureExtractor


class FailureClusterer:
    """Cluster failure features using k-means (PyTorch, no sklearn)."""

    def __init__(self, max_iter: int = 100, tol: float = 1e-6, seed: int = 42):
        self.max_iter = max_iter
        self.tol = tol
        self.seed = seed

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def cluster(
        self,
        failures: list[FailureFeatures],
        max_clusters: int = 8,
    ) -> list[FailurePattern]:
        """Cluster *failures* and return a list of FailurePattern."""
        if not failures:
            return []

        # Build feature matrix.
        vecs = [FailureExtractor.extract_features_vector(f) for f in failures]
        X = torch.tensor(vecs, dtype=torch.float32)

        # Normalise columns to [0, 1] so distance is meaningful.
        X, col_min, col_range = self._normalise(X)

        # Choose k via elbow method.
        k = self._choose_k(X, max_clusters)

        # Run k-means.
        labels, centroids = self._kmeans(X, k)

        # Build FailurePattern objects.
        patterns: list[FailurePattern] = []
        for cluster_id in range(k):
            mask = labels == cluster_id
            indices = torch.nonzero(mask, as_tuple=False).squeeze(-1).tolist()
            if isinstance(indices, int):
                indices = [indices]
            if not indices:
                continue

            cluster_failures = [failures[i] for i in indices]
            deltas = [f.actual_delta for f in cluster_failures]
            avg_delta = sum(deltas) / len(deltas) if deltas else 0.0

            # Determine dominant modification type.
            cat_counts: dict[str, int] = {}
            for f in cluster_failures:
                cat_counts[f.modification_category] = cat_counts.get(f.modification_category, 0) + 1
            dominant_cat = max(cat_counts, key=cat_counts.get)  # type: ignore[arg-type]

            # Determine dominant failure mode.
            mode_counts: dict[str, int] = {}
            for f in cluster_failures:
                mode_counts[f.failure_mode] = mode_counts.get(f.failure_mode, 0) + 1
            dominant_mode = max(mode_counts, key=mode_counts.get)  # type: ignore[arg-type]

            # Un-normalise centroid for interpretability.
            raw_centroid = centroids[cluster_id] * col_range + col_min
            centroid_dict = {"raw": raw_centroid.tolist()}

            description = (
                f"Cluster of {len(indices)} failures — "
                f"dominant category: {dominant_cat}, "
                f"dominant failure mode: {dominant_mode}, "
                f"avg delta: {avg_delta:+.5f}"
            )

            patterns.append(FailurePattern(
                pattern_id=cluster_id,
                description=description,
                modification_type=dominant_cat,
                instance_count=len(indices),
                instance_ids=[f.journal_id for f in cluster_failures],
                centroid_features=centroid_dict,
                avg_actual_delta=avg_delta,
            ))

        return patterns

    @staticmethod
    def save(patterns: list[FailurePattern], path: str) -> None:
        """Save patterns to a JSON file."""
        data = [asdict(p) for p in patterns]
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalise(X: torch.Tensor):
        """Min-max normalise columns. Returns (X_norm, col_min, col_range)."""
        col_min = X.min(dim=0).values
        col_max = X.max(dim=0).values
        col_range = col_max - col_min
        col_range[col_range == 0] = 1.0  # avoid division by zero
        X_norm = (X - col_min) / col_range
        return X_norm, col_min, col_range

    def _kmeans_pp_init(self, X: torch.Tensor, k: int) -> torch.Tensor:
        """K-means++ centroid initialisation."""
        n = X.size(0)
        gen = torch.Generator()
        gen.manual_seed(self.seed)

        # First centroid chosen uniformly at random.
        idx = torch.randint(n, (1,), generator=gen).item()
        centroids = [X[idx]]

        for _ in range(1, k):
            dists = torch.stack([
                ((X - c.unsqueeze(0)) ** 2).sum(dim=1)
                for c in centroids
            ], dim=1)
            min_dists = dists.min(dim=1).values
            # Weighted probability proportional to distance squared.
            probs = min_dists / (min_dists.sum() + 1e-12)
            cum = probs.cumsum(dim=0)
            r = torch.rand(1, generator=gen).item()
            chosen = int((cum >= r).nonzero(as_tuple=False)[0].item())
            centroids.append(X[chosen])

        return torch.stack(centroids)

    def _kmeans(self, X: torch.Tensor, k: int):
        """Run k-means. Returns (labels, centroids)."""
        n = X.size(0)
        if k >= n:
            k = max(1, n)

        centroids = self._kmeans_pp_init(X, k)

        for _ in range(self.max_iter):
            # Assign each point to nearest centroid.
            dists = torch.cdist(X, centroids)  # (n, k)
            labels = dists.argmin(dim=1)

            # Update centroids.
            new_centroids = torch.zeros_like(centroids)
            for c in range(k):
                mask = labels == c
                if mask.any():
                    new_centroids[c] = X[mask].mean(dim=0)
                else:
                    new_centroids[c] = centroids[c]

            shift = ((new_centroids - centroids) ** 2).sum()
            centroids = new_centroids
            if shift < self.tol:
                break

        return labels, centroids

    def _inertia(self, X: torch.Tensor, k: int) -> float:
        """Compute within-cluster sum of squares for a given k."""
        labels, centroids = self._kmeans(X, k)
        total = 0.0
        for c in range(k):
            mask = labels == c
            if mask.any():
                total += ((X[mask] - centroids[c]) ** 2).sum().item()
        return total

    def _choose_k(self, X: torch.Tensor, max_clusters: int) -> int:
        """Pick k using the elbow method (largest drop in inertia)."""
        n = X.size(0)
        max_k = min(max_clusters, n)
        if max_k <= 2:
            return max(1, max_k)

        inertias: list[float] = []
        ks = list(range(2, max_k + 1))
        for k in ks:
            inertias.append(self._inertia(X, k))

        if len(inertias) <= 1:
            return ks[0]

        # Elbow = point with maximum second derivative (largest curvature).
        best_k = ks[0]
        best_curvature = -math.inf
        for i in range(1, len(inertias) - 1):
            curvature = (inertias[i - 1] - inertias[i]) - (inertias[i] - inertias[i + 1])
            if curvature > best_curvature:
                best_curvature = curvature
                best_k = ks[i]

        return best_k
