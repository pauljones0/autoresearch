"""
Attention head clustering: group attention heads by behavioural patterns.

Extracts per-head features (mean attention distance, entropy, local vs global
attention fractions, rare-token attention) and clusters them using k-means
implemented in pure PyTorch.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple

import sys, os
from ..schemas import HeadCluster


class HeadClusterer:
    """Cluster attention heads by their attention pattern characteristics."""

    def __init__(self, device: Optional[torch.device] = None,
                 local_window: int = 5, max_seq_for_attn: int = 512):
        """
        Args:
            device: compute device
            local_window: tokens within +/- this distance count as "local"
            max_seq_for_attn: truncate sequences to this length when computing
                              raw Q*K^T (Flash Attention doesn't return weights)
        """
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.local_window = local_window
        self.max_seq_for_attn = max_seq_for_attn

    # ------------------------------------------------------------------
    # Attention weight extraction (bypass Flash Attention)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _extract_qk_per_layer(self, model: nn.Module, x: torch.Tensor):
        """Extract Q, K projections from each layer for manual attention computation.

        Returns:
            list of (q, k) tensors per layer, each (B, T, n_head, head_dim)
        """
        raw = model._orig_mod if hasattr(model, "_orig_mod") else model

        # Architecture version check
        required_attrs = ["config", "cos", "sin", "transformer"]
        missing = [a for a in required_attrs if not hasattr(raw, a)]
        if missing:
            raise AttributeError(
                f"Model is missing required attributes for head clustering: {missing}. "
                f"Ensure the model matches the expected GPT architecture from train.py."
            )

        config = raw.config
        B, T = x.shape

        # Truncate sequence for memory
        T = min(T, self.max_seq_for_attn)
        x = x[:, :T]

        # Embed + norm (replicating GPT.forward up to block entry)
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            h = raw.transformer.wte(x)
            h = F.rms_norm(h, (h.size(-1),))
            x0 = h

            cos = raw.cos[:, :T]
            sin = raw.sin[:, :T]

            qk_pairs = []
            for i, block in enumerate(raw.transformer.h):
                h = raw.resid_lambdas[i] * h + raw.x0_lambdas[i] * x0
                normed = F.rms_norm(h, (h.size(-1),))

                attn = block.attn
                q = attn.c_q(normed).view(B, T, attn.n_head, attn.head_dim)
                k = attn.c_k(normed).view(B, T, attn.n_kv_head, attn.head_dim)

                # Apply RoPE
                d = q.shape[3] // 2
                q1, q2 = q[..., :d], q[..., d:]
                q = torch.cat([q1 * cos + q2 * sin, q1 * (-sin) + q2 * cos], 3)
                k1, k2 = k[..., :d], k[..., d:]
                k = torch.cat([k1 * cos + k2 * sin, k1 * (-sin) + k2 * cos], 3)

                # QK norm
                q = F.rms_norm(q, (q.size(-1),))
                k = F.rms_norm(k, (k.size(-1),))

                # Expand k if GQA (n_kv_head < n_head)
                if attn.n_kv_head < attn.n_head:
                    repeats = attn.n_head // attn.n_kv_head
                    k = k.unsqueeze(3).expand(B, T, attn.n_kv_head, repeats, attn.head_dim)
                    k = k.reshape(B, T, attn.n_head, attn.head_dim)

                qk_pairs.append((q.float(), k.float()))

                # Continue the forward pass for correct representations in later layers
                # Run the full block to get the next hidden state
                ve = raw.value_embeds[str(i)](x[:, :T]) if str(i) in raw.value_embeds else None
                h = h + block.attn(normed, ve, (cos, sin), raw.window_sizes[i])
                h = h + block.mlp(F.rms_norm(h, (h.size(-1),)))

        return qk_pairs

    @staticmethod
    def _compute_causal_attention(q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        """Compute causal attention weights from Q, K.

        Args:
            q, k: (B, T, n_head, head_dim)

        Returns:
            attn_weights: (B, n_head, T, T) with causal mask applied, softmaxed
        """
        B, T, H, D = q.shape
        # (B, H, T, D)
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        scale = D ** -0.5
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale  # (B, H, T, T)

        # Causal mask
        mask = torch.triu(torch.ones(T, T, device=q.device, dtype=torch.bool), diagonal=1)
        scores.masked_fill_(mask.unsqueeze(0).unsqueeze(0), float("-inf"))

        attn = torch.softmax(scores, dim=-1)
        return attn

    # ------------------------------------------------------------------
    # Per-head feature extraction
    # ------------------------------------------------------------------

    def _extract_head_features(
        self, attn_weights: torch.Tensor, token_ids: torch.Tensor,
        vocab_size: int
    ) -> torch.Tensor:
        """Extract behavioural features for each head.

        Args:
            attn_weights: (B, n_head, T, T)
            token_ids: (B, T)
            vocab_size: vocab size for rare-token detection

        Returns:
            features: (n_head, 5) — [mean_dist, entropy, frac_first, frac_local, frac_rare]
        """
        B, H, T, _ = attn_weights.shape

        # 1) Mean attention distance: weighted mean of |query_pos - key_pos|
        pos = torch.arange(T, device=attn_weights.device, dtype=torch.float32)
        # distance matrix (T, T): dist[i,j] = |i - j|
        dist = (pos.unsqueeze(0) - pos.unsqueeze(1)).abs()  # (T, T)
        mean_dist = (attn_weights * dist.unsqueeze(0).unsqueeze(0)).sum(dim=-1).mean(dim=(0, 2))  # (H,)

        # 2) Attention entropy: -sum(p * log(p))
        eps = 1e-10
        log_attn = (attn_weights + eps).log()
        entropy = -(attn_weights * log_attn).sum(dim=-1).mean(dim=(0, 2))  # (H,)

        # 3) Fraction of attention on first token (position 0)
        frac_first = attn_weights[:, :, :, 0].mean(dim=(0, 2))  # (H,)

        # 4) Fraction of attention on local tokens (within +/- local_window)
        local_mask = dist <= self.local_window  # (T, T)
        # Also must respect causality — already handled since attn_weights are causal
        frac_local = (attn_weights * local_mask.unsqueeze(0).unsqueeze(0).float()).sum(dim=-1).mean(dim=(0, 2))  # (H,)

        # 5) Fraction of attention on rare tokens
        # Rare = tokens outside the top-1000 by frequency in this batch
        flat_ids = token_ids.reshape(-1)
        counts = torch.bincount(flat_ids, minlength=vocab_size)
        top1k = counts.topk(min(1000, vocab_size)).indices
        is_common = torch.zeros(vocab_size, device=attn_weights.device, dtype=torch.bool)
        is_common[top1k] = True
        # per-position rare indicator: (B, T)
        is_rare = ~is_common[token_ids]  # (B, T)
        # fraction of attention going to rare key positions
        rare_mask = is_rare.unsqueeze(1).unsqueeze(2).expand(B, H, T, T).float()  # (B, H, T, T)
        frac_rare = (attn_weights * rare_mask).sum(dim=-1).mean(dim=(0, 2))  # (H,)

        features = torch.stack([mean_dist, entropy, frac_first, frac_local, frac_rare], dim=-1)  # (H, 5)
        return features

    # ------------------------------------------------------------------
    # K-means clustering (pure PyTorch)
    # ------------------------------------------------------------------

    @staticmethod
    def _kmeans(X: torch.Tensor, k: int, max_iters: int = 50) -> Tuple[torch.Tensor, torch.Tensor]:
        """Simple k-means clustering.

        Args:
            X: (N, D) feature matrix
            k: number of clusters

        Returns:
            labels: (N,) cluster assignments
            centroids: (k, D)
        """
        N, D = X.shape
        k = min(k, N)

        # Initialize centroids via k-means++ style
        indices = [torch.randint(N, (1,)).item()]
        for _ in range(1, k):
            dists = torch.cdist(X, X[indices])  # (N, len(indices))
            min_dists = dists.min(dim=1).values  # (N,)
            # Pick next centroid proportional to squared distance
            probs = min_dists ** 2
            probs_sum = probs.sum()
            if probs_sum > 0:
                probs = probs / probs_sum
                idx = torch.multinomial(probs, 1).item()
            else:
                idx = torch.randint(N, (1,)).item()
            indices.append(idx)

        centroids = X[indices].clone()  # (k, D)

        for _ in range(max_iters):
            # Assign
            dists = torch.cdist(X, centroids)  # (N, k)
            labels = dists.argmin(dim=1)  # (N,)

            # Update
            new_centroids = torch.zeros_like(centroids)
            for c in range(k):
                mask = labels == c
                if mask.any():
                    new_centroids[c] = X[mask].mean(dim=0)
                else:
                    new_centroids[c] = centroids[c]

            if torch.allclose(centroids, new_centroids, atol=1e-6):
                break
            centroids = new_centroids

        return labels, centroids

    # ------------------------------------------------------------------
    # Cluster labeling heuristics
    # ------------------------------------------------------------------

    @staticmethod
    def _classify_cluster(centroid: torch.Tensor) -> str:
        """Classify a cluster based on its centroid features.

        Feature order: [mean_dist, entropy, frac_first, frac_local, frac_rare]
        """
        mean_dist, entropy, frac_first, frac_local, frac_rare = centroid.tolist()

        # Positional: high local attention fraction
        if frac_local > 0.6:
            return "positional"
        # Rare token specialist
        if frac_rare > 0.3:
            return "rare_token"
        # Syntactic: high entropy with medium distance
        if entropy > 2.0 and mean_dist > 5.0:
            return "syntactic"
        return "mixed"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @torch.no_grad()
    def cluster_heads(
        self, model: nn.Module, sample_batch, n_clusters: int = 4
    ) -> List[HeadCluster]:
        """Cluster attention heads by their behavioural patterns.

        Args:
            model: the GPT model (compiled or not)
            sample_batch: (x, y, ...) or tensor of token ids (B, T)
            n_clusters: number of clusters for k-means

        Returns:
            List of HeadCluster dataclass instances.
        """
        if isinstance(sample_batch, (list, tuple)):
            x = sample_batch[0]
        else:
            x = sample_batch
        x = x.to(self.device)

        # Use a small batch for efficiency
        max_batch = 4
        if x.shape[0] > max_batch:
            x = x[:max_batch]

        raw = model._orig_mod if hasattr(model, "_orig_mod") else model
        config = raw.config
        n_layers = config.n_layer
        n_head = config.n_head
        vocab_size = config.vocab_size

        # Extract Q, K for each layer and compute attention weights
        qk_pairs = self._extract_qk_per_layer(model, x)

        all_features = []  # (n_layers * n_head, 5)
        head_indices = []  # (layer_idx, head_idx)

        for layer_idx, (q, k) in enumerate(qk_pairs):
            attn_weights = self._compute_causal_attention(q, k)  # (B, H, T, T)
            T = q.shape[1]
            token_ids = x[:, :T]
            features = self._extract_head_features(attn_weights, token_ids, vocab_size)  # (H, 5)
            all_features.append(features.cpu())
            for h in range(n_head):
                head_indices.append((layer_idx, h))

        all_features = torch.cat(all_features, dim=0)  # (total_heads, 5)

        # Normalize features for clustering
        feat_mean = all_features.mean(dim=0, keepdim=True)
        feat_std = all_features.std(dim=0, keepdim=True).clamp(min=1e-6)
        normed = (all_features - feat_mean) / feat_std

        n_clusters = min(n_clusters, len(head_indices))
        labels, centroids = self._kmeans(normed, n_clusters)

        # Un-normalize centroids back to original feature space
        centroids_orig = centroids * feat_std + feat_mean

        # Build HeadCluster results
        feature_names = ["mean_dist", "entropy", "frac_first", "frac_local", "frac_rare"]
        results = []
        for c in range(n_clusters):
            mask = (labels == c).nonzero(as_tuple=True)[0]
            if len(mask) == 0:
                continue
            heads = [head_indices[i] for i in mask.tolist()]
            centroid_dict = {
                name: round(val, 4)
                for name, val in zip(feature_names, centroids_orig[c].tolist())
            }
            pattern = self._classify_cluster(centroids_orig[c])
            results.append(HeadCluster(
                cluster_id=c,
                pattern_type=pattern,
                head_indices=heads,
                centroid_stats=centroid_dict,
            ))

        return results
