"""
Centered Kernel Alignment (CKA) for measuring representational similarity
between transformer layers.

CKA(X, Y) = HSIC(X, Y) / sqrt(HSIC(X, X) * HSIC(Y, Y))

Uses a linear kernel with proper centering matrix H = I - 1/n * 11^T.
"""

import torch
import torch.nn as nn
from typing import List, Optional

import sys, os
from ..schemas import LayerSimilarityEntry


class CKASimilarity:
    """Compute pairwise CKA similarity between all transformer layer
    representations on a single forward pass."""

    def __init__(self, device: Optional[torch.device] = None, max_samples: int = 4096):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.max_samples = max_samples
        self._hooks = []
        self._activations = {}

    # ------------------------------------------------------------------
    # Hook management
    # ------------------------------------------------------------------

    def _make_hook(self, layer_idx: int):
        def hook_fn(module, input, output):
            if isinstance(output, torch.Tensor):
                self._activations[layer_idx] = output.detach()
            elif isinstance(output, tuple):
                self._activations[layer_idx] = output[0].detach()
        return hook_fn

    def _register_hooks(self, model: nn.Module):
        self._remove_hooks()
        self._activations.clear()
        raw = model._orig_mod if hasattr(model, "_orig_mod") else model
        for i, block in enumerate(raw.transformer.h):
            h = block.register_forward_hook(self._make_hook(i))
            self._hooks.append(h)

    def _remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
        self._activations.clear()

    # ------------------------------------------------------------------
    # CKA computation
    # ------------------------------------------------------------------

    @staticmethod
    def _hsic(X: torch.Tensor, Y: torch.Tensor) -> float:
        """Compute HSIC with linear kernel and centering.

        X, Y: (n, d) centered feature matrices.
        HSIC(X, Y) = ||Y^T H X||_F^2 / (n-1)^2
        where H = I - 1/n * 11^T  (centering is applied to X, Y beforehand).
        """
        n = X.shape[0]
        if n < 2:
            return 0.0
        # Since X and Y are already centered, H @ X = X
        # HSIC = trace(K_X H K_Y H) / (n-1)^2
        # With linear kernel: K_X = X X^T, so
        # trace(X X^T H Y Y^T H) = ||Y^T X||_F^2  (when X,Y centered)
        YtX = Y.T @ X  # (d_y, d_x)
        val = (YtX * YtX).sum().item()
        return val / ((n - 1) ** 2)

    @staticmethod
    def _center(X: torch.Tensor) -> torch.Tensor:
        """Center columns: H @ X where H = I - 1/n * 11^T."""
        return X - X.mean(dim=0, keepdim=True)

    def _compute_cka(self, X: torch.Tensor, Y: torch.Tensor) -> float:
        """Compute CKA between two representation matrices."""
        X = self._center(X.float())
        Y = self._center(Y.float())
        hsic_xy = self._hsic(X, Y)
        hsic_xx = self._hsic(X, X)
        hsic_yy = self._hsic(Y, Y)
        denom = (hsic_xx * hsic_yy) ** 0.5
        if denom < 1e-12:
            return 0.0
        return hsic_xy / denom

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @torch.no_grad()
    def compute_similarity_matrix(
        self, model: nn.Module, sample_batch
    ) -> List[LayerSimilarityEntry]:
        """Compute pairwise CKA similarity between all layer representations.

        Args:
            model: the GPT model (compiled or not)
            sample_batch: a single batch — either a tensor of token ids (B, T)
                or a tuple (x, y, ...) where x is the token ids.

        Returns:
            List of LayerSimilarityEntry for every (i, j) pair where i <= j.
        """
        # Parse input
        if isinstance(sample_batch, (list, tuple)):
            x = sample_batch[0]
        else:
            x = sample_batch
        x = x.to(self.device)

        self._register_hooks(model)
        model.eval()

        try:
            raw = model._orig_mod if hasattr(model, "_orig_mod") else model
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                _ = raw(x)
        finally:
            # Grab activations before removing hooks
            acts = {}
            for idx, act in self._activations.items():
                # Flatten to (B*T, D), subsample if needed
                flat = act.float().reshape(-1, act.size(-1))
                if flat.shape[0] > self.max_samples:
                    perm = torch.randperm(flat.shape[0], device=flat.device)[:self.max_samples]
                    flat = flat[perm]
                acts[idx] = flat.cpu()
            self._remove_hooks()

        if not acts:
            return []

        layer_indices = sorted(acts.keys())
        results = []

        # Compute pairwise CKA (memory-efficient: only two layers at a time)
        for i_pos, li in enumerate(layer_indices):
            for lj in layer_indices[i_pos:]:
                cka = self._compute_cka(acts[li], acts[lj])
                results.append(LayerSimilarityEntry(
                    layer_i=li,
                    layer_j=lj,
                    cka_score=round(cka, 6),
                ))

        return results
