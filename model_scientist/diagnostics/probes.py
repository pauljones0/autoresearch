"""
Interpretability probes: train lightweight linear classifiers on frozen
intermediate representations to measure what information each layer encodes.

Synthetic probe tasks (no external labeled data needed):
  - position_prediction: can the representation predict its absolute position?
  - token_identity: can later layers still reconstruct the input token?
  - local_context: can the representation predict the next token?
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional

import sys, os
from ..schemas import ProbeResult


class ProbeTrainer:
    """Registers forward hooks on transformer blocks, captures intermediate
    representations, and trains linear probing classifiers on them."""

    def __init__(self, device: Optional[torch.device] = None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._hooks = []
        self._activations = {}

    # ------------------------------------------------------------------
    # Hook management
    # ------------------------------------------------------------------

    def _make_hook(self, layer_idx: int):
        """Create a forward hook that stores the block output."""
        def hook_fn(module, input, output):
            # Block.forward returns a tensor (the residual-stream state)
            if isinstance(output, torch.Tensor):
                self._activations[layer_idx] = output.detach()
            elif isinstance(output, tuple):
                self._activations[layer_idx] = output[0].detach()
        return hook_fn

    def _register_hooks(self, model: nn.Module):
        """Register hooks on each transformer block."""
        self._remove_hooks()
        self._activations.clear()

        # Handle torch.compile wrapped models
        raw = model._orig_mod if hasattr(model, "_orig_mod") else model

        blocks = raw.transformer.h
        for i, block in enumerate(blocks):
            h = block.register_forward_hook(self._make_hook(i))
            self._hooks.append(h)

    def _remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
        self._activations.clear()

    # ------------------------------------------------------------------
    # Representation collection
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _collect_representations(
        self, model: nn.Module, dataloader, num_batches: int
    ):
        """Run forward passes and collect per-layer activations plus metadata.

        Returns:
            layer_acts: dict[int, Tensor]  — (total_tokens, n_embd)
            positions:  Tensor              — (total_tokens,)
            token_ids:  Tensor              — (total_tokens,)
            next_ids:   Tensor              — (total_tokens,)  (-1 for last pos)
        """
        self._register_hooks(model)
        model.eval()

        all_positions = []
        all_token_ids = []
        all_next_ids = []
        layer_chunks = {}  # layer_idx -> list of tensors

        try:
            for batch_idx, batch in enumerate(dataloader):
                if batch_idx >= num_batches:
                    break

                # Unpack — dataloader yields (x, y, epoch) or (x, y)
                if isinstance(batch, (list, tuple)):
                    x = batch[0]
                    y = batch[1] if len(batch) > 1 else None
                else:
                    x = batch
                    y = None

                x = x.to(self.device)
                B, T = x.shape

                # Forward pass to trigger hooks (use the original model call)
                raw = model._orig_mod if hasattr(model, "_orig_mod") else model
                with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                    _ = raw(x)

                # Positions: 0..T-1 repeated B times
                pos = torch.arange(T, device=self.device).unsqueeze(0).expand(B, T)
                all_positions.append(pos.reshape(-1))
                all_token_ids.append(x.reshape(-1))

                # Next token ids (use y if available, else shift x)
                if y is not None:
                    y = y.to(self.device)
                    all_next_ids.append(y.reshape(-1))
                else:
                    next_ids = torch.full((B, T), -1, dtype=x.dtype, device=self.device)
                    next_ids[:, :-1] = x[:, 1:]
                    all_next_ids.append(next_ids.reshape(-1))

                # Collect activations
                for layer_idx, act in self._activations.items():
                    # act: (B, T, n_embd)
                    flat = act.float().reshape(-1, act.size(-1))
                    layer_chunks.setdefault(layer_idx, []).append(flat.cpu())

                self._activations.clear()

        finally:
            self._remove_hooks()

        positions = torch.cat(all_positions, dim=0).cpu()
        token_ids = torch.cat(all_token_ids, dim=0).cpu()
        next_ids = torch.cat(all_next_ids, dim=0).cpu()

        layer_acts = {}
        for idx, chunks in layer_chunks.items():
            layer_acts[idx] = torch.cat(chunks, dim=0)

        return layer_acts, positions, token_ids, next_ids

    # ------------------------------------------------------------------
    # Linear probe training (closed-form least-squares for speed)
    # ------------------------------------------------------------------

    @staticmethod
    def _train_linear_probe_cls(X: torch.Tensor, labels: torch.Tensor,
                                num_classes: int, reg: float = 1e-3):
        """Train a linear classifier via ridge regression (closed-form).

        Args:
            X: (N, D) float features
            labels: (N,) long class labels
            num_classes: number of classes
            reg: L2 regularisation strength

        Returns:
            accuracy: float
        """
        N, D = X.shape
        # Subsample if too large (keep probes fast)
        max_samples = 50000
        if N > max_samples:
            perm = torch.randperm(N)[:max_samples]
            X = X[perm]
            labels = labels[perm]
            N = max_samples

        # One-hot targets
        Y = torch.zeros(N, num_classes, dtype=X.dtype, device=X.device)
        Y.scatter_(1, labels.unsqueeze(1).to(X.device), 1.0)

        # Ridge: W = (X^T X + lambda I)^{-1} X^T Y
        XtX = X.T @ X + reg * torch.eye(D, dtype=X.dtype, device=X.device)
        XtY = X.T @ Y
        try:
            W = torch.linalg.solve(XtX, XtY)
        except torch.linalg.LinAlgError:
            # Fallback: pseudo-inverse
            W = torch.linalg.lstsq(XtX, XtY).solution

        preds = (X @ W).argmax(dim=1)
        accuracy = (preds == labels.to(X.device)).float().mean().item()
        return accuracy

    @staticmethod
    def _train_linear_probe_regression(X: torch.Tensor, targets: torch.Tensor,
                                       reg: float = 1e-3):
        """Train a linear regressor via ridge regression.

        Returns:
            r_squared: float  (clamped to [0, 1])
        """
        N, D = X.shape
        max_samples = 50000
        if N > max_samples:
            perm = torch.randperm(N)[:max_samples]
            X = X[perm]
            targets = targets[perm]
            N = max_samples

        y = targets.float().to(X.device).unsqueeze(1)
        XtX = X.T @ X + reg * torch.eye(D, dtype=X.dtype, device=X.device)
        XtY = X.T @ y
        try:
            W = torch.linalg.solve(XtX, XtY)
        except torch.linalg.LinAlgError:
            W = torch.linalg.lstsq(XtX, XtY).solution

        y_pred = X @ W
        ss_res = ((y - y_pred) ** 2).sum().item()
        ss_tot = ((y - y.mean()) ** 2).sum().item()
        r2 = max(0.0, 1.0 - ss_res / max(ss_tot, 1e-12))
        return r2

    # ------------------------------------------------------------------
    # Probe tasks
    # ------------------------------------------------------------------

    def _probe_position_prediction(self, layer_acts, positions, seq_len):
        """Can the representation predict its absolute position in the sequence?

        We bucket positions into 16 bins and train a classifier.
        """
        results = []
        num_bins = min(16, seq_len)
        bin_size = max(1, seq_len // num_bins)
        binned_pos = (positions // bin_size).clamp(max=num_bins - 1).long()
        baseline = binned_pos.float().mode().values.item()
        baseline_acc = (binned_pos == int(baseline)).float().mean().item()

        for layer_idx in sorted(layer_acts.keys()):
            X = layer_acts[layer_idx]
            acc = self._train_linear_probe_cls(X, binned_pos, num_bins)
            results.append(ProbeResult(
                task_name="position_prediction",
                layer_idx=layer_idx,
                accuracy=round(acc, 4),
                baseline_accuracy=round(baseline_acc, 4),
            ))
        return results

    def _probe_token_identity(self, layer_acts, token_ids, vocab_size):
        """Can the representation reconstruct which token is at this position?

        Since vocab may be very large, we map tokens to their top-256 bucket
        (frequent tokens get their own class, rest go to an 'other' class).
        """
        results = []
        # Build a mapping: top-255 most frequent tokens keep their id, rest -> 255
        num_classes = 256
        counts = torch.bincount(token_ids, minlength=vocab_size)
        top_ids = counts.topk(num_classes - 1).indices
        id_map = torch.full((vocab_size,), num_classes - 1, dtype=torch.long)
        for new_id, old_id in enumerate(top_ids):
            id_map[old_id] = new_id

        mapped = id_map[token_ids]
        baseline_acc = mapped.float().mode().values.item()
        baseline_acc = (mapped == int(baseline_acc)).float().mean().item()

        for layer_idx in sorted(layer_acts.keys()):
            X = layer_acts[layer_idx]
            acc = self._train_linear_probe_cls(X, mapped, num_classes)
            results.append(ProbeResult(
                task_name="token_identity",
                layer_idx=layer_idx,
                accuracy=round(acc, 4),
                baseline_accuracy=round(baseline_acc, 4),
            ))
        return results

    def _probe_local_context(self, layer_acts, next_ids, vocab_size):
        """Can the representation predict the next token better than baseline?

        Same bucketing as token_identity to keep classes manageable.
        """
        results = []
        # Filter out positions where next_id == -1
        valid = next_ids >= 0
        if valid.sum() < 100:
            return results

        num_classes = 256
        counts = torch.bincount(next_ids[valid], minlength=vocab_size)
        top_ids = counts.topk(num_classes - 1).indices
        id_map = torch.full((vocab_size,), num_classes - 1, dtype=torch.long)
        for new_id, old_id in enumerate(top_ids):
            id_map[old_id] = new_id

        mapped_all = torch.full_like(next_ids, num_classes - 1)
        mapped_all[valid] = id_map[next_ids[valid]]

        valid_idx = valid.nonzero(as_tuple=True)[0]
        mapped = mapped_all[valid_idx]
        baseline_acc_val = mapped.float().mode().values.item()
        baseline_acc = (mapped == int(baseline_acc_val)).float().mean().item()

        for layer_idx in sorted(layer_acts.keys()):
            X = layer_acts[layer_idx][valid_idx]
            acc = self._train_linear_probe_cls(X, mapped, num_classes)
            results.append(ProbeResult(
                task_name="local_context",
                layer_idx=layer_idx,
                accuracy=round(acc, 4),
                baseline_accuracy=round(baseline_acc, 4),
            ))
        return results

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train_probes(self, model: nn.Module, dataloader,
                     num_batches: int = 10) -> List[ProbeResult]:
        """Train all probing classifiers and return results.

        Args:
            model: the GPT model (compiled or not)
            dataloader: yields (x, y, epoch) batches
            num_batches: how many batches to collect representations from

        Returns:
            List of ProbeResult across all tasks and layers.
        """
        raw = model._orig_mod if hasattr(model, "_orig_mod") else model
        config = raw.config
        seq_len = config.sequence_len
        vocab_size = config.vocab_size

        layer_acts, positions, token_ids, next_ids = self._collect_representations(
            model, dataloader, num_batches
        )

        if not layer_acts:
            return []

        results = []
        results.extend(self._probe_position_prediction(layer_acts, positions, seq_len))
        results.extend(self._probe_token_identity(layer_acts, token_ids, vocab_size))
        results.extend(self._probe_local_context(layer_acts, next_ids, vocab_size))

        return results
