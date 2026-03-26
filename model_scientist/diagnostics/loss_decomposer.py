"""
LossDecomposer: decomposes cross-entropy loss by token frequency bucket.

Buckets:
  - top_1k: the 1000 most frequent tokens
  - 1k_10k: tokens ranked 1000-10000
  - 10k_plus: tokens ranked 10000+
  - rare: tokens appearing fewer than a threshold number of times

Usage:
    decomposer = LossDecomposer()
    freq = decomposer.compute_token_frequencies(tokenizer, data_iterator)
    buckets = decomposer.decompose(model, batch_x, batch_y, freq)
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..schemas import LossBucket


# Bucket definitions: (name, rank_start_inclusive, rank_end_exclusive)
# Ranks are 0-indexed by descending frequency.
BUCKET_DEFS = [
    ("top_1k", 0, 1000),
    ("1k_10k", 1000, 10000),
    ("10k_plus", 10000, None),  # None = up to vocab size
]

# Rare token threshold: tokens with frequency count below this
RARE_THRESHOLD = 10


class LossDecomposer:
    """Decomposes model loss by token frequency bucket."""

    def __init__(self, rare_threshold: int = RARE_THRESHOLD):
        self.rare_threshold = rare_threshold

    @staticmethod
    def compute_token_frequencies_from_data(
        tokenizer, data_batches: int = 50, batch_size: int = 128, seq_len: int = 2048
    ) -> torch.Tensor:
        """Compute token frequency counts from training data.

        Args:
            tokenizer: The Tokenizer instance.
            data_batches: Number of batches to sample for frequency estimation.
            batch_size: Batch size for the data loader.
            seq_len: Sequence length.

        Returns:
            1-D tensor of shape (vocab_size,) with token counts.
        """
        # Import here to avoid circular imports at module level
        from prepare import make_dataloader

        vocab_size = tokenizer.get_vocab_size()
        counts = torch.zeros(vocab_size, dtype=torch.long)

        loader = make_dataloader(tokenizer, batch_size, seq_len, "train")
        for i in range(data_batches):
            x, y, _ = next(loader)
            # Count tokens in targets (what the model predicts)
            y_cpu = y.cpu().view(-1)
            valid = y_cpu[y_cpu >= 0]  # ignore padding (-1)
            counts.scatter_add_(
                0, valid.long(), torch.ones_like(valid, dtype=torch.long)
            )

        return counts

    @staticmethod
    def frequencies_to_rank_map(token_frequencies: torch.Tensor) -> torch.Tensor:
        """Convert frequency counts to rank indices (0 = most frequent).

        Returns:
            1-D tensor of shape (vocab_size,) mapping token_id -> rank.
        """
        # Sort by frequency descending; ties broken arbitrarily
        _, sorted_indices = token_frequencies.sort(descending=True)
        ranks = torch.empty_like(sorted_indices)
        ranks[sorted_indices] = torch.arange(len(sorted_indices))
        return ranks

    @torch.no_grad()
    def decompose(
        self,
        model: nn.Module,
        batch_x: torch.Tensor,
        batch_y: torch.Tensor,
        token_frequencies: torch.Tensor,
    ) -> list[dict]:
        """Decompose loss by token frequency bucket.

        Args:
            model: The GPT model.
            batch_x: Input token IDs, shape (B, T).
            batch_y: Target token IDs, shape (B, T).
            token_frequencies: 1-D tensor of token frequency counts.

        Returns:
            List of LossBucket dicts.
        """
        device = batch_x.device
        vocab_size = token_frequencies.size(0)

        # Compute per-token losses
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            loss_per_token = model(batch_x, batch_y, reduction="none")  # (B, T)
        loss_flat = loss_per_token.float().view(-1)
        y_flat = batch_y.view(-1)

        # Build rank map and bucket assignments
        ranks = self.frequencies_to_rank_map(token_frequencies).to(device)
        freq_on_device = token_frequencies.to(device)

        # Assign each token to a bucket
        token_ranks = ranks[y_flat.clamp(0, vocab_size - 1)]
        token_freqs = freq_on_device[y_flat.clamp(0, vocab_size - 1)]

        # Valid mask (ignore padding tokens with id=-1)
        valid_mask = y_flat >= 0

        results = []

        for bucket_name, rank_start, rank_end in BUCKET_DEFS:
            rank_end_val = rank_end if rank_end is not None else vocab_size

            bucket_mask = (
                valid_mask
                & (token_ranks >= rank_start)
                & (token_ranks < rank_end_val)
            )

            count = int(bucket_mask.sum().item())
            if count == 0:
                results.append(vars(LossBucket(
                    bucket_name=bucket_name,
                    token_count=0,
                    mean_loss=0.0,
                    std_loss=0.0,
                )))
                continue

            bucket_losses = loss_flat[bucket_mask]
            results.append(vars(LossBucket(
                bucket_name=bucket_name,
                token_count=count,
                mean_loss=bucket_losses.mean().item(),
                std_loss=bucket_losses.std().item() if count > 1 else 0.0,
            )))

        # Rare tokens bucket (by absolute frequency count)
        rare_mask = valid_mask & (token_freqs < self.rare_threshold)
        rare_count = int(rare_mask.sum().item())
        if rare_count > 0:
            rare_losses = loss_flat[rare_mask]
            results.append(vars(LossBucket(
                bucket_name="rare",
                token_count=rare_count,
                mean_loss=rare_losses.mean().item(),
                std_loss=rare_losses.std().item() if rare_count > 1 else 0.0,
            )))
        else:
            results.append(vars(LossBucket(
                bucket_name="rare",
                token_count=0,
                mean_loss=0.0,
                std_loss=0.0,
            )))

        return results
