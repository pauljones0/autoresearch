"""
Derives consistent model configs at reduced scales for multi-scale testing.
Scales by adjusting depth and width proportionally while keeping HEAD_DIM fixed.
"""

import math
from model_scientist.schemas import ScaleConfig


class ScaleConfigDeriver:
    """Derives model configs at fractional parameter scales."""

    def derive_configs(
        self,
        base_depth: int,
        base_aspect_ratio: int,
        head_dim: int,
        scales: list[float] | None = None,
    ) -> list[ScaleConfig]:
        if scales is None:
            scales = [0.25, 0.5, 1.0]

        base_dim = base_depth * base_aspect_ratio
        base_dim_rounded = ((base_dim + head_dim - 1) // head_dim) * head_dim
        base_n_head = base_dim_rounded // head_dim

        configs = []
        for s in sorted(scales):
            if s == 1.0:
                cfg = ScaleConfig(
                    scale_factor=1.0,
                    depth=base_depth,
                    n_embd=base_dim_rounded,
                    n_head=base_n_head,
                    n_kv_head=base_n_head,
                    estimated_params=0,
                )
                cfg.estimated_params = self.estimate_params(cfg)
                configs.append(cfg)
                continue

            # params ~ depth * width^2, so to get s*params:
            # reduce depth by s^(1/3) and width by s^(1/3)
            # This gives s^(1/3) * (s^(1/3))^2 = s
            cube_root = s ** (1.0 / 3.0)
            new_depth = max(2, round(base_depth * cube_root))
            new_dim_raw = base_dim_rounded * cube_root
            # Round to nearest multiple of head_dim
            new_n_head = max(1, round(new_dim_raw / head_dim))
            new_dim = new_n_head * head_dim

            cfg = ScaleConfig(
                scale_factor=s,
                depth=new_depth,
                n_embd=new_dim,
                n_head=new_n_head,
                n_kv_head=new_n_head,
                estimated_params=0,
            )
            cfg.estimated_params = self.estimate_params(cfg)
            configs.append(cfg)

        return configs

    def estimate_params(self, config: ScaleConfig) -> int:
        """Rough parameter estimate matching the GPT architecture in train.py."""
        d = config.n_embd
        n_layer = config.depth
        # Per layer: attn (4 * d*d) + mlp (4*d * d + d * 4*d) = 4d^2 + 8d^2 = 12d^2
        # Plus embeddings and lm_head
        vocab_size = 32768  # default from train.py
        transformer_params = n_layer * 12 * d * d
        embedding_params = vocab_size * d  # wte
        lm_head_params = vocab_size * d
        # Value embeddings: ~half the layers, each vocab_size * kv_dim
        n_ve_layers = (n_layer + 1) // 2
        kv_dim = config.n_kv_head * (d // config.n_head)
        ve_params = n_ve_layers * vocab_size * kv_dim
        # Scalars
        scalar_params = 2 * n_layer
        return transformer_params + embedding_params + lm_head_params + ve_params + scalar_params

    def validate_config(self, config: ScaleConfig) -> tuple[bool, str]:
        """Check architectural constraints. Returns (valid, reason)."""
        if config.depth < 2:
            return False, f"depth must be >= 2, got {config.depth}"
        if config.n_head < 1:
            return False, f"n_head must be >= 1, got {config.n_head}"
        if config.n_embd <= 0:
            return False, f"n_embd must be > 0, got {config.n_embd}"
        if config.n_embd % config.n_head != 0:
            return False, f"n_embd ({config.n_embd}) not divisible by n_head ({config.n_head})"
        head_dim = config.n_embd // config.n_head
        if head_dim < 1:
            return False, f"head_dim must be >= 1, got {head_dim}"
        if config.n_kv_head < 1:
            return False, f"n_kv_head must be >= 1, got {config.n_kv_head}"
        if config.n_head % config.n_kv_head != 0:
            return False, f"n_head ({config.n_head}) not divisible by n_kv_head ({config.n_kv_head})"
        return True, "ok"
