"""
Attention architecture analyzer: maps the full attention dataflow from train.py.
"""

import re
from typing import Dict, Any

import sys, os


class AttentionArchitectureAnalyzer:
    """Map the full attention dataflow from train.py source code.

    Extracts: backend used, RoPE type, sliding window pattern, GQA ratio,
    value embeddings, softcap, and other attention-related architecture details.
    """

    def _detect_attention_backend(self, source: str) -> dict:
        """Detect which attention backend is used."""
        backend = {"name": "unknown", "version": "", "details": ""}

        if "flash_attn_func" in source:
            backend["name"] = "flash_attention_3"
            if "flash-attention-3" in source:
                backend["version"] = "varunneal/flash-attention-3"
            if "kernels-community" in source:
                backend["details"] = "Falls back to kernels-community/flash-attn3 on non-Hopper"
            # Check for Hopper detection
            if "get_device_capability" in source:
                backend["details"] += "; Hopper (sm_90) detection enabled"

        return backend

    def _detect_rope_config(self, source: str) -> dict:
        """Detect RoPE (Rotary Position Embedding) configuration."""
        config = {
            "type": "standard",
            "base": 10000,
            "implementation": "manual",
            "applied_to": "q_and_k",
        }

        # Check for base frequency
        base_match = re.search(r'base\s*=\s*(\d+)', source)
        if base_match:
            config["base"] = int(base_match.group(1))

        # Check implementation style
        if "apply_rotary_emb" in source:
            config["implementation"] = "custom_apply_rotary_emb"
        if "x1 * cos + x2 * sin" in source:
            config["type"] = "standard_split_half"  # split-half rotation
        if "torch.cat([y1, y2]" in source:
            config["concat_style"] = "cat_last_dim"

        # Check what it's applied to
        if "apply_rotary_emb(q" in source and "apply_rotary_emb(k" in source:
            config["applied_to"] = "q_and_k"

        # QK-norm after RoPE
        if "norm(q)" in source and "norm(k)" in source:
            config["post_rope_norm"] = "rms_norm_on_q_and_k"

        return config

    def _detect_sliding_window(self, source: str) -> dict:
        """Detect sliding window attention configuration."""
        config = {
            "enabled": False,
            "pattern": "",
            "short_window_ratio": 0.5,
            "last_layer_override": "full",
        }

        if "window_pattern" in source:
            config["enabled"] = True
            # Extract pattern
            pattern_match = re.search(r'window_pattern\s*[=:]\s*["\'](\w+)["\']', source)
            if pattern_match:
                config["pattern"] = pattern_match.group(1)

        if "SSSL" in source:
            config["pattern"] = "SSSL"
            config["description"] = (
                "3 short-window layers followed by 1 long-window layer, repeating. "
                "Last layer always uses full context."
            )

        # Window size computation
        if "short_window = long_window // 2" in source:
            config["short_window_ratio"] = 0.5

        if "window_sizes[-1]" in source:
            config["last_layer_override"] = "full_context"

        return config

    def _detect_gqa(self, source: str) -> dict:
        """Detect Grouped Query Attention configuration."""
        config = {
            "enabled": False,
            "n_head": 0,
            "n_kv_head": 0,
            "ratio": 1,
        }

        n_head_match = re.search(r'n_head\s*[=:]\s*(\d+)', source)
        n_kv_head_match = re.search(r'n_kv_head\s*[=:]\s*(\d+)', source)

        if n_head_match:
            config["n_head"] = int(n_head_match.group(1))
        if n_kv_head_match:
            config["n_kv_head"] = int(n_kv_head_match.group(1))

        if config["n_kv_head"] > 0 and config["n_head"] > 0:
            config["ratio"] = config["n_head"] // config["n_kv_head"]
            config["enabled"] = config["ratio"] > 1

        # Also check the config class default
        if "n_kv_head: int = 6" in source and "n_head: int = 6" in source:
            config["note"] = "Default config uses MHA (ratio=1), but supports GQA"

        return config

    def _detect_value_embeddings(self, source: str) -> dict:
        """Detect value embedding (ResFormer) configuration."""
        config = {
            "enabled": False,
            "pattern": "",
            "gate_type": "",
            "gate_channels": 0,
        }

        if "value_embeds" in source or "ve_gate" in source:
            config["enabled"] = True

        if "has_ve" in source:
            config["pattern"] = "alternating_with_last_included"
            if "layer_idx % 2 ==" in source:
                config["pattern_detail"] = "Every other layer, last layer always included"

        if "ve_gate" in source:
            config["gate_type"] = "input_dependent_per_head"
            gate_match = re.search(r've_gate_channels\s*=\s*(\d+)', source)
            if gate_match:
                config["gate_channels"] = int(gate_match.group(1))

        if "2 * torch.sigmoid" in source:
            config["gate_activation"] = "2*sigmoid (range [0,2], neutral at 1.0)"
            config["gate_init"] = "zeros (sigmoid(0)=0.5, *2=1.0 = neutral mixing)"

        return config

    def _detect_softcap(self, source: str) -> dict:
        """Detect logit softcapping."""
        config = {"enabled": False, "value": 0.0}

        softcap_match = re.search(r'softcap\s*=\s*(\d+)', source)
        if softcap_match:
            config["enabled"] = True
            config["value"] = float(softcap_match.group(1))

        if "torch.tanh(logits / softcap)" in source:
            config["formula"] = "softcap * tanh(logits / softcap)"
            config["applied_after"] = "lm_head projection, cast to float32"

        return config

    def analyze(self, train_source: str) -> Dict[str, Any]:
        """Map the full attention dataflow from train.py.

        Args:
            train_source: Source code of train.py.

        Returns:
            Dict with comprehensive attention architecture analysis.
        """
        backend = self._detect_attention_backend(train_source)
        rope = self._detect_rope_config(train_source)
        sliding_window = self._detect_sliding_window(train_source)
        gqa = self._detect_gqa(train_source)
        value_embeddings = self._detect_value_embeddings(train_source)
        softcap = self._detect_softcap(train_source)

        # Attention dataflow summary
        dataflow = [
            "1. Input x -> Linear projections: c_q, c_k, c_v",
            "2. Reshape to (B, T, n_heads, head_dim)",
            "3. Value residual: v = v + gate * value_embedding (alternating layers)",
            "4. RoPE: apply_rotary_emb to q and k (split-half rotation)",
            "5. QK-norm: RMSNorm on q and k after RoPE",
            "6. Flash Attention 3: flash_attn_func(q, k, v, causal=True, window_size=...)",
            "7. Reshape + output projection: c_proj",
        ]

        # Identify fusion opportunities around attention
        fusion_targets = [
            {
                "name": "pre_attention_fusion",
                "ops": ["value_embedding_lookup", "ve_gate_sigmoid", "ve_mix", "rope", "qk_norm"],
                "description": "Fuse value embedding mixing + RoPE + QK-norm before attention",
                "complexity": "high",
            },
            {
                "name": "rope_qknorm_fusion",
                "ops": ["rope_apply", "rms_norm_q", "rms_norm_k"],
                "description": "Fuse RoPE application with QK-normalization",
                "complexity": "medium",
            },
            {
                "name": "post_attention_norm_residual",
                "ops": ["reshape", "c_proj", "residual_add", "rms_norm"],
                "description": "Fuse post-attention projection with residual and norm",
                "complexity": "medium",
            },
        ]

        return {
            "backend": backend,
            "rope": rope,
            "sliding_window": sliding_window,
            "gqa": gqa,
            "value_embeddings": value_embeddings,
            "softcap": softcap,
            "dataflow": dataflow,
            "fusion_targets": fusion_targets,
            "model_specifics": {
                "activation": "relu().square() (ReGLU-like without gating)",
                "norm": "RMSNorm (F.rms_norm)",
                "residual_scaling": "per-layer resid_lambdas and x0_lambdas",
                "x0_connection": "x0 (post-embedding norm) mixed into each layer",
            },
        }
