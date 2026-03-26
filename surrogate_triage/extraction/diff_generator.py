"""
Phase 1.3 -- DiffGenerator: generate concrete code diffs for a technique
against the current train.py using template-based generation.
"""

import difflib
import re
import uuid

from surrogate_triage.schemas import TechniqueDescription, SyntheticDiff

# ---------------------------------------------------------------------------
# Template definitions per category, each with conservative/moderate/aggressive
# variants.  Templates are callables: f(base_source, technique) -> new_source.
# ---------------------------------------------------------------------------


def _replace_in_source(source: str, old: str, new: str) -> str | None:
    """Replace *old* with *new* in *source*. Return None if old not found."""
    if old not in source:
        return None
    return source.replace(old, new, 1)


# ---- Architecture templates ------------------------------------------------

def _arch_conservative(src: str, tech: TechniqueDescription) -> str | None:
    """Add a skip-connection scaling factor to Block.forward."""
    old = "x = x + self.attn(norm(x), ve, cos_sin, window_size)"
    new = "x = x + 0.9 * self.attn(norm(x), ve, cos_sin, window_size)"
    return _replace_in_source(src, old, new)


def _arch_moderate(src: str, tech: TechniqueDescription) -> str | None:
    """Double the MLP intermediate dimension."""
    old = "self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)"
    new = "self.c_fc = nn.Linear(config.n_embd, 6 * config.n_embd, bias=False)"
    result = _replace_in_source(src, old, new)
    if result is None:
        return None
    old2 = "self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)"
    new2 = "self.c_proj = nn.Linear(6 * config.n_embd, config.n_embd, bias=False)"
    return _replace_in_source(result, old2, new2)


def _arch_aggressive(src: str, tech: TechniqueDescription) -> str | None:
    """Add pre-norm to attention output projection."""
    old = "y = self.c_proj(y)"
    new = "y = self.c_proj(norm(y))"
    return _replace_in_source(src, old, new)


def _arch_variant4(src: str, tech: TechniqueDescription) -> str | None:
    """Scale the MLP residual connection."""
    old = "x = x + self.mlp(norm(x))"
    new = "x = x + 0.5 * self.mlp(norm(x))"
    return _replace_in_source(src, old, new)


# ---- Optimizer templates ---------------------------------------------------

def _opt_conservative(src: str, tech: TechniqueDescription) -> str | None:
    """Increase Adam beta1."""
    old = "ADAM_BETAS = (0.8, 0.95)"
    new = "ADAM_BETAS = (0.9, 0.95)"
    return _replace_in_source(src, old, new)


def _opt_moderate(src: str, tech: TechniqueDescription) -> str | None:
    """Adjust matrix learning rate."""
    old = "MATRIX_LR = 0.04"
    new = "MATRIX_LR = 0.02"
    return _replace_in_source(src, old, new)


def _opt_aggressive(src: str, tech: TechniqueDescription) -> str | None:
    """Higher weight decay."""
    old = "WEIGHT_DECAY = 0.2"
    new = "WEIGHT_DECAY = 0.4"
    return _replace_in_source(src, old, new)


def _opt_variant4(src: str, tech: TechniqueDescription) -> str | None:
    """Higher embedding learning rate."""
    old = "EMBEDDING_LR = 0.6"
    new = "EMBEDDING_LR = 0.8"
    return _replace_in_source(src, old, new)


def _opt_variant5(src: str, tech: TechniqueDescription) -> str | None:
    """Lower unembedding learning rate."""
    old = "UNEMBEDDING_LR = 0.004"
    new = "UNEMBEDDING_LR = 0.002"
    return _replace_in_source(src, old, new)


# ---- Hyperparameter templates ----------------------------------------------

def _hp_conservative(src: str, tech: TechniqueDescription) -> str | None:
    """Increase model depth."""
    old = "DEPTH = 8"
    new = "DEPTH = 10"
    return _replace_in_source(src, old, new)


def _hp_moderate(src: str, tech: TechniqueDescription) -> str | None:
    """Adjust aspect ratio."""
    old = "ASPECT_RATIO = 64"
    new = "ASPECT_RATIO = 72"
    return _replace_in_source(src, old, new)


def _hp_aggressive(src: str, tech: TechniqueDescription) -> str | None:
    """Double the total batch size."""
    old = "TOTAL_BATCH_SIZE = 2**19"
    new = "TOTAL_BATCH_SIZE = 2**20"
    return _replace_in_source(src, old, new)


def _hp_variant4(src: str, tech: TechniqueDescription) -> str | None:
    """Change window pattern."""
    old = 'WINDOW_PATTERN = "SSSL"'
    new = 'WINDOW_PATTERN = "SSLL"'
    return _replace_in_source(src, old, new)


# ---- Activation templates --------------------------------------------------

def _act_conservative(src: str, tech: TechniqueDescription) -> str | None:
    """Replace squared ReLU with GELU."""
    old = "x = F.relu(x).square()"
    new = "x = F.gelu(x)"
    return _replace_in_source(src, old, new)


def _act_moderate(src: str, tech: TechniqueDescription) -> str | None:
    """Replace squared ReLU with SiLU."""
    old = "x = F.relu(x).square()"
    new = "x = F.silu(x)"
    return _replace_in_source(src, old, new)


def _act_aggressive(src: str, tech: TechniqueDescription) -> str | None:
    """Replace squared ReLU with GELU squared."""
    old = "x = F.relu(x).square()"
    new = "x = F.gelu(x).square()"
    return _replace_in_source(src, old, new)


# ---- Initialization templates ----------------------------------------------

def _init_conservative(src: str, tech: TechniqueDescription) -> str | None:
    """Smaller lm_head init std."""
    old = 'torch.nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.001)'
    new = 'torch.nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.0005)'
    return _replace_in_source(src, old, new)


def _init_moderate(src: str, tech: TechniqueDescription) -> str | None:
    """Change embedding init std."""
    old = 'torch.nn.init.normal_(self.transformer.wte.weight, mean=0.0, std=1.0)'
    new = 'torch.nn.init.normal_(self.transformer.wte.weight, mean=0.0, std=0.5)'
    return _replace_in_source(src, old, new)


def _init_aggressive(src: str, tech: TechniqueDescription) -> str | None:
    """Change x0_lambdas init."""
    old = "self.x0_lambdas.fill_(0.1)"
    new = "self.x0_lambdas.fill_(0.05)"
    return _replace_in_source(src, old, new)


def _init_variant4(src: str, tech: TechniqueDescription) -> str | None:
    """Change resid_lambdas init."""
    old = "self.resid_lambdas.fill_(1.0)"
    new = "self.resid_lambdas.fill_(0.9)"
    return _replace_in_source(src, old, new)


# ---- Regularization templates -----------------------------------------------

def _reg_conservative(src: str, tech: TechniqueDescription) -> str | None:
    """Reduce softcap value."""
    old = "softcap = 15"
    new = "softcap = 12"
    return _replace_in_source(src, old, new)


def _reg_moderate(src: str, tech: TechniqueDescription) -> str | None:
    """Increase softcap value."""
    old = "softcap = 15"
    new = "softcap = 20"
    return _replace_in_source(src, old, new)


def _reg_aggressive(src: str, tech: TechniqueDescription) -> str | None:
    """Add gradient clipping in training loop."""
    old = "    optimizer.step()"
    new = "    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)\n    optimizer.step()"
    return _replace_in_source(src, old, new)


# ---- Scheduling templates ---------------------------------------------------

def _sched_conservative(src: str, tech: TechniqueDescription) -> str | None:
    """Add warmup."""
    old = "WARMUP_RATIO = 0.0"
    new = "WARMUP_RATIO = 0.05"
    return _replace_in_source(src, old, new)


def _sched_moderate(src: str, tech: TechniqueDescription) -> str | None:
    """Longer warmdown."""
    old = "WARMDOWN_RATIO = 0.5"
    new = "WARMDOWN_RATIO = 0.6"
    return _replace_in_source(src, old, new)


def _sched_aggressive(src: str, tech: TechniqueDescription) -> str | None:
    """Non-zero final LR."""
    old = "FINAL_LR_FRAC = 0.0"
    new = "FINAL_LR_FRAC = 0.05"
    return _replace_in_source(src, old, new)


def _sched_variant4(src: str, tech: TechniqueDescription) -> str | None:
    """Warmup + shorter warmdown."""
    old = "WARMUP_RATIO = 0.0"
    new = "WARMUP_RATIO = 0.02"
    result = _replace_in_source(src, old, new)
    if result is None:
        return None
    old2 = "WARMDOWN_RATIO = 0.5"
    new2 = "WARMDOWN_RATIO = 0.4"
    return _replace_in_source(result, old2, new2)


# ---------------------------------------------------------------------------
# Template registry
# ---------------------------------------------------------------------------

_TEMPLATES: dict[str, list[tuple[str, callable]]] = {
    "architecture": [
        ("conservative", _arch_conservative),
        ("moderate", _arch_moderate),
        ("aggressive", _arch_aggressive),
        ("variant4", _arch_variant4),
    ],
    "optimizer": [
        ("conservative", _opt_conservative),
        ("moderate", _opt_moderate),
        ("aggressive", _opt_aggressive),
        ("variant4", _opt_variant4),
        ("variant5", _opt_variant5),
    ],
    "hyperparameter": [
        ("conservative", _hp_conservative),
        ("moderate", _hp_moderate),
        ("aggressive", _hp_aggressive),
        ("variant4", _hp_variant4),
    ],
    "activation": [
        ("conservative", _act_conservative),
        ("moderate", _act_moderate),
        ("aggressive", _act_aggressive),
    ],
    "initialization": [
        ("conservative", _init_conservative),
        ("moderate", _init_moderate),
        ("aggressive", _init_aggressive),
        ("variant4", _init_variant4),
    ],
    "regularization": [
        ("conservative", _reg_conservative),
        ("moderate", _reg_moderate),
        ("aggressive", _reg_aggressive),
    ],
    "scheduling": [
        ("conservative", _sched_conservative),
        ("moderate", _sched_moderate),
        ("aggressive", _sched_aggressive),
        ("variant4", _sched_variant4),
    ],
}

# For "other" category, try all categories and pick whichever produces diffs
_ALL_TEMPLATES = [tpl for templates in _TEMPLATES.values() for tpl in templates]


class DiffGenerator:
    """Generate concrete code diffs for a technique against current train.py."""

    def generate_diffs(
        self,
        technique: TechniqueDescription,
        base_source: str,
    ) -> list[SyntheticDiff]:
        """Generate synthetic diffs for a technique.

        Parameters
        ----------
        technique : TechniqueDescription
            The extracted technique description.
        base_source : str
            The current content of train.py.

        Returns
        -------
        list[SyntheticDiff]
            One diff per successfully applied template variant.
        """
        if not base_source or not base_source.strip():
            return []

        category = technique.modification_category or "other"
        templates = _TEMPLATES.get(category, _ALL_TEMPLATES)

        # For "other", try a curated subset from each category
        if category == "other":
            templates = _ALL_TEMPLATES

        results: list[SyntheticDiff] = []
        base_lines = base_source.splitlines(keepends=True)

        for variant_idx, (variant_name, template_fn) in enumerate(templates):
            try:
                new_source = template_fn(base_source, technique)
            except Exception:
                continue

            if new_source is None or new_source == base_source:
                continue

            new_lines = new_source.splitlines(keepends=True)
            diff_lines = list(difflib.unified_diff(
                base_lines, new_lines,
                fromfile="train.py", tofile="train.py",
            ))
            if not diff_lines:
                continue

            diff_text = "".join(diff_lines)
            diff_id = f"diff_{technique.technique_id}_{variant_idx}_{uuid.uuid4().hex[:6]}"

            results.append(SyntheticDiff(
                diff_id=diff_id,
                technique_id=technique.technique_id,
                paper_id=technique.paper_id,
                variant_index=variant_idx,
                diff_text=diff_text,
                modification_category=category,
            ))

        return results
