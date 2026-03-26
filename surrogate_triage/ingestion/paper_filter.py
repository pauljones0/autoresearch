"""
Paper relevance filtering for the Surrogate Triage Pipeline.
Two-layer scoring: keyword matching + diagnostics-informed boosting.
"""

import logging
import re
from typing import Optional

from surrogate_triage.schemas import PaperMetadata

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Keyword categories with weights
# ---------------------------------------------------------------------------

ARCHITECTURE_KEYWORDS = {
    "transformer": 0.3, "attention mechanism": 0.4, "self-attention": 0.4,
    "multi-head attention": 0.5, "feedforward": 0.2, "residual connection": 0.4,
    "skip connection": 0.3, "layer normalization": 0.3, "positional encoding": 0.3,
    "positional embedding": 0.3, "mixture of experts": 0.4, "moe": 0.3,
    "sparse attention": 0.4, "linear attention": 0.4, "gated linear unit": 0.3,
    "glu": 0.2, "swiglu": 0.3, "geglu": 0.3, "rotary embedding": 0.4,
    "rope": 0.3, "grouped query attention": 0.4, "gqa": 0.3,
    "multi-query attention": 0.4, "sliding window": 0.3,
    "value embedding": 0.4, "value residual": 0.4,
}

OPTIMIZATION_KEYWORDS = {
    "learning rate": 0.3, "learning rate schedule": 0.4, "warmup": 0.3,
    "cosine schedule": 0.3, "cosine annealing": 0.3, "adam": 0.2,
    "adamw": 0.3, "muon": 0.4, "sgd": 0.2, "momentum": 0.2,
    "weight decay": 0.3, "gradient clipping": 0.3, "gradient accumulation": 0.3,
    "loss function": 0.3, "cross entropy": 0.2, "label smoothing": 0.3,
    "logit softcapping": 0.4, "batch size": 0.2, "optimizer": 0.2,
    "convergence": 0.2, "training stability": 0.4, "loss landscape": 0.3,
    "sharpness aware": 0.4, "sam optimizer": 0.4,
}

EFFICIENCY_KEYWORDS = {
    "training efficiency": 0.5, "compute efficient": 0.4, "parameter efficient": 0.4,
    "flops": 0.3, "throughput": 0.3, "tokens per second": 0.4,
    "mixed precision": 0.3, "bfloat16": 0.3, "fp16": 0.2,
    "gradient checkpointing": 0.3, "memory efficient": 0.3,
    "flash attention": 0.4, "kernel fusion": 0.3, "compilation": 0.2,
    "torch.compile": 0.4, "distillation": 0.3, "pruning": 0.3,
    "quantization": 0.3, "scaling law": 0.4, "chinchilla": 0.3,
}

REGULARIZATION_KEYWORDS = {
    "regularization": 0.3, "dropout": 0.3, "weight initialization": 0.4,
    "initialization": 0.3, "xavier": 0.3, "kaiming": 0.3,
    "spectral norm": 0.3, "batch normalization": 0.2, "rms norm": 0.4,
    "pre-norm": 0.3, "post-norm": 0.3, "normalization": 0.2,
    "data augmentation": 0.2, "overfit": 0.2, "generalization": 0.2,
    "dead neuron": 0.4, "activation function": 0.3, "relu": 0.2,
    "gelu": 0.2, "squared relu": 0.4,
}

CATEGORY_WEIGHTS = {
    "architecture": 1.0,
    "optimization": 1.0,
    "efficiency": 0.8,
    "regularization": 0.7,
}

KEYWORD_SETS = {
    "architecture": ARCHITECTURE_KEYWORDS,
    "optimization": OPTIMIZATION_KEYWORDS,
    "efficiency": EFFICIENCY_KEYWORDS,
    "regularization": REGULARIZATION_KEYWORDS,
}

# ---------------------------------------------------------------------------
# Diagnostics-informed boosting rules
# ---------------------------------------------------------------------------

# Maps diagnostic conditions to search terms and boost amounts.
# Each rule: (condition_check_fn_name, boost_terms, boost_weight)
DIAGNOSTICS_BOOST_RULES = [
    {
        "name": "attention_entropy_collapse",
        "description": "Attention entropy is low / collapsed",
        "boost_terms": [
            "attention diversity", "attention entropy", "head diversity",
            "multi-head redundancy", "attention collapse", "sparse attention",
        ],
        "boost_weight": 0.3,
    },
    {
        "name": "gradient_vanishing",
        "description": "Gradient norms are vanishing in early layers",
        "boost_terms": [
            "residual connection", "skip connection", "gradient flow",
            "vanishing gradient", "deep residual", "initialization",
        ],
        "boost_weight": 0.3,
    },
    {
        "name": "gradient_exploding",
        "description": "Gradient norms are exploding",
        "boost_terms": [
            "gradient clipping", "gradient normalization", "training stability",
            "gradient explosion", "norm scaling", "pre-norm",
        ],
        "boost_weight": 0.3,
    },
    {
        "name": "dead_neurons",
        "description": "High fraction of dead neurons in activations",
        "boost_terms": [
            "dead neuron", "activation function", "relu alternative",
            "leaky relu", "gelu", "swish", "neuron death",
        ],
        "boost_weight": 0.25,
    },
    {
        "name": "layer_collapse",
        "description": "High CKA similarity between adjacent layers (representation collapse)",
        "boost_terms": [
            "representation collapse", "layer diversity", "deep network",
            "feature diversity", "CKA similarity", "representation learning",
        ],
        "boost_weight": 0.25,
    },
    {
        "name": "rare_token_loss",
        "description": "Loss on rare tokens is disproportionately high",
        "boost_terms": [
            "rare token", "token frequency", "vocabulary", "loss weighting",
            "focal loss", "class imbalance", "tail distribution",
        ],
        "boost_weight": 0.2,
    },
]


def _check_attention_entropy_collapse(diagnostics) -> bool:
    """Check if attention entropy is collapsed."""
    if not diagnostics.attention_stats:
        return False
    stats = diagnostics.attention_stats
    if isinstance(stats[0], dict):
        entropies = [s.get("entropy", 1.0) for s in stats]
        collapse_scores = [s.get("collapse_score", 0.0) for s in stats]
    else:
        entropies = [s.entropy for s in stats]
        collapse_scores = [s.collapse_score for s in stats]
    avg_entropy = sum(entropies) / len(entropies) if entropies else 1.0
    avg_collapse = sum(collapse_scores) / len(collapse_scores) if collapse_scores else 0.0
    return avg_entropy < 1.0 or avg_collapse > 0.5


def _check_gradient_vanishing(diagnostics) -> bool:
    """Check if gradients are vanishing in early layers."""
    if not diagnostics.gradient_stats:
        return False
    stats = diagnostics.gradient_stats
    if isinstance(stats[0], dict):
        norms = [s.get("norm", 1.0) for s in stats]
    else:
        norms = [s.norm for s in stats]
    if len(norms) < 2:
        return False
    early_norms = norms[:len(norms) // 2]
    return any(n < 1e-6 for n in early_norms)


def _check_gradient_exploding(diagnostics) -> bool:
    """Check if gradients are exploding."""
    if not diagnostics.gradient_stats:
        return False
    stats = diagnostics.gradient_stats
    if isinstance(stats[0], dict):
        norms = [s.get("norm", 1.0) for s in stats]
    else:
        norms = [s.norm for s in stats]
    return any(n > 100.0 for n in norms)


def _check_dead_neurons(diagnostics) -> bool:
    """Check if there's a high fraction of dead neurons."""
    if not diagnostics.activation_stats:
        return False
    stats = diagnostics.activation_stats
    if isinstance(stats[0], dict):
        fractions = [s.get("dead_neuron_fraction", 0.0) for s in stats]
    else:
        fractions = [s.dead_neuron_fraction for s in stats]
    avg_dead = sum(fractions) / len(fractions) if fractions else 0.0
    return avg_dead > 0.1


def _check_layer_collapse(diagnostics) -> bool:
    """Check if adjacent layers have high CKA similarity."""
    if not diagnostics.layer_similarity_matrix:
        return False
    entries = diagnostics.layer_similarity_matrix
    adjacent_scores = []
    for e in entries:
        if isinstance(e, dict):
            i, j, score = e.get("layer_i", 0), e.get("layer_j", 0), e.get("cka_score", 0.0)
        else:
            i, j, score = e.layer_i, e.layer_j, e.cka_score
        if abs(i - j) == 1:
            adjacent_scores.append(score)
    if not adjacent_scores:
        return False
    return sum(adjacent_scores) / len(adjacent_scores) > 0.85


def _check_rare_token_loss(diagnostics) -> bool:
    """Check if rare token loss is disproportionately high."""
    if not diagnostics.loss_decomposition:
        return False
    buckets = diagnostics.loss_decomposition
    rare_loss = None
    common_loss = None
    for b in buckets:
        name = b.get("bucket_name", "") if isinstance(b, dict) else b.bucket_name
        loss = b.get("mean_loss", 0.0) if isinstance(b, dict) else b.mean_loss
        if name == "rare":
            rare_loss = loss
        elif name == "top_1k":
            common_loss = loss
    if rare_loss is not None and common_loss is not None and common_loss > 0:
        return rare_loss / common_loss > 2.0
    return False


_CONDITION_CHECKS = {
    "attention_entropy_collapse": _check_attention_entropy_collapse,
    "gradient_vanishing": _check_gradient_vanishing,
    "gradient_exploding": _check_gradient_exploding,
    "dead_neurons": _check_dead_neurons,
    "layer_collapse": _check_layer_collapse,
    "rare_token_loss": _check_rare_token_loss,
}


class PaperFilter:
    """Two-layer paper relevance filter: keyword + diagnostics-informed."""

    def __init__(self, threshold: float = 0.3):
        self.threshold = threshold
        # Pre-compile keyword patterns for faster matching
        self._compiled_patterns = {}
        for category, keywords in KEYWORD_SETS.items():
            patterns = {}
            for kw, weight in keywords.items():
                patterns[re.compile(r"\b" + re.escape(kw) + r"\b", re.IGNORECASE)] = weight
            self._compiled_patterns[category] = patterns

    def _keyword_score(self, text: str) -> float:
        """Score text against all keyword categories."""
        if not text:
            return 0.0
        total = 0.0
        text_lower = text.lower()
        for category, patterns in self._compiled_patterns.items():
            cat_weight = CATEGORY_WEIGHTS[category]
            cat_score = 0.0
            for pattern, kw_weight in patterns.items():
                if pattern.search(text_lower):
                    cat_score += kw_weight
            total += cat_score * cat_weight
        return total

    def _diagnostics_boost(self, text: str, diagnostics) -> float:
        """Compute diagnostics-informed boost for a paper."""
        if diagnostics is None:
            return 0.0
        text_lower = text.lower()
        boost = 0.0
        for rule in DIAGNOSTICS_BOOST_RULES:
            check_fn = _CONDITION_CHECKS.get(rule["name"])
            if check_fn is None:
                continue
            try:
                if not check_fn(diagnostics):
                    continue
            except Exception:
                continue
            # Check if any boost terms appear in the text
            for term in rule["boost_terms"]:
                if term.lower() in text_lower:
                    boost += rule["boost_weight"]
                    break  # one match per rule is enough

        return boost

    def score(self, paper: PaperMetadata, diagnostics=None) -> float:
        """Score a paper's relevance.

        Args:
            paper: Paper metadata to score.
            diagnostics: Optional DiagnosticsReport for diagnostics-informed boosting.

        Returns:
            Combined relevance score (keyword_score + diagnostics_boost).
        """
        text = f"{paper.title} {paper.abstract}"
        kw_score = self._keyword_score(text)
        diag_boost = self._diagnostics_boost(text, diagnostics)

        paper.keyword_score = kw_score
        paper.diagnostics_boost = diag_boost
        paper.relevance_score = kw_score + diag_boost

        return paper.relevance_score

    def filter_papers(self, papers: list, diagnostics=None, threshold: float = None):
        """Filter papers by relevance score.

        Args:
            papers: List of PaperMetadata to filter.
            diagnostics: Optional DiagnosticsReport for boosting.
            threshold: Score threshold (defaults to self.threshold).

        Returns:
            Tuple of (relevant_papers, filtered_papers).
        """
        if threshold is None:
            threshold = self.threshold

        relevant = []
        filtered = []

        for paper in papers:
            score = self.score(paper, diagnostics)
            if score >= threshold:
                paper.filtered_out = False
                paper.filter_reason = ""
                relevant.append(paper)
            else:
                paper.filtered_out = True
                paper.filter_reason = f"score {score:.3f} below threshold {threshold}"
                filtered.append(paper)
                logger.debug(
                    "Filtered out: '%s' (score=%.3f, kw=%.3f, diag=%.3f)",
                    paper.title[:60], score, paper.keyword_score, paper.diagnostics_boost,
                )

        logger.info(
            "Filtered %d papers: %d relevant, %d below threshold %.2f",
            len(papers), len(relevant), len(filtered), threshold,
        )
        return relevant, filtered
