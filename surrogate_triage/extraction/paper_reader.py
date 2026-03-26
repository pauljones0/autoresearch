"""
Phase 1.2 -- PaperReader: rule-based extraction of structured TechniqueDescription
from paper abstracts and full text.
"""

import re
import uuid

from surrogate_triage.schemas import TechniqueDescription

# Taxonomy aligned with FailureExtractor._CATEGORY_KEYWORDS
_CATEGORY_KEYWORDS: dict[str, list[str]] = {
    "architecture": [
        "attention", "transformer", "mlp", "feedforward", "residual", "skip",
        "block", "layer", "head", "embedding", "encoder", "decoder",
        "convolution", "pooling", "normalization", "layer norm", "rmsnorm",
        "groupnorm", "multi-head", "cross-attention", "self-attention",
    ],
    "optimizer": [
        "optimizer", "adam", "adamw", "sgd", "momentum", "weight decay",
        "learning rate", "gradient clip", "gradient accumulation", "muon",
        "lion", "shampoo", "adafactor",
    ],
    "hyperparameter": [
        "batch size", "sequence length", "context length", "max steps",
        "tokens", "accumulation", "micro batch", "model size", "width",
        "depth", "hidden dimension", "vocabulary",
    ],
    "activation": [
        "relu", "gelu", "silu", "swish", "tanh", "sigmoid", "activation",
        "softmax", "softplus", "mish", "squared relu", "glu",
    ],
    "initialization": [
        "initialization", "init", "xavier", "kaiming", "normal init",
        "uniform init", "orthogonal", "truncated normal", "zero init",
    ],
    "regularization": [
        "dropout", "drop path", "label smoothing", "weight norm",
        "spectral norm", "regularization", "l2", "l1", "stochastic depth",
    ],
    "scheduling": [
        "scheduler", "cosine", "linear decay", "warmup", "cooldown",
        "cycle", "one cycle", "learning rate schedule", "annealing",
    ],
}

# Patterns for extracting technique claims from abstracts
_CLAIM_PATTERNS = [
    re.compile(r"we\s+(?:propose|introduce|present|develop)\s+(.+?)(?:\.|,\s+which)", re.I),
    re.compile(r"(?:our|this)\s+(?:method|approach|technique|model|framework)\s*,?\s*(?:called|named|dubbed)?\s*([^,.]+)", re.I),
    re.compile(r"(?:a|an)\s+(?:novel|new|simple|efficient)\s+(.+?)(?:\s+that\s|\s+which\s|\.|,)", re.I),
]

# Patterns for extracting reported improvements
_IMPROVEMENT_PATTERNS = [
    re.compile(r"(\d+(?:\.\d+)?)\s*%\s*(?:improvement|better|reduction|gain|increase)\s*(?:over|compared|than|relative)?\s*([\w\s]*)", re.I),
    re.compile(r"(?:achieves?|obtains?|reaches?)\s+(\d+(?:\.\d+)?)\s*%?\s*(?:improvement|better|lower|higher)", re.I),
    re.compile(r"(?:reduces?|lowers?|decreases?)\s+(?:the\s+)?(?:loss|perplexity|error)\s+(?:by\s+)?(\d+(?:\.\d+)?)\s*%?", re.I),
    re.compile(r"(?:outperforms?|surpasses?|exceeds?)\s+([\w\s]+)\s+by\s+(\d+(?:\.\d+)?)\s*%?", re.I),
    re.compile(r"(\d+(?:\.\d+)?)\s*(?:bpb|perplexity|ppl)\s*(?:vs\.?|compared to|versus)\s*([\w\s.]+)", re.I),
]

# Patterns for extracting applicability conditions
_APPLICABILITY_PATTERNS = [
    re.compile(r"for\s+(transformer|language|vision|image|speech|audio|nlp|gpt|bert|llm)\s+models?", re.I),
    re.compile(r"(?:at|for)\s+(large|small|medium)\s+scale", re.I),
    re.compile(r"with\s+(pre-norm|post-norm|pre-ln|post-ln|layer norm)", re.I),
    re.compile(r"(?:on|for|with)\s+(pre-?training|fine-?tuning|transfer learning)", re.I),
    re.compile(r"(?:requires?|needs?|assumes?)\s+(.+?)(?:\.|,|$)", re.I),
]

_VALID_CATEGORIES = set(_CATEGORY_KEYWORDS.keys()) | {"other"}


def _classify_category(text: str) -> str:
    """Classify technique category from text using keyword matching."""
    if not text:
        return "other"
    text_lower = text.lower()
    scores: dict[str, int] = {}
    for category, keywords in _CATEGORY_KEYWORDS.items():
        scores[category] = sum(1 for kw in keywords if kw in text_lower)
    best = max(scores, key=scores.get)
    if scores[best] == 0:
        return "other"
    return best


def _extract_technique_name(text: str) -> str:
    """Try to extract a technique name from text using claim patterns."""
    for pattern in _CLAIM_PATTERNS:
        m = pattern.search(text)
        if m:
            name = m.group(1).strip()
            # Trim to a reasonable length
            if len(name) > 120:
                name = name[:120].rsplit(" ", 1)[0]
            return name
    return ""


def _extract_improvements(text: str) -> tuple[str, str]:
    """Extract reported improvement text and baseline from text."""
    improvements = []
    baseline = ""
    for pattern in _IMPROVEMENT_PATTERNS:
        for m in pattern.finditer(text):
            improvements.append(m.group(0).strip())
            # Try to pull baseline from the match
            groups = m.groups()
            if len(groups) >= 2 and groups[-1]:
                candidate = groups[-1].strip()
                if candidate and len(candidate) > 2:
                    baseline = candidate
    return "; ".join(improvements), baseline


def _extract_applicability(text: str) -> list[str]:
    """Extract applicability conditions from text."""
    conditions = []
    for pattern in _APPLICABILITY_PATTERNS:
        for m in pattern.finditer(text):
            cond = m.group(0).strip()
            if cond and cond not in conditions:
                conditions.append(cond)
    return conditions


def _generate_technique_id(paper_id: str, index: int) -> str:
    """Generate a unique technique ID."""
    short_uuid = uuid.uuid4().hex[:8]
    safe_paper = re.sub(r"[^a-zA-Z0-9]", "_", paper_id) if paper_id else "unknown"
    return f"tech_{safe_paper}_{index}_{short_uuid}"


class PaperReader:
    """Rule-based extraction of structured TechniqueDescription from paper text."""

    def extract_from_abstract(self, paper_metadata: dict) -> list[TechniqueDescription]:
        """Extract technique descriptions from paper metadata (must include 'abstract').

        Parameters
        ----------
        paper_metadata : dict
            Should contain at least 'arxiv_id' (or 'paper_id') and 'abstract'.
            May also contain 'title'.

        Returns
        -------
        list[TechniqueDescription]
        """
        paper_id = paper_metadata.get("arxiv_id") or paper_metadata.get("paper_id", "")
        abstract = paper_metadata.get("abstract", "")
        title = paper_metadata.get("title", "")

        if not abstract:
            return []

        combined = f"{title}. {abstract}" if title else abstract
        return self._extract_techniques(paper_id, combined)

    def extract_from_text(self, paper_id: str, text: str) -> list[TechniqueDescription]:
        """Extract technique descriptions from arbitrary paper text.

        Parameters
        ----------
        paper_id : str
            Identifier for the source paper.
        text : str
            Full text or section text to extract from.

        Returns
        -------
        list[TechniqueDescription]
        """
        if not text or not text.strip():
            return []
        return self._extract_techniques(paper_id, text)

    def _extract_techniques(self, paper_id: str, text: str) -> list[TechniqueDescription]:
        """Core extraction logic."""
        results: list[TechniqueDescription] = []

        # Try to extract a primary technique name
        name = _extract_technique_name(text)
        category = _classify_category(text)
        improvement_text, baseline = _extract_improvements(text)
        conditions = _extract_applicability(text)

        # Build description from the first 2 sentences if we have a name
        description = self._build_description(text)

        if not name:
            # Fall back: use the first meaningful sentence as the name
            sentences = re.split(r'(?<=[.!?])\s+', text.strip())
            for s in sentences:
                s = s.strip()
                if len(s) > 20:
                    name = s[:100].rsplit(" ", 1)[0] if len(s) > 100 else s
                    break

        if not name:
            return []

        # Confidence heuristic: higher if we found more structured info
        confidence = 0.3
        if improvement_text:
            confidence += 0.2
        if baseline:
            confidence += 0.1
        if conditions:
            confidence += 0.1
        if category != "other":
            confidence += 0.1
        if len(description) > 50:
            confidence += 0.1
        confidence = min(confidence, 1.0)

        technique = TechniqueDescription(
            technique_id=_generate_technique_id(paper_id, 0),
            paper_id=paper_id,
            name=name,
            modification_category=category,
            description=description,
            reported_improvement=improvement_text,
            baseline=baseline,
            applicability_conditions=conditions,
            extraction_confidence=round(confidence, 2),
        )
        results.append(technique)

        # If the text contains multiple distinct technique claims, try to extract more
        additional = self._extract_secondary_techniques(paper_id, text, name)
        results.extend(additional)

        return results

    def _build_description(self, text: str) -> str:
        """Extract first 2-3 sentences as description."""
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        desc_parts = []
        char_count = 0
        for s in sentences:
            s = s.strip()
            if not s:
                continue
            desc_parts.append(s)
            char_count += len(s)
            if len(desc_parts) >= 3 or char_count > 400:
                break
        return " ".join(desc_parts)

    def _extract_secondary_techniques(self, paper_id: str, text: str,
                                       primary_name: str) -> list[TechniqueDescription]:
        """Look for additional distinct techniques mentioned in the text."""
        results = []
        # Find additional "we also propose/introduce" patterns
        secondary_pattern = re.compile(
            r"(?:we\s+)?(?:also|additionally|further)\s+(?:propose|introduce|present)\s+(.+?)(?:\.|,)",
            re.I,
        )
        idx = 1
        for m in secondary_pattern.finditer(text):
            name = m.group(1).strip()
            if not name or len(name) < 5:
                continue
            if len(name) > 120:
                name = name[:120].rsplit(" ", 1)[0]
            # Skip if it looks like the primary technique
            if primary_name and name.lower() in primary_name.lower():
                continue

            category = _classify_category(name)
            technique = TechniqueDescription(
                technique_id=_generate_technique_id(paper_id, idx),
                paper_id=paper_id,
                name=name,
                modification_category=category,
                description=name,
                extraction_confidence=0.2,
            )
            results.append(technique)
            idx += 1

        return results
