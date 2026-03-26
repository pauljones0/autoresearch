"""
Phase 2 — DiffEmbedder: embed code diffs into fixed-dimensional vectors
using character/word n-gram hashing (hash trick).
"""

import hashlib
import math
import re
import tokenize
import io


# Patterns for preprocessing
_COMMENT_RE = re.compile(r'#[^\n]*')
_WHITESPACE_RE = re.compile(r'\s+')


def _tokenize_python(text: str) -> list[str]:
    """Tokenize Python source, falling back to simple word splitting."""
    tokens = []
    try:
        reader = io.StringIO(text)
        for tok in tokenize.generate_tokens(reader.readline):
            if tok.type in (tokenize.COMMENT, tokenize.NL, tokenize.NEWLINE,
                            tokenize.INDENT, tokenize.DEDENT, tokenize.ENCODING):
                continue
            if tok.string.strip():
                tokens.append(tok.string)
    except tokenize.TokenError:
        # Fall back to simple splitting for partial/invalid Python
        tokens = re.findall(r'[a-zA-Z_]\w*|[^\s]', text)
    return tokens


def _preprocess(diff_text: str) -> str:
    """Normalize whitespace, strip comments, clean diff metadata."""
    lines = []
    for line in diff_text.splitlines():
        # Skip diff metadata lines
        if line.startswith('---') or line.startswith('+++') or line.startswith('@@'):
            continue
        # Strip leading +/- diff markers but keep the content
        if line.startswith('+') or line.startswith('-'):
            line = line[1:]
        lines.append(line)

    text = '\n'.join(lines)
    text = _COMMENT_RE.sub('', text)
    text = _WHITESPACE_RE.sub(' ', text).strip()
    return text


def _hash_to_index(s: str, dim: int) -> int:
    """Deterministic hash of string to an index in [0, dim)."""
    h = hashlib.md5(s.encode('utf-8')).hexdigest()
    return int(h, 16) % dim


def _hash_sign(s: str) -> float:
    """Deterministic sign (+1 or -1) for the hash trick."""
    h = hashlib.sha1(s.encode('utf-8')).hexdigest()
    return 1.0 if int(h, 16) % 2 == 0 else -1.0


def _normalize(vec: list[float]) -> list[float]:
    """Normalize vector to unit length. Returns zero vector if norm is 0."""
    norm = math.sqrt(sum(x * x for x in vec))
    if norm < 1e-12:
        return vec
    return [x / norm for x in vec]


class DiffEmbedder:
    """Embed code diffs into fixed-dimensional vectors using n-gram hashing."""

    def __init__(self, ngram_range: tuple[int, int] = (1, 3)):
        self.ngram_range = ngram_range

    def embed(self, diff_text: str, dim: int = 256) -> list[float]:
        """Embed a single diff into a fixed-dimensional vector.

        Args:
            diff_text: Unified diff text.
            dim: Output vector dimension.

        Returns:
            Unit-normalized float vector of length dim.
        """
        if not diff_text or not diff_text.strip():
            return [0.0] * dim

        text = _preprocess(diff_text)
        tokens = _tokenize_python(text)

        if not tokens:
            return [0.0] * dim

        vec = [0.0] * dim

        # Generate n-grams and hash them into the vector
        for n in range(self.ngram_range[0], self.ngram_range[1] + 1):
            for i in range(len(tokens) - n + 1):
                ngram = ' '.join(tokens[i:i + n])
                idx = _hash_to_index(ngram, dim)
                sign = _hash_sign(ngram)
                vec[idx] += sign

        # Also add character-level trigrams for subword information
        char_text = ' '.join(tokens)
        for i in range(len(char_text) - 2):
            trigram = char_text[i:i + 3]
            idx = _hash_to_index('char:' + trigram, dim)
            sign = _hash_sign('char:' + trigram)
            vec[idx] += sign * 0.5  # lower weight for char n-grams

        return _normalize(vec)

    def embed_batch(self, diffs: list[str], dim: int = 256) -> list[list[float]]:
        """Embed multiple diffs.

        Args:
            diffs: List of unified diff texts.
            dim: Output vector dimension.

        Returns:
            List of unit-normalized float vectors.
        """
        return [self.embed(d, dim=dim) for d in diffs]
