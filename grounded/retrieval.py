"""Hybrid retrieval: BM25 for exact terms, a vector index for paraphrase, fused by rank.

Neither signal is enough alone. BM25 finds "LRU" when the question says "LRU" but
misses "evict the least recently touched item"; embeddings catch the paraphrase
but blur rare identifiers. Reciprocal rank fusion (RRF) combines the two rank
lists without having to calibrate their very different score scales.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Protocol, Sequence

import numpy as np

from .corpus import Chunk

TOKEN = re.compile(r"[a-z0-9]+")
STOPWORDS = frozenset(
    "a an and are as at be by can do does for from how in is it its of on or that the this to "
    "was what when where which who why with you your".split()
)


def tokenize(text: str) -> list[str]:
    return [t for t in TOKEN.findall(text.lower()) if t not in STOPWORDS]


@dataclass
class Hit:
    chunk: Chunk
    score: float
    ranks: dict[str, int] = field(default_factory=dict)


class BM25Index:
    def __init__(self, chunks: Sequence[Chunk], k1: float = 1.5, b: float = 0.75):
        self.chunks = list(chunks)
        self.k1, self.b = k1, b
        self.docs = [Counter(tokenize(f"{c.heading} {c.text}")) for c in self.chunks]
        self.lengths = [sum(d.values()) for d in self.docs]
        self.avg_len = sum(self.lengths) / max(len(self.lengths), 1)
        df = Counter(term for d in self.docs for term in d)
        n = len(self.docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def search(self, query: str, k: int = 5) -> list[Hit]:
        terms = tokenize(query)
        scored = []
        for i, doc in enumerate(self.docs):
            s = 0.0
            for t in terms:
                tf = doc.get(t, 0)
                if tf:
                    norm = tf + self.k1 * (1 - self.b + self.b * self.lengths[i] / self.avg_len)
                    s += self.idf[t] * tf * (self.k1 + 1) / norm
            if s > 0:
                scored.append(Hit(self.chunks[i], s))
        scored.sort(key=lambda h: h.score, reverse=True)
        return scored[:k]


class Embedder(Protocol):
    def embed(self, texts: Sequence[str]) -> np.ndarray:
        """Return one L2-normalised row per text."""


class HashingEmbedder:
    """Dependency-free, deterministic embedder (signed feature hashing of words and word pairs).

    It is not semantic: it rewards shared vocabulary and phrasing, which is enough
    to exercise the vector path offline and in tests. For real paraphrase recall,
    pass any object with the same ``embed`` method backed by a sentence-embedding
    model.
    """

    def __init__(self, dim: int = 512):
        self.dim = dim

    def _features(self, text: str) -> list[str]:
        words = tokenize(text)
        return words + [f"{a}_{b}" for a, b in zip(words, words[1:])]

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for feat in self._features(text):
                h = int.from_bytes(hashlib.blake2b(feat.encode(), digest_size=8).digest(), "little")
                out[row, h % self.dim] += 1.0 if (h >> 63) & 1 else -1.0
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.where(norms == 0, 1, norms)


class FastEmbedEmbedder:
    """Local sentence embeddings via fastembed (ONNX, no API key). pip install "grounded-agent-harness[embeddings]".

    The default model, BAAI/bge-small-en-v1.5 (384 dimensions), is downloaded once and cached.
    """

    def __init__(self, model: str = "BAAI/bge-small-en-v1.5"):
        from fastembed import TextEmbedding

        self.model = TextEmbedding(model)

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        out = np.array(list(self.model.embed(list(texts))), dtype=np.float32)
        return out / np.linalg.norm(out, axis=1, keepdims=True)


def make_embedder(kind: str = "auto") -> Embedder:
    """'fastembed', 'hashing', or 'auto' (fastembed when installed, else hashing)."""
    if kind in ("auto", "fastembed"):
        try:
            return FastEmbedEmbedder()
        except ImportError:
            if kind == "fastembed":
                raise
    return HashingEmbedder()


class VectorIndex:
    """Exact inner-product search over normalised embeddings. Uses FAISS when installed."""

    def __init__(self, chunks: Sequence[Chunk], embedder: Embedder):
        self.chunks = list(chunks)
        self.embedder = embedder
        self.matrix = embedder.embed([f"{c.heading} {c.text}" for c in self.chunks]).astype(np.float32)
        try:
            import faiss  # optional: pip install "grounded-agent-harness[faiss]"

            self._faiss = faiss.IndexFlatIP(self.matrix.shape[1])
            self._faiss.add(self.matrix)
        except ImportError:
            self._faiss = None

    def search(self, query: str, k: int = 5) -> list[Hit]:
        q = self.embedder.embed([query]).astype(np.float32)
        k = min(k, len(self.chunks))
        if self._faiss is not None:
            scores, idx = self._faiss.search(q, k)
            pairs = zip(idx[0].tolist(), scores[0].tolist())
        else:
            sims = self.matrix @ q[0]
            top = np.argsort(-sims)[:k]
            pairs = ((int(i), float(sims[i])) for i in top)
        return [Hit(self.chunks[i], s) for i, s in pairs if s > 0]


class HybridRetriever:
    def __init__(self, chunks: Sequence[Chunk], embedder: Embedder | None = None, rrf_k: int = 60):
        self.chunks = {c.id: c for c in chunks}
        self.bm25 = BM25Index(chunks)
        self.vector = VectorIndex(chunks, embedder or HashingEmbedder())
        self.rrf_k = rrf_k

    def search(self, query: str, k: int = 5, pool: int = 20) -> list[Hit]:
        fused: dict[str, Hit] = {}
        for name, hits in (("bm25", self.bm25.search(query, pool)), ("vector", self.vector.search(query, pool))):
            for rank, hit in enumerate(hits, start=1):
                entry = fused.setdefault(hit.chunk.id, Hit(hit.chunk, 0.0))
                entry.score += 1.0 / (self.rrf_k + rank)
                entry.ranks[name] = rank
        return sorted(fused.values(), key=lambda h: h.score, reverse=True)[:k]

    def get(self, chunk_id: str) -> Chunk | None:
        return self.chunks.get(chunk_id)
