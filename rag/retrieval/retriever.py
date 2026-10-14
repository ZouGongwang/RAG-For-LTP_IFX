"""Hybrid retrieval: dense vectors + BM25 keywords, fused with reciprocal rank fusion,
optionally reranked by a cross-encoder.

BM25 catches exact identifiers (part numbers, project codes such as VRFC2506) that
embedding models tend to blur.
"""

from __future__ import annotations

import logging
import re
import threading
from pathlib import Path
from typing import Any

from rag.config import RetrievalConfig
from rag.embedding.base import Embedder
from rag.models import RetrievedChunk
from rag.vectorstore.base import VectorStore

log = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[._-][a-z0-9]+)*")


def tokenize(text: str) -> list[str]:
    tokens = _TOKEN_RE.findall(text.lower())
    # Also index the parts of compound identifiers ("vrfc-2506" -> "vrfc", "2506").
    extra = [p for t in tokens if re.search(r"[._-]", t) for p in re.split(r"[._-]", t)]
    return tokens + extra


def reciprocal_rank_fusion(rankings: list[list[RetrievedChunk]], k: int = 60) -> list[RetrievedChunk]:
    scores: dict[str, float] = {}
    by_id: dict[str, RetrievedChunk] = {}
    for ranking in rankings:
        for rank, chunk in enumerate(ranking):
            scores[chunk.chunk_id] = scores.get(chunk.chunk_id, 0.0) + 1.0 / (k + rank + 1)
            by_id.setdefault(chunk.chunk_id, chunk)
    fused = sorted(scores, key=scores.get, reverse=True)
    return [RetrievedChunk(by_id[c].chunk_id, by_id[c].text, by_id[c].metadata, scores[c]) for c in fused]


class _BM25Index:
    def __init__(self, chunks: list[RetrievedChunk]):
        from rank_bm25 import BM25Okapi

        self.chunks = chunks
        self.bm25 = BM25Okapi([tokenize(c.text) for c in chunks]) if chunks else None

    def search(self, query: str, k: int, where: dict[str, Any] | None) -> list[RetrievedChunk]:
        if self.bm25 is None:
            return []
        scores = self.bm25.get_scores(tokenize(query))
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        out = []
        for i in order:
            if scores[i] <= 0:
                break
            c = self.chunks[i]
            if where and any(c.metadata.get(key) != val for key, val in where.items() if val not in (None, "")):
                continue
            out.append(RetrievedChunk(c.chunk_id, c.text, c.metadata, float(scores[i])))
            if len(out) >= k:
                break
        return out


class HybridRetriever:
    def __init__(
        self,
        cfg: RetrievalConfig,
        embedder: Embedder,
        store: VectorStore,
        device: str = "cpu",
        cache_dir: Path | None = None,
    ):
        self.cfg = cfg
        self.embedder = embedder
        self.store = store
        self._bm25: _BM25Index | None = None
        self._bm25_count = -1
        self._lock = threading.Lock()
        self._reranker = None
        if cfg.rerank.enabled:
            from sentence_transformers import CrossEncoder

            log.info("Loading reranker %s", cfg.rerank.model)
            self._reranker = CrossEncoder(
                cfg.rerank.model, device=device, cache_folder=str(cache_dir) if cache_dir else None
            )

    def _bm25_index(self) -> _BM25Index:
        with self._lock:
            n = self.store.count()
            if self._bm25 is None or n != self._bm25_count:
                log.info("Building BM25 index over %d chunks", n)
                self._bm25 = _BM25Index(list(self.store.iter_all()))
                self._bm25_count = n
            return self._bm25

    def invalidate(self) -> None:
        with self._lock:
            self._bm25 = None

    def retrieve(self, query: str, top_k: int | None = None, filters: dict[str, Any] | None = None) -> list[RetrievedChunk]:
        top_k = top_k or self.cfg.top_k
        dense = self.store.query(self.embedder.embed_query(query), self.cfg.dense_k, where=filters)
        rankings = [dense]
        if self.cfg.bm25_k > 0:
            rankings.append(self._bm25_index().search(query, self.cfg.bm25_k, filters))
        fused = reciprocal_rank_fusion(rankings, k=self.cfg.rrf_k)

        if self._reranker is not None and fused:
            candidates = fused[: self.cfg.rerank.candidates]
            scores = self._reranker.predict([(query, c.text) for c in candidates])
            for c, s in zip(candidates, scores):
                c.score = float(s)
            fused = sorted(candidates, key=lambda c: c.score, reverse=True)
        return fused[:top_k]
