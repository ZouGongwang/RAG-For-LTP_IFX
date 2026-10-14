from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator

import chromadb
from chromadb.config import Settings as ChromaSettings

from rag.models import Chunk, RetrievedChunk

_BATCH = 2000


def build_where(filters: dict[str, Any] | None) -> dict[str, Any] | None:
    """{"space": "BSP", "source": "confluence"} -> Chroma where clause."""
    if not filters:
        return None
    clauses = [{k: v} for k, v in filters.items() if v not in (None, "")]
    if not clauses:
        return None
    return clauses[0] if len(clauses) == 1 else {"$and": clauses}


class ChromaStore:
    def __init__(self, path: Path, collection: str, embedding_model: str):
        path.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(
            path=str(path), settings=ChromaSettings(anonymized_telemetry=False)
        )
        self._name = collection
        self._embedding_model = embedding_model
        self._col = self._open()

    def _open(self):
        col = self._client.get_or_create_collection(
            self._name,
            metadata={"hnsw:space": "cosine", "embedding_model": self._embedding_model},
        )
        stored = (col.metadata or {}).get("embedding_model")
        if stored and stored != self._embedding_model:
            raise RuntimeError(
                f"Collection '{self._name}' was built with embedding model '{stored}' but config uses "
                f"'{self._embedding_model}'. Run `rag ingest --rebuild` to re-index."
            )
        return col

    def upsert(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        for i in range(0, len(chunks), _BATCH):
            batch = chunks[i : i + _BATCH]
            self._col.upsert(
                ids=[c.chunk_id for c in batch],
                documents=[c.text for c in batch],
                metadatas=[c.metadata for c in batch],
                embeddings=embeddings[i : i + _BATCH],
            )

    def delete_doc(self, doc_id: str) -> None:
        self._col.delete(where={"doc_id": doc_id})

    def query(self, embedding: list[float], k: int, where: dict[str, Any] | None = None) -> list[RetrievedChunk]:
        n = self.count()
        if n == 0:
            return []
        res = self._col.query(
            query_embeddings=[embedding],
            n_results=min(k, n),
            where=build_where(where),
            include=["documents", "metadatas", "distances"],
        )
        return [
            RetrievedChunk(chunk_id=cid, text=doc, metadata=meta or {}, score=1.0 - dist)
            for cid, doc, meta, dist in zip(
                res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0]
            )
        ]

    def iter_all(self, where: dict[str, Any] | None = None) -> Iterator[RetrievedChunk]:
        offset = 0
        while True:
            res = self._col.get(
                where=build_where(where), limit=_BATCH, offset=offset, include=["documents", "metadatas"]
            )
            if not res["ids"]:
                return
            for cid, doc, meta in zip(res["ids"], res["documents"], res["metadatas"]):
                yield RetrievedChunk(chunk_id=cid, text=doc, metadata=meta or {})
            offset += len(res["ids"])

    def count(self) -> int:
        return self._col.count()

    def reset(self) -> None:
        try:
            self._client.delete_collection(self._name)
        except Exception:
            pass
        self._col = self._open()
