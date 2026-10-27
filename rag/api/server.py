"""HTTP API. Start with ``rag serve`` (or ``uvicorn rag.api.server:app``)."""

from __future__ import annotations

import threading
from functools import lru_cache
from pathlib import Path
from typing import Any

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from rag import __version__
from rag.config import get_settings
from rag.models import RetrievedChunk
from rag.pipeline import IngestPipeline, QueryPipeline
from rag.pipeline.components import Components

app = FastAPI(title="BSP/IFX RAG", version=__version__)
_WIDGET = Path(__file__).resolve().parent / "static" / "index.html"


@app.get("/")
def home() -> FileResponse:
    return FileResponse(_WIDGET)


@lru_cache(maxsize=1)
def components() -> Components:
    return Components(get_settings())


class QueryRequest(BaseModel):
    question: str = Field(min_length=1)
    space: str | None = None
    source: str | None = None
    top_k: int | None = Field(default=None, ge=1, le=50)


class SourceOut(BaseModel):
    n: int
    label: str
    title: str
    url: str
    page: int
    section: str
    content_type: str


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceOut]
    context: list[SourceOut]


class IngestRequest(BaseModel):
    sources: list[str] | None = None
    force: bool = False
    rebuild: bool = False


def _out(chunks: list[RetrievedChunk], used: list[RetrievedChunk]) -> list[SourceOut]:
    index = {c.chunk_id: i + 1 for i, c in enumerate(used)}
    return [
        SourceOut(
            n=index.get(c.chunk_id, 0),
            label=c.label(),
            title=str(c.metadata.get("title", "")),
            url=str(c.metadata.get("url", "")),
            page=int(c.metadata.get("page") or 0),
            section=str(c.metadata.get("section", "")),
            content_type=str(c.metadata.get("content_type", "")),
        )
        for c in chunks
    ]


_ingest_lock = threading.Lock()
_ingest_status: dict[str, Any] = {"running": False, "last": None}


def _run_ingest(req: IngestRequest) -> None:
    with _ingest_lock:
        _ingest_status["running"] = True
        try:
            stats = IngestPipeline(components()).run(req.sources, force=req.force, rebuild=req.rebuild)
            components().retriever.invalidate()
            _ingest_status["last"] = {"summary": stats.summary(), "errors": stats.errors}
        except Exception as exc:
            _ingest_status["last"] = {"summary": "failed", "errors": [str(exc)]}
        finally:
            _ingest_status["running"] = False


@app.get("/health")
def health() -> dict[str, Any]:
    c = components()
    return {
        "status": "ok",
        "version": __version__,
        "chunks": c.store.count(),
        "llm": c.settings.llm.provider,
        "embedding_model": c.settings.embedding.model,
    }


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest) -> QueryResponse:
    filters = {"space": req.space, "source": req.source}
    try:
        result = QueryPipeline(components()).ask(req.question, filters=filters, top_k=req.top_k)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"{type(exc).__name__}: {exc}") from exc
    return QueryResponse(
        answer=result.answer,
        sources=_out(result.sources, result.context_chunks),
        context=_out(result.context_chunks, result.context_chunks),
    )


@app.post("/ingest", status_code=202)
def ingest(req: IngestRequest, background: BackgroundTasks) -> dict[str, Any]:
    if _ingest_status["running"]:
        raise HTTPException(status_code=409, detail="An ingest run is already in progress")
    background.add_task(_run_ingest, req)
    return {"status": "started"}


@app.get("/ingest/status")
def ingest_status() -> dict[str, Any]:
    return _ingest_status
