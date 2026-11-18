"""HTTP API. Start with ``rag serve`` (or ``uvicorn rag.api.server:app``)."""

from __future__ import annotations

import json
import threading
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from rag import __version__
from rag.config import get_settings
from rag.models import RetrievedChunk
from rag.pipeline import IngestPipeline, QueryPipeline
from rag.pipeline.components import Components

app = FastAPI(title="BSP/IFX RAG", version=__version__)
_STATIC = Path(__file__).resolve().parent / "static"
_NO_CACHE = {"Cache-Control": "no-cache"}

_origins = [origin.rstrip("/") for origin in get_settings().api.cors_origins if origin]
if _origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_origins,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )


@app.get("/")
def home() -> FileResponse:
    return FileResponse(_STATIC / "index.html", headers=_NO_CACHE)


@app.get("/widget.js")
def widget_js() -> FileResponse:
    return FileResponse(_STATIC / "widget.js", media_type="application/javascript", headers=_NO_CACHE)


@lru_cache(maxsize=1)
def components() -> Components:
    return Components(get_settings())


class ChatTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(max_length=20000)


class QueryRequest(BaseModel):
    question: str = Field(min_length=1)
    space: str | None = None
    source: str | None = None
    top_k: int | None = Field(default=None, ge=1, le=50)
    # The server keeps no sessions. The page sends the earlier turns of the chat with each question.
    history: list[ChatTurn] = Field(default_factory=list, max_length=50)

    def prior_turns(self) -> list[dict[str, str]]:
        return [turn.model_dump() for turn in self.history]


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
        result = QueryPipeline(components()).ask(
            req.question, filters=filters, top_k=req.top_k, history=req.prior_turns()
        )
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"{type(exc).__name__}: {exc}") from exc
    return QueryResponse(
        answer=result.answer,
        sources=_out(result.sources, result.context_chunks),
        context=_out(result.context_chunks, result.context_chunks),
    )


@app.post("/query/stream")
def query_stream(req: QueryRequest) -> StreamingResponse:
    """Same streamed answer as ``rag ask``, then the cited sources."""
    pipeline = QueryPipeline(components())
    filters = {"space": req.space, "source": req.source}

    def events():
        try:
            stream, used = pipeline.ask_stream(
                req.question, filters=filters, top_k=req.top_k, history=req.prior_turns()
            )
            parts: list[str] = []
            for piece in stream:
                parts.append(piece)
                yield json.dumps({"type": "delta", "text": piece}) + "\n"
            answer = "".join(parts)
            sources = [item.model_dump() for item in _out(pipeline.select_sources(answer, used), used)]
            yield json.dumps({"type": "done", "answer": answer, "sources": sources}) + "\n"
        except Exception as exc:
            yield json.dumps({"type": "error", "detail": f"{type(exc).__name__}: {exc}"}) + "\n"

    return StreamingResponse(events(), media_type="application/x-ndjson")


@app.post("/ingest", status_code=202)
def ingest(req: IngestRequest, background: BackgroundTasks) -> dict[str, Any]:
    if _ingest_status["running"]:
        raise HTTPException(status_code=409, detail="An ingest run is already in progress")
    background.add_task(_run_ingest, req)
    return {"status": "started"}


@app.get("/ingest/status")
def ingest_status() -> dict[str, Any]:
    return _ingest_status
