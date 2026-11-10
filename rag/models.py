"""Data structures passed between pipeline stages."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Literal

SegmentKind = Literal["text", "table", "image_caption", "page_caption"]


@dataclass
class Segment:
    """A piece of parsed Markdown with its location in the source document."""

    text: str
    page: int | None = None
    kind: SegmentKind = "text"


@dataclass
class ParsedDocument:
    segments: list[Segment]


@dataclass
class SourceItem:
    """One document announced by a source.

    ``load`` is deferred so the pipeline can skip unchanged documents (same ``version``)
    without downloading or parsing them.
    """

    doc_id: str
    version: str
    title: str
    source: str
    load: Callable[[], ParsedDocument]
    url: str = ""
    breadcrumb: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    text: str
    metadata: dict[str, Any]


@dataclass
class RetrievedChunk:
    chunk_id: str
    text: str
    metadata: dict[str, Any]
    score: float = 0.0

    def label(self) -> str:
        title = self.metadata.get("title") or self.metadata.get("doc_id", "Unknown")
        section = self.metadata.get("section") or ""
        page = self.metadata.get("page") or 0
        unit = "Slide" if self.metadata.get("file_type") == "pptx" else "Page"
        parts = [title]
        if section and section != title:
            parts.append(f"> {section}")
        if page:
            parts.append(f"({unit} {page})")
        return " ".join(parts)


@dataclass
class Answer:
    question: str
    answer: str
    sources: list[RetrievedChunk]
    context_chunks: list[RetrievedChunk]
