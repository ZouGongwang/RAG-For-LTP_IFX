"""retrieve -> build prompt -> LLM -> answer with cited sources."""

from __future__ import annotations

from typing import Any, Iterator

from rag.llm.base import ThinkFilter, strip_think
from rag.models import Answer, RetrievedChunk
from rag.pipeline.components import Components
from rag.prompt import cited_indices


class QueryPipeline:
    def __init__(self, components: Components):
        self.c = components
        self.s = components.settings

    def _prepare(self, question: str, filters: dict[str, Any] | None, top_k: int | None):
        chunks = self.c.retriever.retrieve(question, top_k=top_k, filters=filters)
        messages, used = self.c.prompt_builder.build(question, chunks)
        return messages, used

    def select_sources(self, answer: str, used: list[RetrievedChunk]) -> list[RetrievedChunk]:
        cited = [used[i - 1] for i in cited_indices(answer) if 1 <= i <= len(used)]
        candidates = cited or used[: self.s.prompt.display_sources]
        out, seen = [], set()
        for ch in candidates:
            key = (ch.metadata.get("doc_id"), ch.metadata.get("page"))
            if key not in seen:
                seen.add(key)
                out.append(ch)
        return out

    def ask(self, question: str, filters: dict[str, Any] | None = None, top_k: int | None = None) -> Answer:
        messages, used = self._prepare(question, filters, top_k)
        answer = strip_think(self.c.llm.generate(messages))
        return Answer(question, answer, self.select_sources(answer, used), used)

    def ask_stream(
        self, question: str, filters: dict[str, Any] | None = None, top_k: int | None = None
    ) -> tuple[Iterator[str], list[RetrievedChunk]]:
        messages, used = self._prepare(question, filters, top_k)

        def filtered() -> Iterator[str]:
            f = ThinkFilter()
            for piece in self.c.llm.stream(messages):
                if out := f.feed(piece):
                    yield out
            if tail := f.flush():
                yield tail

        return filtered(), used
