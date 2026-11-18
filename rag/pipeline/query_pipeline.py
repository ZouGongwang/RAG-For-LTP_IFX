"""retrieve -> build prompt -> LLM -> answer with cited sources."""

from __future__ import annotations

import logging
from typing import Any, Iterator

from rag.llm.base import Message, ThinkFilter, strip_think
from rag.models import Answer, RetrievedChunk
from rag.pipeline.components import Components
from rag.prompt import cited_indices

log = logging.getLogger(__name__)


class QueryPipeline:
    def __init__(self, components: Components):
        self.c = components
        self.s = components.settings

    def standalone_question(self, question: str, history: list[Message]) -> str:
        """Rewrite a follow-up so that retrieval does not depend on the earlier turns.

        "What about the second one?" finds nothing on its own, so the model first turns it into a
        complete question using the conversation. On any failure the original question is used.
        """
        if not history:
            return question
        try:
            raw = strip_think(self.c.llm.generate(self.c.prompt_builder.condense_messages(question, history)))
        except Exception as exc:  # retrieval with the raw question is still better than failing
            log.warning("Could not rewrite follow-up question (%s: %s); using it as typed", type(exc).__name__, exc)
            return question
        lines = [line.strip().strip("\"'") for line in raw.splitlines() if line.strip()]
        rewritten = lines[0][:400] if lines else ""
        return rewritten or question

    def _prepare(
        self,
        question: str,
        filters: dict[str, Any] | None,
        top_k: int | None,
        history: list[Message] | None = None,
    ):
        history = self.c.prompt_builder.trim_history(history)
        search_query = self.standalone_question(question, history)
        chunks = self.c.retriever.retrieve(search_query, top_k=top_k, filters=filters)
        messages, used = self.c.prompt_builder.build(question, chunks, history)
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

    def ask(
        self,
        question: str,
        filters: dict[str, Any] | None = None,
        top_k: int | None = None,
        history: list[Message] | None = None,
    ) -> Answer:
        messages, used = self._prepare(question, filters, top_k, history)
        answer = strip_think(self.c.llm.generate(messages))
        return Answer(question, answer, self.select_sources(answer, used), used)

    def ask_stream(
        self,
        question: str,
        filters: dict[str, Any] | None = None,
        top_k: int | None = None,
        history: list[Message] | None = None,
    ) -> tuple[Iterator[str], list[RetrievedChunk]]:
        messages, used = self._prepare(question, filters, top_k, history)

        def filtered() -> Iterator[str]:
            f = ThinkFilter()
            for piece in self.c.llm.stream(messages):
                if out := f.feed(piece):
                    yield out
            if tail := f.flush():
                yield tail

        return filtered(), used
