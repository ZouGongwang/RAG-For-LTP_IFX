from __future__ import annotations

import re
from pathlib import Path

from rag.config import PromptConfig
from rag.llm.base import Message
from rag.models import RetrievedChunk

TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"
_CITE_RE = re.compile(r"\[(\d+(?:\s*[,;]\s*\d+)*)\]")


def cited_indices(answer: str) -> list[int]:
    """Passage numbers cited in the answer, in first-cited order ("[1][3]", "[2, 4]")."""
    seen: list[int] = []
    for group in _CITE_RE.findall(answer):
        for n in re.split(r"\s*[,;]\s*", group):
            i = int(n)
            if i not in seen:
                seen.append(i)
    return seen


def _without_citations(text: str) -> str:
    """Drop "[1]" markers: passage numbers only mean something for the answer they were written for."""
    text = _CITE_RE.sub("", text)
    text = re.sub(r"(?<=\S)[ \t]{2,}", " ", text)
    text = re.sub(r"[ \t]+([.,;:])", r"\1", text)
    return re.sub(r"[ \t]+\n", "\n", text).strip()


class PromptBuilder:
    def __init__(self, cfg: PromptConfig, templates_dir: Path = TEMPLATES_DIR):
        self.cfg = cfg
        self.system = (templates_dir / "system.md").read_text(encoding="utf-8").strip()
        self.template = (templates_dir / cfg.template).read_text(encoding="utf-8")
        self.followup = (templates_dir / "followup.md").read_text(encoding="utf-8").strip()
        self.condense = (templates_dir / "condense.md").read_text(encoding="utf-8")

    def trim_history(self, history: list[Message] | None) -> list[Message]:
        """Keep the most recent turns, starting at a user message, with citations removed and long text cut."""
        if not history or self.cfg.history_turns <= 0:
            return []
        limit = self.cfg.history_chars
        kept: list[Message] = []
        for m in history[-2 * self.cfg.history_turns :]:
            role, text = m.get("role"), (m.get("content") or "").strip()
            if role not in ("user", "assistant") or not text:
                continue
            if role == "assistant":
                text = _without_citations(text)
            if len(text) > limit:
                text = text[: max(limit - 1, 1)].rstrip() + "…"
            kept.append({"role": role, "content": text})
        while kept and kept[0]["role"] != "user":
            kept.pop(0)
        return kept

    def condense_messages(self, question: str, history: list[Message]) -> list[Message]:
        """Prompt that turns a follow-up like "and the second one?" into a standalone search query."""
        transcript = "\n".join(
            f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content']}" for m in history
        )
        return [{"role": "user", "content": self.condense.format(history=transcript, question=question)}]

    def build(
        self, question: str, chunks: list[RetrievedChunk], history: list[Message] | None = None
    ) -> tuple[list[Message], list[RetrievedChunk]]:
        """Return chat messages and the chunks that fit in the context budget (in passage order)."""
        blocks, used, total = [], [], 0
        for chunk in chunks:
            n = len(used) + 1
            url = chunk.metadata.get("url") or ""
            head = f"[{n}] {chunk.label()}" + (f"\nURL: {url}" if url else "")
            block = f"{head}\n{chunk.text}"
            if used and total + len(block) > self.cfg.max_context_chars:
                break
            blocks.append(block)
            used.append(chunk)
            total += len(block)

        context = "\n\n---\n\n".join(blocks) if blocks else "(no relevant passages found)"
        user = self.template.format(context=context, question=question)
        prior = self.trim_history(history)
        system = self.system + ("\n\n" + self.followup if prior else "")
        messages: list[Message] = [{"role": "system", "content": system}, *prior]
        messages.append({"role": "user", "content": user})
        return messages, used
