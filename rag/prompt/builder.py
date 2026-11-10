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


class PromptBuilder:
    def __init__(self, cfg: PromptConfig, templates_dir: Path = TEMPLATES_DIR):
        self.cfg = cfg
        self.system = (templates_dir / "system.md").read_text(encoding="utf-8").strip()
        self.template = (templates_dir / cfg.template).read_text(encoding="utf-8")

    def build(self, question: str, chunks: list[RetrievedChunk]) -> tuple[list[Message], list[RetrievedChunk]]:
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
        return [{"role": "system", "content": self.system}, {"role": "user", "content": user}], used
