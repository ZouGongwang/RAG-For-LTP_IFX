"""Heading-aware Markdown chunker.

* Starts a new chunk at headings (unless the current chunk is still tiny).
* Packs paragraphs up to ``max_tokens`` and carries ~``overlap_ratio`` of trailing text
  into the next chunk.
* Keeps tables, code blocks and image descriptions whole; oversized tables are split by
  rows with the header row repeated.
* Never mixes pages/slides in one chunk, so citations stay precise.
* Prefixes each chunk with a breadcrumb (space > parents > title > section) so the chunk is
  self-describing for both the embedder and the LLM.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable, Literal

from rag.config import ChunkingConfig
from rag.models import Chunk, ParsedDocument, SourceItem

BlockKind = Literal["heading", "text", "table", "code", "image_caption", "page_caption"]

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")
_SENTENCE_RE = re.compile(r"(?<=[.!?;])\s+")
_CELL_SPLIT_RE = re.compile(r"(?<!\\)\|")
# Private-use code points: icon-font glyphs (Wingdings etc.) extracted from PDFs/slides.
_PRIVATE_USE_RE = re.compile("[\ue000-\uf8ff]")


def clean_text(text: str) -> str:
    return _PRIVATE_USE_RE.sub("", text)


def compact_table(table: str) -> str:
    """Strip cell padding and long separator dashes; they waste embedding tokens."""
    out = []
    for line in table.splitlines():
        cells = [c.strip() for c in _CELL_SPLIT_RE.split(line.strip())[1:-1]]
        if cells and all(re.fullmatch(r":?-+:?", c) for c in cells):
            cells = ["---"] * len(cells)
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


@dataclass
class Block:
    kind: BlockKind
    text: str
    level: int = 0


def split_blocks(markdown: str) -> list[Block]:
    blocks: list[Block] = []
    lines = markdown.splitlines()
    i = 0
    para: list[str] = []

    def flush_para() -> None:
        if para:
            text = "\n".join(para).strip()
            if text:
                blocks.append(Block("text", text))
            para.clear()

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        if stripped.startswith("```"):
            flush_para()
            code = [line]
            i += 1
            while i < len(lines):
                code.append(lines[i])
                if lines[i].strip().startswith("```"):
                    break
                i += 1
            blocks.append(Block("code", "\n".join(code)))
        elif stripped.startswith("|"):
            flush_para()
            table = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                table.append(lines[i])
                i += 1
            blocks.append(Block("table", compact_table("\n".join(table))))
            continue
        elif m := _HEADING_RE.match(stripped):
            flush_para()
            blocks.append(Block("heading", stripped, level=len(m.group(1))))
        elif not stripped:
            flush_para()
        else:
            para.append(line)
        i += 1
    flush_para()
    return blocks


def _scalar(v: Any) -> str | int | float | bool:
    if v is None:
        return ""
    if isinstance(v, (str, int, float, bool)):
        return v
    if isinstance(v, (list, tuple)):
        return " > ".join(str(x) for x in v)
    return str(v)


class MarkdownChunker:
    def __init__(self, cfg: ChunkingConfig, count_tokens: Callable[[str], int]):
        self.cfg = cfg
        self.count = count_tokens
        self.overlap_tokens = int(cfg.max_tokens * cfg.overlap_ratio)

    # ---------- splitting oversized blocks ----------
    def _split_large(self, block: Block) -> list[Block]:
        out: list[Block] = []
        for piece in self._split_structured(block):
            out.extend(self._split_words(piece))
        return out

    def _split_words(self, block: Block) -> list[Block]:
        """Last resort for a single row/sentence that is still over budget."""
        if self.count(block.text) <= self.cfg.max_tokens:
            return [block]
        words, out, cur = block.text.split(" "), [], []
        for w in words:
            if cur and self.count(" ".join(cur + [w])) > self.cfg.max_tokens:
                out.append(Block(block.kind, " ".join(cur)))
                cur = []
            cur.append(w)
        if cur:
            out.append(Block(block.kind, " ".join(cur)))
        return out

    def _split_structured(self, block: Block) -> list[Block]:
        if self.count(block.text) <= self.cfg.max_tokens:
            return [block]
        if block.kind == "table":
            rows = block.text.splitlines()
            header = rows[:2] if len(rows) > 1 and set(rows[1].replace("|", "").strip()) <= set("-: ") else rows[:1]
            body = rows[len(header):]
            return [Block("table", "\n".join(header + part)) for part in self._pack_lines(body, header)]
        if block.kind == "code":
            return [Block("code", "\n".join(part)) for part in self._pack_lines(block.text.splitlines(), [])]
        sentences = _SENTENCE_RE.split(block.text)
        out, cur = [], ""
        for s in sentences:
            cand = f"{cur} {s}".strip()
            if cur and self.count(cand) > self.cfg.max_tokens:
                out.append(Block(block.kind, cur))
                cur = s
            else:
                cur = cand
        if cur:
            out.append(Block(block.kind, cur))
        return out

    def _pack_lines(self, lines: list[str], header: list[str]) -> list[list[str]]:
        budget = self.cfg.max_tokens - self.count("\n".join(header))
        parts, cur, cur_tok = [], [], 0
        for line in lines:
            t = self.count(line) + 1
            if cur and cur_tok + t > budget:
                parts.append(cur)
                cur, cur_tok = [], 0
            cur.append(line)
            cur_tok += t
        if cur:
            parts.append(cur)
        return parts

    # ---------- main ----------
    def chunk(self, item: SourceItem, parsed: ParsedDocument) -> list[Chunk]:
        base_meta = {k: _scalar(v) for k, v in item.metadata.items()}
        base_meta.update(
            doc_id=item.doc_id,
            source=item.source,
            title=item.title,
            url=item.url,
            breadcrumb=" > ".join(item.breadcrumb),
            version=item.version,
        )

        chunks: list[Chunk] = []
        headings: list[tuple[int, str]] = []
        cur: list[Block] = []
        cur_tok = 0
        cur_section = ""
        cur_page: int | None = None

        def section_path() -> str:
            return " > ".join(h for _, h in headings)

        def emit(carry_overlap: bool) -> None:
            nonlocal cur, cur_tok, cur_section
            body_blocks = [b for b in cur if b.kind != "heading"]
            if body_blocks:
                body = "\n\n".join(b.text for b in cur)
                kinds = {b.kind for b in body_blocks}
                content_type = next(
                    (k for k in ("table", "image_caption", "page_caption", "code") if k in kinds), "text"
                )
                section = cur_section
                if section == item.title or section.startswith(item.title + " > "):
                    section = section[len(item.title) :].lstrip(" >")
                crumb = [*item.breadcrumb, item.title] + ([section] if section else [])
                header = " > ".join(c for c in crumb if c)
                meta = dict(base_meta, section=section, page=cur_page or 0, content_type=content_type)
                chunks.append(
                    Chunk(
                        chunk_id=f"{item.doc_id}::{len(chunks)}",
                        doc_id=item.doc_id,
                        text=f"{header}\n\n{body}" if header else body,
                        metadata=meta,
                    )
                )
            carried: list[Block] = []
            if carry_overlap and self.overlap_tokens:
                tok = 0
                for b in reversed(body_blocks):
                    if b.kind != "text":
                        break
                    t = self.count(b.text)
                    if tok + t > self.overlap_tokens:
                        break
                    carried.insert(0, b)
                    tok += t
            cur = carried
            cur_tok = sum(self.count(b.text) for b in carried)
            cur_section = section_path()

        for seg in parsed.segments:
            if seg.page != cur_page:
                emit(carry_overlap=False)
                cur, cur_tok = [], 0
                cur_page = seg.page
                cur_section = section_path()

            text = clean_text(seg.text)
            if seg.kind == "table":
                blocks = [Block("table", compact_table(text.strip()))] if text.strip().startswith("|") else split_blocks(text)
            elif seg.kind in ("image_caption", "page_caption"):
                blocks = [Block(seg.kind, text.strip())]
            else:
                blocks = split_blocks(text)

            for block in blocks:
                if block.kind == "heading":
                    if cur_tok >= self.cfg.min_tokens:
                        emit(carry_overlap=False)
                        cur, cur_tok = [], 0
                    title = _HEADING_RE.match(block.text).group(2).strip()
                    headings[:] = [h for h in headings if h[0] < block.level] + [(block.level, title)]
                    if not any(b.kind != "heading" for b in cur):
                        cur_section = section_path()
                    cur.append(block)
                    cur_tok += self.count(block.text)
                    continue

                for piece in self._split_large(block):
                    t = self.count(piece.text)
                    if cur_tok + t > self.cfg.max_tokens and any(b.kind != "heading" for b in cur):
                        emit(carry_overlap=piece.kind == "text")
                    cur.append(piece)
                    cur_tok += t
        emit(carry_overlap=False)
        return chunks
