"""Lightweight fallback parser (no layout models). Use ``parsing.backend: simple`` if Docling
cannot be installed, e.g. on a locked-down laptop."""

from __future__ import annotations

import logging
from pathlib import Path

from rag.config import ParsingConfig
from rag.models import ParsedDocument, Segment
from rag.parsing.figures import ImageDescriber

log = logging.getLogger(__name__)

SIMPLE_SUFFIXES = {".pdf", ".pptx", ".docx", ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp"}


def _table_to_md(rows: list[list[str]]) -> str:
    rows = [[c.replace("\n", " ").replace("|", "\\|").strip() for c in r] for r in rows if any(c.strip() for c in r)]
    if not rows:
        return ""
    width = max(len(r) for r in rows)
    rows = [r + [""] * (width - len(r)) for r in rows]
    lines = ["| " + " | ".join(rows[0]) + " |", "|" + "---|" * width]
    lines += ["| " + " | ".join(r) + " |" for r in rows[1:]]
    return "\n".join(lines)


class SimpleParser:
    def __init__(self, cfg: ParsingConfig, describer: ImageDescriber):
        self.cfg = cfg
        self.describer = describer

    def parse(self, path: Path, title: str = "") -> ParsedDocument:
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            return self._pdf(path, title)
        if suffix == ".pptx":
            return self._pptx(path, title)
        if suffix == ".docx":
            return self._docx(path)
        desc = self.describer.describe(path.read_bytes(), context=title)
        return ParsedDocument([Segment(f"[Image]\n{desc}", None, "image_caption")] if desc else [])

    def _pdf(self, path: Path, title: str) -> ParsedDocument:
        import pypdfium2 as pdfium

        segments: list[Segment] = []
        pdf = pdfium.PdfDocument(str(path))
        try:
            for i in range(len(pdf)):
                page = pdf[i]
                text = page.get_textpage().get_text_range().strip()
                if text:
                    segments.append(Segment(text, i + 1))
                if len(text) < self.cfg.low_text_threshold and self.cfg.describe_low_text_pages:
                    pil = page.render(scale=self.cfg.image_scale).to_pil()
                    desc = self.describer.describe(pil, whole_page=True, context=title)
                    if desc:
                        segments.append(Segment(f"[Page {i + 1} visual summary]\n{desc}", i + 1, "page_caption"))
        finally:
            pdf.close()
        return ParsedDocument(segments)

    def _pptx(self, path: Path, title: str) -> ParsedDocument:
        from pptx import Presentation
        from pptx.enum.shapes import MSO_SHAPE_TYPE

        segments: list[Segment] = []
        for n, slide in enumerate(Presentation(str(path)).slides, start=1):
            lines: list[str] = []
            if slide.shapes.title is not None and slide.shapes.title.has_text_frame:
                lines.append(f"## {slide.shapes.title.text_frame.text.strip()}")
            for shape in slide.shapes:
                if shape == slide.shapes.title:
                    continue
                if shape.has_text_frame:
                    for para in shape.text_frame.paragraphs:
                        t = "".join(r.text for r in para.runs).strip()
                        if t:
                            lines.append(("  " * para.level) + f"- {t}")
                elif getattr(shape, "has_table", False) and shape.has_table:
                    rows = [[c.text for c in row.cells] for row in shape.table.rows]
                    md = _table_to_md(rows)
                    if md:
                        segments.append(Segment(md, n, "table"))
                elif shape.shape_type == MSO_SHAPE_TYPE.PICTURE and self.cfg.describe_images:
                    desc = self.describer.describe(shape.image.blob, context=title)
                    if desc:
                        segments.append(Segment(f"[Figure]\n{desc}", n, "image_caption"))
            if slide.has_notes_slide:
                notes = slide.notes_slide.notes_text_frame.text.strip()
                if notes:
                    lines.append(f"Speaker notes: {notes}")
            if lines:
                segments.append(Segment("\n".join(lines), n))
        segments.sort(key=lambda s: s.page or 0)
        return ParsedDocument(segments)

    def _docx(self, path: Path) -> ParsedDocument:
        import docx

        d = docx.Document(str(path))
        lines: list[str] = []
        for para in d.paragraphs:
            t = para.text.strip()
            if not t:
                continue
            style = (para.style.name or "").lower() if para.style is not None else ""
            if style.startswith("heading"):
                level = int(style.split()[-1]) if style.split()[-1].isdigit() else 1
                lines.append(f"{'#' * min(level + 1, 6)} {t}")
            else:
                lines.append(t)
        segments = [Segment("\n\n".join(lines))] if lines else []
        for table in d.tables:
            md = _table_to_md([[c.text for c in row.cells] for row in table.rows])
            if md:
                segments.append(Segment(md, None, "table"))
        return ParsedDocument(segments)
