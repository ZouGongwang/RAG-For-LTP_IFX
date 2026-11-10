"""Layout-aware parsing of PDF, PPTX, DOCX, XLSX and images with Docling (runs locally)."""

from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path

from rag.config import ParsingConfig
from rag.models import ParsedDocument, Segment
from rag.parsing.figures import ImageDescriber

log = logging.getLogger(__name__)

DOCLING_SUFFIXES = {".pdf", ".pptx", ".docx", ".xlsx", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".gif", ".webp"}


class DoclingParser:
    def __init__(self, cfg: ParsingConfig, describer: ImageDescriber):
        self.cfg = cfg
        self.describer = describer
        self._converter = None

    def _get_converter(self):
        if self._converter is not None:
            return self._converter
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import PdfPipelineOptions
        from docling.document_converter import DocumentConverter, ImageFormatOption, PdfFormatOption

        opts = PdfPipelineOptions()
        if self.cfg.docling_artifacts_path:
            opts.artifacts_path = str(self.cfg.docling_artifacts_path)
        opts.do_ocr = self.cfg.ocr
        opts.do_table_structure = self.cfg.table_structure
        opts.images_scale = self.cfg.image_scale
        opts.generate_picture_images = self.cfg.describe_images
        opts.generate_page_images = self.cfg.describe_low_text_pages

        self._converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=opts),
                InputFormat.IMAGE: ImageFormatOption(pipeline_options=opts),
            }
        )
        return self._converter

    def parse(self, path: Path, title: str = "") -> ParsedDocument:
        from docling_core.types.doc import (
            ListItem,
            PictureItem,
            SectionHeaderItem,
            TableItem,
            TextItem,
            TitleItem,
        )

        if path.suffix.lower() not in DOCLING_SUFFIXES:
            desc = self.describer.describe(path.read_bytes(), context=title)
            return ParsedDocument([Segment(f"[Image]\n{desc}", None, "image_caption")] if desc else [])

        doc = self._get_converter().convert(str(path)).document
        is_image_file = path.suffix.lower() in IMAGE_SUFFIXES
        is_pdf = path.suffix.lower() == ".pdf"

        segments: list[Segment] = []
        text_buf: list[str] = []
        buf_page: int | None = None
        chars_per_page: dict[int, int] = defaultdict(int)

        def flush() -> None:
            nonlocal text_buf
            if text_buf:
                segments.append(Segment("\n\n".join(text_buf), page=buf_page))
                text_buf = []

        for item, _level in doc.iterate_items():
            prov = getattr(item, "prov", None)
            page = prov[0].page_no if prov else None
            if page != buf_page:
                flush()
                buf_page = page

            if isinstance(item, TableItem):
                flush()
                md = item.export_to_markdown(doc=doc).strip()
                if md:
                    caption = item.caption_text(doc)
                    segments.append(Segment(f"{caption}\n\n{md}" if caption else md, page, "table"))
                    chars_per_page[page or 0] += len(md)
            elif isinstance(item, PictureItem):
                if is_image_file or not self.cfg.describe_images:
                    continue
                caption = item.caption_text(doc)
                desc = None
                try:
                    img = item.get_image(doc)
                    if img is not None:
                        desc = self.describer.describe(img, context=title)
                except Exception as exc:
                    log.debug("Picture extraction failed: %s", exc)
                if desc or caption:
                    flush()
                    head = f"[Figure: {caption}]" if caption else "[Figure]"
                    segments.append(Segment(f"{head}\n{desc or ''}".strip(), page, "image_caption"))
            elif isinstance(item, TitleItem):
                text_buf.append(f"# {item.text.strip()}")
            elif isinstance(item, SectionHeaderItem):
                text_buf.append(f"{'#' * min(item.level + 1, 6)} {item.text.strip()}")
            elif isinstance(item, ListItem):
                marker = (getattr(item, "marker", "") or "-").strip() or "-"
                text_buf.append(f"{marker} {item.text.strip()}")
            elif isinstance(item, TextItem):
                if item.text.strip():
                    text_buf.append(item.text.strip())
                    chars_per_page[page or 0] += len(item.text)
        flush()

        if is_image_file:
            desc = self.describer.describe(path.read_bytes(), context=title)
            if desc:
                segments.append(Segment(f"[Image]\n{desc}", None, "image_caption"))

        if is_pdf and self.cfg.describe_low_text_pages and self.describer.available:
            for page_no, page in sorted(doc.pages.items()):
                if chars_per_page.get(page_no, 0) >= self.cfg.low_text_threshold:
                    continue
                pil = page.image.pil_image if page.image is not None else None
                if pil is None:
                    continue
                desc = self.describer.describe(pil, whole_page=True, context=title)
                if desc:
                    segments.append(Segment(f"[Page {page_no} visual summary]\n{desc}", page_no, "page_caption"))

        segments.sort(key=lambda s: (s.page or 0))
        return ParsedDocument(segments)
