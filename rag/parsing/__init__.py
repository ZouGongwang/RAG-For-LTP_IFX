"""Single entry point that picks the right parser for a file."""

from __future__ import annotations

import logging
from pathlib import Path

from rag.config import ParsingConfig
from rag.models import ParsedDocument, Segment
from rag.parsing.confluence_html import export_html_to_markdown
from rag.parsing.figures import ImageDescriber
from rag.parsing.spreadsheet import SPREADSHEET_SUFFIXES, SpreadsheetParser

log = logging.getLogger(__name__)

TEXT_SUFFIXES = {".md", ".markdown", ".txt"}
HTML_SUFFIXES = {".html", ".htm"}
IMAGE_SUFFIXES_ALL = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp", ".tif", ".tiff"}
BINARY_SUFFIXES = {".pdf", ".pptx", ".docx"} | IMAGE_SUFFIXES_ALL
SUPPORTED_SUFFIXES = TEXT_SUFFIXES | HTML_SUFFIXES | BINARY_SUFFIXES | SPREADSHEET_SUFFIXES


class Parsers:
    def __init__(self, cfg: ParsingConfig, describer: ImageDescriber):
        self.cfg = cfg
        self.describer = describer
        self._binary = None

    def _binary_parser(self):
        if self._binary is not None:
            return self._binary
        if self.cfg.backend == "docling":
            try:
                import docling  # noqa: F401

                from rag.parsing.docling_parser import DoclingParser

                self._binary = DoclingParser(self.cfg, self.describer)
                return self._binary
            except ImportError:
                log.warning("Docling is not installed; falling back to the simple parser.")
        from rag.parsing.simple_parser import SimpleParser

        self._binary = SimpleParser(self.cfg, self.describer)
        return self._binary

    def supports(self, path: Path) -> bool:
        return path.suffix.lower() in SUPPORTED_SUFFIXES

    def parse_file(self, path: Path, title: str = "") -> ParsedDocument:
        suffix = path.suffix.lower()
        if suffix in TEXT_SUFFIXES:
            return ParsedDocument([Segment(path.read_text(encoding="utf-8", errors="replace"))])
        if suffix in HTML_SUFFIXES:
            _title, md, _crumbs = export_html_to_markdown(path.read_text(encoding="utf-8", errors="replace"))
            return ParsedDocument([Segment(md)])
        if suffix in SPREADSHEET_SUFFIXES:
            return SpreadsheetParser(self.cfg).parse(path, title=title)
        return self._binary_parser().parse(path, title=title)
