"""Read an unzipped Confluence space export (HTML export, or a folder of Markdown files).

HTML export layout (Space tools > Content tools > Export > HTML):
    index.html
    Page-Title_123456.html
    attachments/123456/789012.png
"""

from __future__ import annotations

import hashlib
import logging
import re
from pathlib import Path
from typing import Iterator
from urllib.parse import unquote

from bs4 import BeautifulSoup

from rag.ingestion.base import file_hash, is_ignored
from rag.models import ParsedDocument, Segment, SourceItem
from rag.parsing import IMAGE_SUFFIXES_ALL, Parsers
from rag.parsing.confluence_html import export_html_to_markdown
from rag.parsing.figures import ImageDescriber

log = logging.getLogger(__name__)

_PAGE_ID_RE = re.compile(r"(?:_|^)(\d{3,})\.html$")
_MD_IMAGE_RE = re.compile(r"!\[([^\]]*)\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")
_ATTACHMENT_SUFFIXES = {".pdf", ".pptx", ".docx", ".xlsx", ".xlsm", ".xls", ".csv", ".tsv", ".txt"}


class ConfluenceExportSource:
    name = "confluence_export"
    complete_listing = True

    def __init__(self, root: Path, space: str, parsers: Parsers, describer: ImageDescriber):
        self.root = root
        self.space = space
        self.parsers = parsers
        self.describer = describer

    def iter_items(self) -> Iterator[SourceItem]:
        if not self.root.exists():
            raise FileNotFoundError(f"Confluence export folder not found: {self.root}")
        for path in sorted(self.root.rglob("*.html")):
            if path.name == "index.html" or is_ignored(path) or "attachments" in path.parts:
                continue
            yield from self._html_items(path)
        for path in sorted(self.root.rglob("*.md")):
            if not is_ignored(path):
                yield self._markdown_item(path)

    def _describe_file(self, path: Path, context: str) -> str | None:
        if not path.is_file() or not self.describer.available:
            return None
        return self.describer.describe(path.read_bytes(), context=context)

    def _html_items(self, path: Path) -> Iterator[SourceItem]:
        html = path.read_text(encoding="utf-8", errors="replace")
        title, _md, breadcrumb = export_html_to_markdown(html)
        title = title or path.stem
        m = _PAGE_ID_RE.search(path.name)
        page_id = m.group(1) if m else hashlib.sha1(path.name.encode()).hexdigest()[:10]
        rel = path.relative_to(self.root).as_posix()

        def load() -> ParsedDocument:
            def resolve(src: str) -> str | None:
                return self._describe_file((path.parent / unquote(src.split("?")[0])).resolve(), title)

            _t, md, _b = export_html_to_markdown(html, resolve)
            return ParsedDocument([Segment(md)])

        meta = {"space": self.space, "page_id": page_id, "file_type": "confluence_page", "file_path": rel}
        yield SourceItem(
            doc_id=f"confluence_export:{page_id}",
            version=file_hash(path),
            title=title,
            source=self.name,
            load=load,
            url=path.resolve().as_uri(),
            breadcrumb=breadcrumb,
            metadata=meta,
        )

        # The page's attachment list maps stored file names (ids) to original names.
        soup = BeautifulSoup(html, "html.parser")
        names = {
            unquote(a["href"].split("?")[0]): a.get_text(strip=True)
            for a in soup.select('a[href^="attachments/"]')
        }
        att_dir = path.parent / "attachments" / page_id
        if not att_dir.is_dir():
            return
        for att in sorted(att_dir.iterdir()):
            suffix = att.suffix.lower()
            if not att.is_file() or suffix not in _ATTACHMENT_SUFFIXES:
                continue
            att_rel = att.relative_to(path.parent).as_posix()
            name = names.get(att_rel) or att.name
            yield SourceItem(
                doc_id=f"confluence_export:{page_id}:att:{att.name}",
                version=file_hash(att),
                title=name,
                source=self.name,
                load=lambda a=att, n=name: self.parsers.parse_file(a, title=n),
                url=att.resolve().as_uri(),
                breadcrumb=[*breadcrumb, title],
                metadata={**meta, "attachment": name, "file_type": suffix.lstrip(".")},
            )

    def _markdown_item(self, path: Path) -> SourceItem:
        text = path.read_text(encoding="utf-8", errors="replace")
        heading = next((ln[2:].strip() for ln in text.splitlines() if ln.startswith("# ")), "")
        title = heading or path.stem
        rel = path.relative_to(self.root)

        def load() -> ParsedDocument:
            def repl(m: re.Match) -> str:
                alt, src = m.group(1), m.group(2)
                img = (path.parent / unquote(src)).resolve()
                caption = self._describe_file(img, title) if img.suffix.lower() in IMAGE_SUFFIXES_ALL else None
                label = alt or img.name
                return f"[Image: {label}] {caption}" if caption else f"[Image: {label}]"

            return ParsedDocument([Segment(_MD_IMAGE_RE.sub(repl, text))])

        return SourceItem(
            doc_id=f"confluence_export:md:{rel.as_posix()}",
            version=file_hash(path),
            title=title,
            source=self.name,
            load=load,
            url=path.resolve().as_uri(),
            breadcrumb=[self.space, *rel.parent.parts] if self.space else list(rel.parent.parts),
            metadata={"space": self.space, "file_type": "md", "file_path": rel.as_posix()},
        )
