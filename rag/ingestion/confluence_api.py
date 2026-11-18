"""Pull pages and attachments live from Confluence (Cloud or Data Center).

Credentials come from environment variables named in config (see .env.example):
* Data Center / Server: CONFLUENCE_URL + CONFLUENCE_TOKEN (personal access token).
* Cloud: CONFLUENCE_URL (https://<site>.atlassian.net/wiki) + CONFLUENCE_USER (email)
  + CONFLUENCE_TOKEN (API token from id.atlassian.com).
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Iterator

from rag.config import ConfluenceAPIConfig
from rag.ingestion.confluence_client import Attachment, Page, connect
from rag.models import ParsedDocument, Segment, SourceItem
from rag.parsing import Parsers
from rag.parsing.confluence_html import storage_to_markdown
from rag.parsing.figures import ImageDescriber

log = logging.getLogger(__name__)

_ATTACHMENT_SUFFIXES = {".pdf", ".pptx", ".docx", ".xlsx", ".xlsm", ".xls", ".csv", ".tsv", ".md", ".txt"}


class ConfluenceAPISource:
    name = "confluence"

    def __init__(self, cfg: ConfluenceAPIConfig, parsers: Parsers, describer: ImageDescriber):
        if not cfg.spaces:
            raise RuntimeError("sources.confluence_api.spaces is empty in config.yaml")
        self.cfg = cfg
        self.parsers = parsers
        self.describer = describer
        self.api = connect(cfg)
        self.complete_listing = cfg.page_limit is None

    def iter_items(self) -> Iterator[SourceItem]:
        for space in self.cfg.spaces:
            space_name = self.api.space_name(space)
            log.info("Listing Confluence space %s (%s)", space, space_name)
            for n, page in enumerate(self.api.pages(space)):
                if self.cfg.page_limit and n >= self.cfg.page_limit:
                    break
                yield from self._page_items(space, space_name, page)

    def _page_items(self, space: str, space_name: str, page: Page) -> Iterator[SourceItem]:
        breadcrumb = [space_name, *page.ancestors]
        common = {"space": space, "page_id": page.id, "last_modified": page.when}

        attachments = self.api.attachments(page.id) if self.cfg.include_attachments else []
        images = {a.title: a for a in attachments if a.media_type.startswith("image/")}
        image_sig = ",".join(sorted(f"{a.id}:{a.version}" for a in images.values()))

        def load_page() -> ParsedDocument:
            def resolve_image(filename: str) -> str | None:
                att = images.get(filename)
                if att is None or not self.describer.available:
                    return None
                try:
                    return self.describer.describe(self.api.download(att), context=page.title)
                except Exception as exc:
                    log.warning("Could not fetch image %s on page %s: %s", filename, page.title, exc)
                    return None

            return ParsedDocument([Segment(storage_to_markdown(self.api.body(page.id), resolve_image))])

        yield SourceItem(
            doc_id=f"confluence:{page.id}",
            version=f"{page.version}|{image_sig}",
            title=page.title,
            source=self.name,
            load=load_page,
            url=page.url,
            breadcrumb=breadcrumb,
            metadata={**common, "file_type": "confluence_page"},
        )

        max_bytes = self.cfg.max_attachment_mb * 1024 * 1024
        for att in attachments:
            suffix = Path(att.title).suffix.lower()
            if suffix not in _ATTACHMENT_SUFFIXES:
                continue
            if att.size > max_bytes:
                log.info("Skipping large attachment %s (%d MB)", att.title, att.size // (1024 * 1024))
                continue
            yield SourceItem(
                doc_id=f"confluence:{page.id}:att:{att.id}",
                version=str(att.version),
                title=att.title,
                source=self.name,
                load=lambda att=att, suffix=suffix: self._load_attachment(att, suffix),
                url=att.url,
                breadcrumb=[*breadcrumb, page.title],
                metadata={**common, "attachment": att.title, "file_type": suffix.lstrip(".")},
            )

    def _load_attachment(self, att: Attachment, suffix: str) -> ParsedDocument:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / f"attachment{suffix}"
            path.write_bytes(self.api.download(att))
            return self.parsers.parse_file(path, title=att.title)
