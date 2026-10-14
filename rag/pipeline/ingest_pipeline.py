"""source -> parse -> chunk -> embed -> upsert, skipping documents whose version is unchanged."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

from rag.ingestion.base import Source
from rag.pipeline.components import Components

log = logging.getLogger(__name__)

SOURCE_NAMES = ("local", "confluence_api", "confluence_export")


@dataclass
class IngestStats:
    added: int = 0
    updated: int = 0
    unchanged: int = 0
    deleted: int = 0
    failed: int = 0
    chunks: int = 0
    seconds: float = 0.0
    errors: list[str] = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"added={self.added} updated={self.updated} unchanged={self.unchanged} "
            f"deleted={self.deleted} failed={self.failed} chunks_written={self.chunks} "
            f"time={self.seconds:.1f}s"
        )


class IngestPipeline:
    def __init__(self, components: Components):
        self.c = components
        self.s = components.settings

    def build_source(self, name: str) -> Source:
        src = self.s.sources
        if name == "local":
            from rag.ingestion.local_files import LocalFilesSource

            return LocalFilesSource(self.s.resolve(src.local.path), self.c.parsers)
        if name == "confluence_api":
            from rag.ingestion.confluence_api import ConfluenceAPISource

            return ConfluenceAPISource(src.confluence_api, self.c.parsers, self.c.describer)
        if name == "confluence_export":
            from rag.ingestion.confluence_export import ConfluenceExportSource

            cfg = src.confluence_export
            return ConfluenceExportSource(self.s.resolve(cfg.path), cfg.space, self.c.parsers, self.c.describer)
        raise ValueError(f"Unknown source '{name}'. Choose from {SOURCE_NAMES}")

    def enabled_sources(self) -> list[str]:
        src = self.s.sources
        return [n for n in SOURCE_NAMES if getattr(src, n).enabled]

    def run(self, source_names: list[str] | None = None, force: bool = False, rebuild: bool = False) -> IngestStats:
        stats = IngestStats()
        started = time.perf_counter()
        store, state = self.c.store, self.c.state
        if rebuild:
            log.info("Rebuilding index from scratch")
            store.reset()
            state.clear_documents()
            force = True

        for name in source_names or self.enabled_sources():
            source = self.build_source(name)
            seen: set[str] = set()
            listing_ok = True
            try:
                for item in source.iter_items():
                    seen.add(item.doc_id)
                    previous = state.get_version(item.doc_id)
                    if previous == item.version and not force:
                        stats.unchanged += 1
                        continue
                    log.info("Ingesting %s", item.doc_id)
                    try:
                        parsed = item.load()
                        chunks = self.c.chunker.chunk(item, parsed)
                        embeddings = self.c.embedder.embed_documents([ch.text for ch in chunks]) if chunks else []
                    except Exception as exc:
                        log.exception("Failed to ingest %s", item.doc_id)
                        stats.failed += 1
                        stats.errors.append(f"{item.doc_id}: {exc}")
                        continue
                    store.delete_doc(item.doc_id)
                    if chunks:
                        store.upsert(chunks, embeddings)
                    else:
                        log.warning("No text extracted from %s", item.doc_id)
                    state.set_document(item.doc_id, source.name, item.version, item.title, len(chunks))
                    stats.chunks += len(chunks)
                    if previous is None:
                        stats.added += 1
                    else:
                        stats.updated += 1
            except Exception as exc:
                listing_ok = False
                log.exception("Listing source %s failed", name)
                stats.errors.append(f"{name}: {exc}")

            if listing_ok and source.complete_listing:
                for doc_id in state.doc_ids_for_source(source.name) - seen:
                    log.info("Removing deleted document %s", doc_id)
                    store.delete_doc(doc_id)
                    state.delete_document(doc_id)
                    stats.deleted += 1

        stats.seconds = time.perf_counter() - started
        return stats
