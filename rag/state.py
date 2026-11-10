"""SQLite store for incremental-sync state and the image-caption cache."""

from __future__ import annotations

import sqlite3
import threading
import time
from pathlib import Path


class StateStore:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._lock = threading.Lock()
        with self._lock, self._conn:
            self._conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    doc_id TEXT PRIMARY KEY,
                    source TEXT NOT NULL,
                    version TEXT NOT NULL,
                    title TEXT,
                    n_chunks INTEGER,
                    updated_at REAL
                );
                CREATE TABLE IF NOT EXISTS captions (
                    image_hash TEXT NOT NULL,
                    model TEXT NOT NULL,
                    caption TEXT NOT NULL,
                    PRIMARY KEY (image_hash, model)
                );
                """
            )

    def get_version(self, doc_id: str) -> str | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT version FROM documents WHERE doc_id = ?", (doc_id,)
            ).fetchone()
        return row[0] if row else None

    def set_document(self, doc_id: str, source: str, version: str, title: str, n_chunks: int) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT OR REPLACE INTO documents VALUES (?, ?, ?, ?, ?, ?)",
                (doc_id, source, version, title, n_chunks, time.time()),
            )

    def delete_document(self, doc_id: str) -> None:
        with self._lock, self._conn:
            self._conn.execute("DELETE FROM documents WHERE doc_id = ?", (doc_id,))

    def doc_ids_for_source(self, source: str) -> set[str]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT doc_id FROM documents WHERE source = ?", (source,)
            ).fetchall()
        return {r[0] for r in rows}

    def list_documents(self) -> list[tuple]:
        with self._lock:
            return self._conn.execute(
                "SELECT doc_id, source, version, title, n_chunks, updated_at FROM documents ORDER BY doc_id"
            ).fetchall()

    def clear_documents(self) -> None:
        with self._lock, self._conn:
            self._conn.execute("DELETE FROM documents")

    def get_caption(self, image_hash: str, model: str) -> str | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT caption FROM captions WHERE image_hash = ? AND model = ?", (image_hash, model)
            ).fetchone()
        return row[0] if row else None

    def set_caption(self, image_hash: str, model: str, caption: str) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT OR REPLACE INTO captions VALUES (?, ?, ?)", (image_hash, model, caption)
            )
