from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterator, Protocol

from rag.models import SourceItem


class Source(Protocol):
    name: str
    # True when iter_items() lists every document, so anything not listed was deleted upstream.
    complete_listing: bool

    def iter_items(self) -> Iterator[SourceItem]: ...


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()[:16]


def is_ignored(path: Path) -> bool:
    name = path.name
    return name.startswith((".", "~$")) or name in {"Thumbs.db", "desktop.ini"}
