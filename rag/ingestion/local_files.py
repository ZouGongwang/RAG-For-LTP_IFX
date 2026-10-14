from __future__ import annotations

from pathlib import Path
from typing import Iterator

from rag.ingestion.base import file_hash, is_ignored
from rag.models import SourceItem
from rag.parsing import Parsers


class LocalFilesSource:
    name = "local"
    complete_listing = True

    def __init__(self, root: Path, parsers: Parsers):
        self.root = root
        self.parsers = parsers

    def iter_items(self) -> Iterator[SourceItem]:
        if not self.root.exists():
            raise FileNotFoundError(f"Documents folder not found: {self.root}")
        for path in sorted(self.root.rglob("*")):
            if not path.is_file() or is_ignored(path) or not self.parsers.supports(path):
                continue
            rel = path.relative_to(self.root).as_posix()
            title = path.stem
            yield SourceItem(
                doc_id=f"local:{rel}",
                version=file_hash(path),
                title=title,
                source=self.name,
                load=lambda p=path, t=title: self.parsers.parse_file(p, title=t),
                url=path.resolve().as_uri(),
                breadcrumb=list(Path(rel).parent.parts),
                metadata={"file_type": path.suffix.lower().lstrip("."), "file_path": rel},
            )
