from rag.config import ConfluenceAPIConfig
from rag.ingestion import confluence_api
from rag.ingestion.confluence_client import Attachment, Page
from rag.pipeline import IngestPipeline

BODY = """<h1>Overview</h1><p>The GaN line runs 3 shifts.</p>
<p><ac:image><ri:attachment ri:filename="line.png" /></ac:image></p>"""


class FakeAPI:
    def __init__(self):
        self.page_version = 1
        self.body_calls = 0

    def space_name(self, key):
        return "BSP Space"

    def pages(self, key):
        yield Page("100", "GaN Line", self.page_version, "2026-10-01", ["Manufacturing"], "https://c/pages/100")

    def body(self, page_id):
        self.body_calls += 1
        return BODY

    def attachments(self, page_id):
        return [
            Attachment("a1", "line.png", 1, "image/png", 10, "https://c/dl/line.png", "https://c/a1"),
            Attachment("a2", "notes.txt", 2, "text/plain", 10, "https://c/dl/notes.txt", "https://c/a2"),
            Attachment("a3", "logo.svg", 1, "image/svg+xml", 10, "https://c/dl/logo.svg", "https://c/a3"),
        ]

    def download(self, att):
        return b"Shift handover checklist v2" if att.title == "notes.txt" else b"png-bytes"


class FakeDescriber:
    available = True

    def describe(self, image, **kw):
        return "Diagram of the GaN line with 3 stations"


def test_confluence_source_ingest(tmp_path, make_components, monkeypatch):
    api = FakeAPI()
    monkeypatch.setattr(confluence_api, "connect", lambda cfg: api)
    c = make_components(tmp_path / "unused")
    c.describer = FakeDescriber()
    c.settings.sources.confluence_api = ConfluenceAPIConfig(enabled=True, spaces=["BSP"])

    stats = IngestPipeline(c).run(["confluence_api"])
    assert (stats.added, stats.failed) == (2, 0)  # page + notes.txt (svg skipped)

    page = next(ch for ch in c.store.iter_all() if ch.metadata["doc_id"] == "confluence:100")
    assert page.text.startswith("BSP Space > Manufacturing > GaN Line > Overview")
    assert "[Image: line.png] Diagram of the GaN line" in page.text
    assert page.metadata["url"] == "https://c/pages/100"
    assert page.metadata["space"] == "BSP"

    att = next(ch for ch in c.store.iter_all() if ch.metadata["doc_id"] == "confluence:100:att:a2")
    assert "Shift handover checklist" in att.text
    assert att.metadata["breadcrumb"] == "BSP Space > Manufacturing > GaN Line"

    # Unchanged page is not re-fetched.
    IngestPipeline(c).run(["confluence_api"])
    assert api.body_calls == 1
    api.page_version = 2
    stats = IngestPipeline(c).run(["confluence_api"])
    assert stats.updated == 1 and api.body_calls == 2
