from fastapi.testclient import TestClient

from rag.api import server
from rag.pipeline import IngestPipeline


def test_query_endpoint(tmp_path, make_components, monkeypatch):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "cap.md").write_text("# Capacity\n\nThe capacity target is 42 wafers.", encoding="utf-8")
    c = make_components(docs)
    IngestPipeline(c).run()
    monkeypatch.setattr(server, "components", lambda: c)

    client = TestClient(server.app)
    assert client.get("/health").json()["chunks"] == 1

    resp = client.post("/query", json={"question": "What is the capacity target?"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["answer"] == "The capacity target is 42 wafers [1]."
    assert body["sources"][0]["n"] == 1
    assert body["sources"][0]["title"] == "cap"
