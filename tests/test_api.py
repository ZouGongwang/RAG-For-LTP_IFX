import json

from fastapi.testclient import TestClient

from rag.api import server
from rag.pipeline import IngestPipeline


def test_home_page_serves_chat_widget():
    client = TestClient(server.app)
    page = client.get("/")
    assert page.status_code == 200
    assert 'src="/widget.js"' in page.text
    script = client.get("/widget.js")
    assert script.status_code == 200
    assert "bsp-launcher" in script.text
    assert "/query/stream" in script.text


def test_confluence_origin_can_call_the_api():
    resp = TestClient(server.app).options(
        "/query/stream",
        headers={
            "Origin": "https://confluencewikiprod.intra.infineon.com",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type",
        },
    )
    assert resp.headers.get("access-control-allow-origin") == "https://confluencewikiprod.intra.infineon.com"


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


def test_query_stream_matches_terminal_answer(tmp_path, make_components, monkeypatch):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "cap.md").write_text("# Capacity\n\nThe capacity target is 42 wafers.", encoding="utf-8")
    c = make_components(docs)
    IngestPipeline(c).run()
    monkeypatch.setattr(server, "components", lambda: c)

    resp = TestClient(server.app).post("/query/stream", json={"question": "What is the capacity target?"})
    assert resp.status_code == 200
    events = [json.loads(line) for line in resp.text.splitlines() if line.strip()]
    assert events[-1]["type"] == "done"
    assert events[-1]["answer"] == "The capacity target is 42 wafers [1]."
    assert any(event.get("text") for event in events if event["type"] == "delta")
