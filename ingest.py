"""Backwards-compatible entry point. Prefer: python -m rag ingest"""

from rag.api.cli import app

if __name__ == "__main__":
    app(["ingest", "--source", "local"])
