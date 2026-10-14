"""Backwards-compatible entry point. Prefer: python -m rag ask"""

from rag.api.cli import app

if __name__ == "__main__":
    app(["ask"])
