from __future__ import annotations

import hashlib
import math
import re
from pathlib import Path

import pytest

from rag.config import Settings
from rag.pipeline.components import Components


class FakeEmbedder:
    """Deterministic bag-of-words hashing embedder; no model download."""

    name = "fake-embedder"
    dim = 256

    def _vec(self, text: str) -> list[float]:
        v = [0.0] * self.dim
        for tok in re.findall(r"\w+", text.lower()):
            v[int(hashlib.md5(tok.encode()).hexdigest(), 16) % self.dim] += 1.0
        norm = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / norm for x in v]

    def embed_documents(self, texts):
        return [self._vec(t) for t in texts]

    def embed_query(self, text):
        return self._vec(text)

    def count_tokens(self, text):
        return len(text.split())


class FakeLLM:
    name = "fake-llm"

    def __init__(self):
        self.last_messages = None

    def generate(self, messages):
        self.last_messages = messages
        return "<think>reasoning</think>The capacity target is 42 wafers [1]."

    def stream(self, messages):
        yield self.generate(messages)


@pytest.fixture
def make_components(tmp_path: Path):
    def _make(docs_dir: Path, **overrides) -> Components:
        settings = Settings(
            paths={"data_dir": str(tmp_path / "data"), "models_dir": None},
            sources={"local": {"enabled": True, "path": str(docs_dir)}},
            embedding={"model": FakeEmbedder.name},
            vision={"enabled": False},
            parsing={"backend": "simple", "describe_images": False, "describe_low_text_pages": False},
            **overrides,
        )
        c = Components(settings)
        c.embedder = FakeEmbedder()
        c.llm = FakeLLM()
        return c

    return _make
