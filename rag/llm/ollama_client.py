from __future__ import annotations

from typing import Iterator

import ollama

from rag.config import OllamaConfig
from rag.llm.base import Message, ThinkFilter, strip_think


class OllamaLLM:
    def __init__(self, cfg: OllamaConfig):
        self.cfg = cfg
        self.name = f"ollama:{cfg.model}"
        self._client = ollama.Client(host=cfg.host, timeout=cfg.timeout_s)
        self._options = {"temperature": cfg.temperature, "num_ctx": cfg.num_ctx}

    def generate(self, messages: list[Message]) -> str:
        resp = self._client.chat(model=self.cfg.model, messages=messages, options=self._options)
        return strip_think(resp["message"]["content"])

    def stream(self, messages: list[Message]) -> Iterator[str]:
        filt = ThinkFilter()
        for part in self._client.chat(
            model=self.cfg.model, messages=messages, options=self._options, stream=True
        ):
            text = filt.feed(part["message"]["content"])
            if text:
                yield text
        tail = filt.flush()
        if tail:
            yield tail


class OllamaVision:
    def __init__(self, cfg: OllamaConfig):
        self.cfg = cfg
        self.name = f"ollama:{cfg.model}"
        self._client = ollama.Client(host=cfg.host, timeout=cfg.timeout_s)

    def describe(self, image_png: bytes, prompt: str) -> str:
        resp = self._client.chat(
            model=self.cfg.model,
            messages=[{"role": "user", "content": prompt, "images": [image_png]}],
            options={"temperature": 0.0},
        )
        return strip_think(resp["message"]["content"])
