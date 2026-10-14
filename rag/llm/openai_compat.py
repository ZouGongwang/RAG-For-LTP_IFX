"""Client for any OpenAI-style chat endpoint (Azure OpenAI, vLLM, LiteLLM gateways, ...)."""

from __future__ import annotations

import base64
from typing import Iterator

from rag.config import OpenAICompatConfig, secret
from rag.llm.base import Message, strip_think


def _client(cfg: OpenAICompatConfig):
    from openai import OpenAI

    if not cfg.base_url or not cfg.model:
        raise RuntimeError("llm.openai_compat.base_url and .model must be set in config.yaml")
    return OpenAI(base_url=cfg.base_url, api_key=secret(cfg.api_key_env), timeout=cfg.timeout_s)


class OpenAICompatLLM:
    def __init__(self, cfg: OpenAICompatConfig):
        self.cfg = cfg
        self.name = f"openai_compat:{cfg.model}"
        self._client = _client(cfg)

    def generate(self, messages: list[Message]) -> str:
        resp = self._client.chat.completions.create(
            model=self.cfg.model, messages=messages, temperature=self.cfg.temperature
        )
        return strip_think(resp.choices[0].message.content or "")

    def stream(self, messages: list[Message]) -> Iterator[str]:
        stream = self._client.chat.completions.create(
            model=self.cfg.model, messages=messages, temperature=self.cfg.temperature, stream=True
        )
        for event in stream:
            if event.choices and event.choices[0].delta.content:
                yield event.choices[0].delta.content


class OpenAICompatVision:
    def __init__(self, cfg: OpenAICompatConfig):
        self.cfg = cfg
        self.name = f"openai_compat:{cfg.model}"
        self._client = _client(cfg)

    def describe(self, image_png: bytes, prompt: str) -> str:
        data_uri = "data:image/png;base64," + base64.b64encode(image_png).decode()
        resp = self._client.chat.completions.create(
            model=self.cfg.model,
            temperature=0.0,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": data_uri}},
                    ],
                }
            ],
        )
        return strip_think(resp.choices[0].message.content or "")
