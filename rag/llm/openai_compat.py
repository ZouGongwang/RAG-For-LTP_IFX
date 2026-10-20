"""Client for any OpenAI-style chat endpoint (Azure OpenAI, vLLM, LiteLLM gateways, ...)."""

from __future__ import annotations

import base64
import logging
import os
from typing import Any, Iterator

from rag.config import OpenAICompatConfig, secret
from rag.llm.base import Message, strip_think

log = logging.getLogger(__name__)


def _http_client(timeout_s: float):
    """Company CA bundle (LiteLLM doc: ``httpx.Client(verify=ca-bundle.crt)``)."""
    import httpx

    cert = (os.environ.get("SSL_CERT_FILE") or os.environ.get("SSL_CERT") or "").strip()
    if not cert:
        return None
    if not os.path.isfile(cert):
        raise FileNotFoundError(f"SSL_CERT_FILE points to a missing file: {cert}")
    return httpx.Client(verify=cert, timeout=timeout_s)


class _Chat:
    """Shared chat-completions call; drops ``temperature`` once if the model rejects it."""

    def __init__(self, cfg: OpenAICompatConfig):
        from openai import OpenAI

        if not cfg.base_url or not cfg.model:
            raise RuntimeError("llm.openai_compat.base_url and .model must be set in config.yaml")
        self.cfg = cfg
        self.name = f"openai_compat:{cfg.model}"
        kwargs: dict[str, Any] = dict(
            base_url=cfg.base_url,
            api_key=secret(cfg.api_key_env or "LLM_API_KEY"),
            timeout=cfg.timeout_s,
        )
        if (http_client := _http_client(cfg.timeout_s)) is not None:
            kwargs["http_client"] = http_client
        self._client = OpenAI(**kwargs)
        self._temperature = cfg.temperature

    def _create(self, **kwargs):
        from openai import BadRequestError

        if self._temperature is not None:
            kwargs["temperature"] = self._temperature
        try:
            return self._client.chat.completions.create(model=self.cfg.model, **kwargs)
        except BadRequestError as exc:
            if "temperature" not in kwargs or "temperature" not in str(exc).lower():
                raise
            log.info("%s rejects temperature=%s; using the model default", self.cfg.model, self._temperature)
            self._temperature = None
            kwargs.pop("temperature")
            return self._client.chat.completions.create(model=self.cfg.model, **kwargs)


class OpenAICompatLLM(_Chat):
    def generate(self, messages: list[Message]) -> str:
        resp = self._create(messages=messages)
        return strip_think(resp.choices[0].message.content or "")

    def stream(self, messages: list[Message]) -> Iterator[str]:
        for event in self._create(messages=messages, stream=True):
            if event.choices and event.choices[0].delta.content:
                yield event.choices[0].delta.content


class OpenAICompatVision(_Chat):
    def describe(self, image_png: bytes, prompt: str) -> str:
        data_uri = "data:image/png;base64," + base64.b64encode(image_png).decode()
        resp = self._create(
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
