"""Provider-neutral LLM interfaces. Pipelines depend only on these protocols."""

from __future__ import annotations

import re
from typing import Iterator, Protocol, TypedDict


class Message(TypedDict):
    role: str  # "system" | "user" | "assistant"
    content: str


class LLMClient(Protocol):
    name: str

    def generate(self, messages: list[Message]) -> str: ...

    def stream(self, messages: list[Message]) -> Iterator[str]: ...


class VisionClient(Protocol):
    name: str

    def describe(self, image_png: bytes, prompt: str) -> str: ...


_THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL | re.IGNORECASE)


def strip_think(text: str) -> str:
    """Remove reasoning blocks emitted by models such as deepseek-r1."""
    text = _THINK_RE.sub("", text)
    if "</think>" in text.lower():
        text = text[text.lower().rindex("</think>") + len("</think>") :]
    return text.strip()


class ThinkFilter:
    """Streaming counterpart of ``strip_think``: suppresses text inside <think>...</think>."""

    OPEN, CLOSE = "<think>", "</think>"

    def __init__(self) -> None:
        self._buf = ""
        self._inside = False

    def feed(self, piece: str) -> str:
        self._buf += piece
        out = []
        while True:
            tag = self.CLOSE if self._inside else self.OPEN
            idx = self._buf.lower().find(tag)
            if idx == -1:
                # Hold back a possible partial tag at the end of the buffer.
                safe = len(self._buf)
                for k in range(min(len(tag) - 1, len(self._buf)), 0, -1):
                    if tag.startswith(self._buf[-k:].lower()):
                        safe = len(self._buf) - k
                        break
                if not self._inside:
                    out.append(self._buf[:safe])
                self._buf = self._buf[safe:]
                return "".join(out)
            if not self._inside:
                out.append(self._buf[:idx])
            self._buf = self._buf[idx + len(tag) :]
            self._inside = not self._inside

    def flush(self) -> str:
        rest, self._buf = self._buf, ""
        return "" if self._inside else rest


def flatten_messages(messages: list[Message]) -> tuple[str, str]:
    """Split messages into (system_text, conversation_text) for prompt-only backends."""
    system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
    convo = [m for m in messages if m["role"] != "system"]
    if len(convo) == 1:
        return system, convo[0]["content"]
    body = "\n\n".join(f"{m['role'].upper()}: {m['content']}" for m in convo)
    return system, body
