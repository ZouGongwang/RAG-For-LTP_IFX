"""Use an LLM exposed as a tool on an MCP server (e.g. the company cloud LLM).

The exact tool name and argument names depend on how IT publishes the server, so they
are all configurable under ``llm.mcp`` in config.yaml. Run ``rag mcp-tools`` to list the
tools and input schemas the server offers, then map them in config.
"""

from __future__ import annotations

import asyncio
import base64
from contextlib import AsyncExitStack, asynccontextmanager
from typing import Any, AsyncIterator, Iterator

from rag.config import MCPConfig, secret
from rag.llm.base import Message, flatten_messages, strip_think


@asynccontextmanager
async def _session(cfg: MCPConfig) -> AsyncIterator[Any]:
    from mcp import ClientSession

    headers = dict(cfg.headers)
    token = secret(cfg.auth_token_env, required=False)
    if token:
        headers.setdefault("Authorization", f"Bearer {token}")

    if cfg.transport != "stdio" and not cfg.url:
        raise RuntimeError("llm.mcp.url must be set in config.yaml")

    async with AsyncExitStack() as stack:
        if cfg.transport == "stdio":
            from mcp.client.stdio import StdioServerParameters, stdio_client

            if not cfg.command:
                raise RuntimeError("llm.mcp.command must be set for the stdio transport")
            params = StdioServerParameters(command=cfg.command[0], args=cfg.command[1:])
            transport = stdio_client(params)
        elif cfg.transport == "sse":
            from mcp.client.sse import sse_client

            transport = sse_client(cfg.url, headers=headers)
        else:
            try:  # mcp >= 2.0
                from mcp.client.streamable_http import streamable_http_client
                from mcp.shared._httpx_utils import create_mcp_http_client

                http_client = await stack.enter_async_context(create_mcp_http_client(headers=headers))
                transport = streamable_http_client(cfg.url, http_client=http_client)
            except ImportError:  # mcp 1.x
                from mcp.client.streamable_http import streamablehttp_client

                transport = streamablehttp_client(cfg.url, headers=headers)

        streams = await stack.enter_async_context(transport)
        session = await stack.enter_async_context(ClientSession(streams[0], streams[1]))
        await session.initialize()
        yield session


def _run(coro):
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    # Called from inside an event loop: run in a separate thread with its own loop.
    import concurrent.futures

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coro).result()


def _field(obj: Any, snake: str, camel: str) -> Any:
    """mcp 2.x uses snake_case attribute names, 1.x camelCase."""
    return getattr(obj, snake, None) if hasattr(obj, snake) else getattr(obj, camel, None)


def _result_text(result: Any) -> str:
    parts = [getattr(c, "text", "") for c in (getattr(result, "content", None) or []) if getattr(c, "type", "") == "text"]
    text = "\n".join(p for p in parts if p)
    if _field(result, "is_error", "isError"):
        raise RuntimeError(f"MCP tool returned an error: {text}")
    sc = _field(result, "structured_content", "structuredContent")
    if not text and sc:
        text = str(sc.get("result", sc)) if isinstance(sc, dict) else str(sc)
    return text


async def _call_tool(cfg: MCPConfig, arguments: dict[str, Any]) -> str:
    async with _session(cfg) as session:
        result = await asyncio.wait_for(
            session.call_tool(cfg.tool_name, arguments), timeout=cfg.timeout_s
        )
        return _result_text(result)


async def _list_tools(cfg: MCPConfig) -> list[dict[str, Any]]:
    async with _session(cfg) as session:
        tools = await session.list_tools()
        return [
            {"name": t.name, "description": t.description, "input_schema": _field(t, "input_schema", "inputSchema")}
            for t in tools.tools
        ]


def list_tools(cfg: MCPConfig) -> list[dict[str, Any]]:
    return _run(_list_tools(cfg))


class MCPLLM:
    def __init__(self, cfg: MCPConfig):
        self.cfg = cfg
        self.name = f"mcp:{cfg.tool_name}@{cfg.url or ' '.join(cfg.command)}"

    def _arguments(self, messages: list[Message]) -> dict[str, Any]:
        args = dict(self.cfg.extra_args)
        if self.cfg.argument_style == "messages":
            args[self.cfg.messages_arg] = [dict(m) for m in messages]
            return args
        system, prompt = flatten_messages(messages)
        if self.cfg.system_arg:
            args[self.cfg.system_arg] = system
        elif system:
            prompt = f"{system}\n\n{prompt}"
        args[self.cfg.prompt_arg] = prompt
        return args

    def generate(self, messages: list[Message]) -> str:
        return strip_think(_run(_call_tool(self.cfg, self._arguments(messages))))

    def stream(self, messages: list[Message]) -> Iterator[str]:
        # MCP tool calls return a single result; yield it as one piece.
        yield self.generate(messages)


class MCPVision:
    def __init__(self, cfg: MCPConfig):
        self.cfg = cfg
        self.name = f"mcp:{cfg.tool_name}"

    def describe(self, image_png: bytes, prompt: str) -> str:
        args = dict(self.cfg.extra_args)
        args[self.cfg.prompt_arg] = prompt
        args[self.cfg.image_arg] = base64.b64encode(image_png).decode()
        return strip_think(_run(_call_tool(self.cfg, args)))
