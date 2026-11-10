import sys
from pathlib import Path

import pytest

pytest.importorskip("mcp")

from rag.config import MCPConfig  # noqa: E402
from rag.llm.mcp_client import MCPLLM, list_tools  # noqa: E402

SERVER = Path(__file__).with_name("mock_mcp_server.py")


def cfg(**kw) -> MCPConfig:
    return MCPConfig(transport="stdio", command=[sys.executable, str(SERVER)], tool_name="generate", **kw)


def test_list_tools():
    tools = list_tools(cfg())
    assert [t["name"] for t in tools] == ["generate"]
    assert "prompt" in tools[0]["input_schema"]["properties"]


def test_generate_via_mcp_with_system_arg():
    llm = MCPLLM(cfg(system_arg="system"))
    out = llm.generate([{"role": "system", "content": "ABC"}, {"role": "user", "content": "What is BSP?"}])
    assert out == "ECHO system=3 prompt=What is BSP?"
