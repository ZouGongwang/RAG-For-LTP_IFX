"""Stand-in for the company LLM MCP server, used by tests (stdio transport)."""

try:  # mcp >= 2.0
    from mcp.server.mcpserver import MCPServer as Server
except ImportError:  # mcp 1.x
    from mcp.server.fastmcp import FastMCP as Server

server = Server("mock-company-llm")


@server.tool()
def generate(prompt: str, system: str = "") -> str:
    """Generate a completion for a prompt."""
    return f"<think>internal</think>ECHO system={len(system)} prompt={prompt[-20:]}"


if __name__ == "__main__":
    server.run()
