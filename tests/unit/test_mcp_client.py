"""McpHttpToolClient against a real in-process MCP server (Streamable HTTP on localhost)."""

from __future__ import annotations

import asyncio
import socket
from typing import AsyncIterator

import pytest
import uvicorn
from mcp.server.mcpserver import MCPServer
from starlette.types import ASGIApp, Receive, Scope, Send

from assistant.adapters.mcp.client import McpHttpToolClient
from assistant.adapters.net.allowlist import EgressBlocked, HttpClientFactory


class RecordAuth:
    """ASGI middleware that remembers the Authorization header of each request."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app
        self.seen: list[str] = []

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "http":
            headers = dict(scope["headers"])
            self.seen.append(headers.get(b"authorization", b"").decode())
        await self.app(scope, receive, send)


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


@pytest.fixture
async def server() -> AsyncIterator[tuple[str, RecordAuth]]:
    mcp = MCPServer("test")

    @mcp.tool()
    def search_threads(query: str = "") -> str:
        """Lists email threads."""
        return f"2 threads for {query or 'everything'}"

    @mcp.tool()
    def fail() -> str:
        """Always fails."""
        raise ValueError("boom")

    app = RecordAuth(mcp.streamable_http_app())
    port = free_port()
    uv = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning", lifespan="on"))
    task = asyncio.create_task(uv.serve())
    while not uv.started:
        await asyncio.sleep(0.05)
    yield f"http://127.0.0.1:{port}/mcp", app
    uv.should_exit = True
    await task


async def test_lists_and_calls_tools_with_bearer_token(server: tuple[str, RecordAuth]) -> None:
    url, recorder = server

    async def token() -> str:
        return "tok-123"

    client = McpHttpToolClient(url, HttpClientFactory(()), token)

    tools = {spec.name: spec for spec in await client.list_tools()}
    result = await client.call_tool("search_threads", {"query": "is:unread"})
    failed = await client.call_tool("fail", {})

    assert set(tools) == {"search_threads", "fail"}
    assert "query" in tools["search_threads"].input_schema["properties"]
    assert result.text == "2 threads for is:unread" and not result.is_error
    assert failed.is_error
    assert recorder.seen and all(h == "Bearer tok-123" for h in recorder.seen)


async def test_mcp_traffic_obeys_the_allowlist() -> None:
    client = McpHttpToolClient("https://gmailmcp.googleapis.com/mcp/v1", HttpClientFactory(()))
    with pytest.raises(BaseException) as caught:
        await client.list_tools()
    assert _contains(caught.value, EgressBlocked)


def _contains(error: BaseException, kind: type[BaseException]) -> bool:
    """The SDK runs requests in task groups, so the error may arrive wrapped in an ExceptionGroup."""
    if isinstance(error, kind):
        return True
    children = getattr(error, "exceptions", ()) or ()
    cause = error.__cause__ or error.__context__
    return any(_contains(e, kind) for e in children) or (cause is not None and _contains(cause, kind))
