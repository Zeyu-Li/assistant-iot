"""ToolClient for one remote MCP server over Streamable HTTP.

Each operation opens a short session (initialize, then the call). That costs a round
trip per call but survives dropped connections and expiring tokens without any
reconnect logic, which matters more for an always-on device than a few hundred ms.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Awaitable, Callable, Mapping, Optional, TypeVar

from mcp import ClientSession, types
from mcp.client.streamable_http import streamable_http_client
from pydantic import BaseModel, ConfigDict

from assistant.adapters.net.allowlist import HttpClientFactory
from assistant.ports.tools import ToolResult, ToolSpec

log = logging.getLogger(__name__)

T = TypeVar("T")
TokenProvider = Callable[[], Awaitable[str]]


class McpHttpSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    url: str
    auth: Optional[str] = None
    """"google" to send a Google OAuth bearer token; omit for servers without auth."""
    timeout_seconds: float = 30.0


class McpHttpToolClient:
    def __init__(
        self,
        url: str,
        http: HttpClientFactory,
        token_provider: Optional[TokenProvider] = None,
        timeout_seconds: float = 30.0,
    ) -> None:
        self._url = url
        self._http = http
        self._token_provider = token_provider
        self._timeout = timeout_seconds

    async def list_tools(self) -> list[ToolSpec]:
        async def op(session: ClientSession) -> list[ToolSpec]:
            tools: list[ToolSpec] = []
            cursor: Optional[str] = None
            while True:
                params = types.PaginatedRequestParams(cursor=cursor) if cursor else None
                page = await session.list_tools(params=params)
                tools += [ToolSpec(t.name, t.description or "", t.input_schema or {}) for t in page.tools]
                cursor = page.next_cursor
                if not cursor:
                    return tools

        return await self._with_session(op)

    async def call_tool(self, name: str, arguments: Mapping[str, Any]) -> ToolResult:
        async def op(session: ClientSession) -> ToolResult:
            result = await session.call_tool(name, dict(arguments))
            if not isinstance(result, types.CallToolResult):
                return ToolResult(f"unexpected result type {type(result).__name__}", is_error=True)
            return to_tool_result(result)

        return await self._with_session(op)

    async def _with_session(self, op: Callable[[ClientSession], Awaitable[T]]) -> T:
        headers = {}
        if self._token_provider is not None:
            headers["Authorization"] = f"Bearer {await self._token_provider()}"
        async with self._http.create_for_mcp(timeout=self._timeout, headers=headers) as client:
            async with streamable_http_client(self._url, http_client=client) as streams:
                read, write = streams[0], streams[1]
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    return await op(session)


def to_tool_result(result: types.CallToolResult) -> ToolResult:
    texts = [block.text for block in result.content if isinstance(block, types.TextContent)]
    data = result.structured_content if isinstance(result.structured_content, dict) else None
    text = "\n".join(texts) or (json.dumps(data) if data is not None else "")
    return ToolResult(text=text, is_error=bool(result.is_error), data=data)
