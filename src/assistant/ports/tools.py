"""Tool server port (one MCP server per client)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Protocol


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    input_schema: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ToolResult:
    text: str
    is_error: bool = False
    data: Optional[Mapping[str, Any]] = None
    """The tool's structured output, when it returns one."""


class ToolClient(Protocol):
    async def list_tools(self) -> list[ToolSpec]: ...

    async def call_tool(self, name: str, arguments: Mapping[str, Any]) -> ToolResult: ...
