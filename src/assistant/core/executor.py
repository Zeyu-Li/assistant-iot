"""Run a capability through its flow handler or its pinned MCP tool."""

from __future__ import annotations

import logging
from typing import Any, Mapping, Optional

from assistant.core.registry import Capability, McpBinding, Registry
from assistant.core.render import ResultRenderer
from assistant.ports.flow import DescribesAction, FlowContext, FlowHandler
from assistant.ports.tools import ToolClient, ToolSpec

log = logging.getLogger(__name__)

_NO_ARGUMENTS: Mapping[str, Any] = {"type": "object", "properties": {}}


class CapabilityExecutor:
    def __init__(
        self,
        flows: Mapping[str, FlowHandler],
        tool_clients: Mapping[str, ToolClient],
        renderer: ResultRenderer,
        failure_text: str,
    ) -> None:
        self._flows = flows
        self._tool_clients = tool_clients
        self._renderer = renderer
        self._failure_text = failure_text
        self._tool_specs: dict[str, dict[str, ToolSpec]] = {}

    async def verify(self, registry: Registry) -> None:
        """Fail at startup, not mid-conversation, if a capability has nothing behind it."""
        for capability in registry:
            if capability.kind == "flow":
                if capability.id not in self._flows:
                    raise LookupError(f"no flow handler registered for capability '{capability.id}'")
            else:
                await self._tool_spec(capability.id, _binding(capability))

    async def schema_for(self, capability: Capability) -> Mapping[str, Any]:
        if capability.slots is not None:
            return capability.slots
        if capability.kind == "flow":
            return _NO_ARGUMENTS
        spec = await self._tool_spec(capability.id, _binding(capability))
        return spec.input_schema or _NO_ARGUMENTS

    def describe(self, capability: Capability, arguments: Mapping[str, Any]) -> Optional[str]:
        """The flow's own wording for a confirmation question, if it has one."""
        handler = self._flows.get(capability.id) if capability.kind == "flow" else None
        if isinstance(handler, DescribesAction):
            try:
                return handler.describe(arguments)
            except (KeyError, ValueError):
                log.exception("could not describe %s", capability.id)
        return None

    async def execute(self, capability: Capability, arguments: Mapping[str, Any], request: str) -> str:
        try:
            if capability.kind == "flow":
                return await self._flows[capability.id].run(FlowContext(request, arguments))
            binding = _binding(capability)
            result = await self._tool_clients[binding.server].call_tool(binding.tool, arguments)
            return await self._renderer.render(capability, result, request)
        except Exception:  # top-level action boundary: log it and say something instead of going silent
            log.exception("capability %s failed", capability.id)
            return self._failure_text

    async def _tool_spec(self, capability_id: str, binding: McpBinding) -> ToolSpec:
        client = self._tool_clients.get(binding.server)
        if client is None:
            raise LookupError(f"capability '{capability_id}' needs MCP server '{binding.server}', which the profile does not define")
        if binding.server not in self._tool_specs:
            self._tool_specs[binding.server] = {spec.name: spec for spec in await client.list_tools()}
        specs = self._tool_specs[binding.server]
        if binding.tool not in specs:
            raise LookupError(
                f"capability '{capability_id}' pins tool '{binding.tool}', but server '{binding.server}' "
                f"offers: {', '.join(sorted(specs)) or 'nothing'}"
            )
        return specs[binding.tool]


def _binding(capability: Capability) -> McpBinding:
    assert capability.mcp is not None, "validated by Capability"
    return capability.mcp
