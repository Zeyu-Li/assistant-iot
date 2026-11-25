"""Capability registry: the fixed set of things the assistant can do."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Iterator, Literal, Optional

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

NONE_ID = "none"
"""Reserved route label meaning "no capability fits"."""
RESERVED_IDS = frozenset({NONE_ID, "other"})
"""Labels the router uses for itself ("other" is what the decision model sees for none)."""


class McpBinding(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    server: str
    tool: str


class RenderSpec(BaseModel):
    """How an MCP result becomes speech. Flow handlers phrase their own replies."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    mode: Literal["summary", "fixed"] = "summary"
    instruction: str = ""
    text: str = ""


class Capability(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    kind: Literal["flow", "mcp"]
    description: str
    examples: tuple[str, ...] = ()
    slots: Optional[dict[str, Any]] = None
    """JSON Schema for the arguments. Omit on an mcp capability to use the tool's inputSchema."""
    prompts: dict[str, str] = Field(default_factory=dict)
    defaults: dict[str, Any] = Field(default_factory=dict)
    confirm: bool = False
    confirm_prompt: str = ""
    mcp: Optional[McpBinding] = None
    render: RenderSpec = RenderSpec()

    @model_validator(mode="after")
    def _check_binding(self) -> "Capability":
        if self.id in RESERVED_IDS:
            raise ValueError(f"'{self.id}' is reserved and cannot be a capability id")
        if (self.kind == "mcp") != (self.mcp is not None):
            raise ValueError(f"{self.id}: an 'mcp' binding is required for kind 'mcp' and only allowed there")
        return self

    @property
    def option_text(self) -> str:
        """What the decision model reads when judging whether this capability fits."""
        if not self.examples:
            return self.description
        return f"{self.description} Examples: {'; '.join(self.examples)}"


class Registry:
    def __init__(self, capabilities: Iterable[Capability]) -> None:
        self._by_id: dict[str, Capability] = {}
        for capability in capabilities:
            if capability.id in self._by_id:
                raise ValueError(f"duplicate capability id: {capability.id}")
            self._by_id[capability.id] = capability

    @classmethod
    def load(cls, path: Path, disabled: Iterable[str] = ()) -> "Registry":
        raw = yaml.safe_load(path.read_text())
        capabilities = [Capability.model_validate(entry) for entry in raw["capabilities"]]
        disabled_ids = set(disabled)
        unknown = disabled_ids - {c.id for c in capabilities}
        if unknown:
            raise ValueError(f"cannot disable unknown capabilities: {sorted(unknown)}")
        return cls(c for c in capabilities if c.id not in disabled_ids)

    def get(self, capability_id: str) -> Capability:
        return self._by_id[capability_id]

    def __iter__(self) -> Iterator[Capability]:
        return iter(self._by_id.values())

    def __len__(self) -> int:
        return len(self._by_id)
