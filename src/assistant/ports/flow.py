"""Flow handler contract."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol, runtime_checkable


@dataclass(frozen=True)
class FlowContext:
    request: str
    """What the user originally said."""
    arguments: Mapping[str, Any]
    """Validated arguments; every required slot is present."""


class FlowHandler(Protocol):
    async def run(self, context: FlowContext) -> str:
        """Do the work and return the sentence to speak."""
        ...


@runtime_checkable
class DescribesAction(Protocol):
    """Optional for flows: phrase the "should I go ahead?" question for a confirmed capability."""

    def describe(self, arguments: Mapping[str, Any]) -> str: ...
