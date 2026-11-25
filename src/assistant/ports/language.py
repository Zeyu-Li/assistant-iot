"""Small generative model port, used only to fill arguments and phrase answers."""

from __future__ import annotations

from typing import Any, Mapping, Protocol, runtime_checkable


class LanguageModel(Protocol):
    async def generate_json(self, system: str, user: str, schema: Mapping[str, Any]) -> dict[str, Any]:
        """Return an object that conforms to ``schema`` (the adapter constrains decoding)."""
        ...

    async def generate_text(self, system: str, user: str) -> str: ...


@runtime_checkable
class Preloadable(Protocol):
    """A model that can be loaded at startup, so the first request is not slowed by the load."""

    async def preload(self) -> None: ...
