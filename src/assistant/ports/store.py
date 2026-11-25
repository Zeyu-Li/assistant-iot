"""Small local documents the assistant keeps between runs: alarms, lists, remembered notes."""

from __future__ import annotations

from typing import Any, Protocol


class DocumentStore(Protocol):
    def load(self, name: str) -> dict[str, Any]:
        """The saved document, or an empty dict if there is none yet."""
        ...

    def save(self, name: str, document: dict[str, Any]) -> None:
        """Replace the document; it survives a crash mid-write."""
        ...
