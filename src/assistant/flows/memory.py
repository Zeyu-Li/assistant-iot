"""Remembered notes: key-value pairs found again by the closest key above a threshold."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Callable, Mapping

from assistant.flows.matching import best_match, key_words
from assistant.ports.flow import FlowContext
from assistant.ports.store import DocumentStore

_DOCUMENT = "memory"
DEFAULT_MATCH_THRESHOLD = 0.5


class MemoryFlow:
    def __init__(
        self,
        store: DocumentStore,
        clock: Callable[[], datetime],
        match_threshold: float = DEFAULT_MATCH_THRESHOLD,
    ) -> None:
        self._store = store
        self._clock = clock
        self._threshold = match_threshold

    def describe(self, arguments: Mapping[str, Any]) -> str:
        note = self._find(self._notes(), str(arguments.get("key") or ""))
        return f"Forget {note['key']}?" if note else "Forget that?"

    async def run(self, context: FlowContext) -> str:
        key = str(context.arguments.get("key") or "").strip()
        value = str(context.arguments.get("value") or "").strip()
        action = context.arguments.get("action", "recall")
        notes = self._notes()

        if action == "remember":
            if not value:
                return f"Sorry, I didn't catch what to remember about {key}."
            same = [n for n in notes if key_words(n["key"]) == key_words(key)]
            for note in same:
                notes.remove(note)
            notes.append({"key": key, "value": value, "saved": self._clock().isoformat()})
            self._save(notes)
            return f"Okay, I'll remember {key}: {value}."

        note = self._find(notes, key)
        if note is None:
            return f"I don't have anything saved about {key}."
        if action == "forget":
            notes.remove(note)
            self._save(notes)
            return f"Okay, I forgot {note['key']}."
        return f"{note['key'][:1].upper()}{note['key'][1:]}: {note['value']}."

    def _find(self, notes: list[dict[str, str]], key: str) -> dict[str, str] | None:
        match = best_match(key, [n["key"] for n in notes], self._threshold)
        return next((n for n in reversed(notes) if n["key"] == match), None) if match else None

    def _notes(self) -> list[dict[str, str]]:
        return list(self._store.load(_DOCUMENT).get("notes", []))

    def _save(self, notes: list[dict[str, str]]) -> None:
        self._store.save(_DOCUMENT, {"notes": notes})
