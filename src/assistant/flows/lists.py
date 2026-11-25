"""Named lists (shopping, to-do, packing...) kept on the device."""

from __future__ import annotations

import re
from typing import Any, Mapping, Sequence

from assistant.flows.matching import best_match
from assistant.flows.speech import and_list, plural
from assistant.ports.flow import FlowContext
from assistant.ports.store import DocumentStore

_DOCUMENT = "lists"
DEFAULT_LIST = "shopping"
_ITEM_MATCH = 0.75
_ALIASES = {"todo": "to-do", "to do": "to-do", "grocery": "shopping", "groceries": "shopping"}


def list_name(raw: Any) -> str:
    """'My Shopping List' -> 'shopping'; 'todo' -> 'to-do'."""
    name = re.sub(r"\b(my|the|list)\b", " ", str(raw or "").lower())
    name = re.sub(r"\s+", " ", name).strip(" .")
    return _ALIASES.get(name, name) or DEFAULT_LIST


class ListsFlow:
    def __init__(self, store: DocumentStore) -> None:
        self._store = store

    def describe(self, arguments: Mapping[str, Any]) -> str:
        return f"Clear everything on your {list_name(arguments.get('list'))} list?"

    async def run(self, context: FlowContext) -> str:
        args = context.arguments
        name = list_name(args.get("list"))
        items = [str(i).strip() for i in args.get("items") or [] if str(i).strip()]
        document = self._store.load(_DOCUMENT)
        lists: dict[str, list[str]] = document.setdefault("lists", {})
        entries = lists.setdefault(name, [])
        action = args.get("action", "add")

        if action == "read":
            reply = self._read(name, entries)
        elif action == "clear":
            entries.clear()
            reply = f"Okay, your {name} list is empty now."
        elif action == "remove":
            reply = self._remove(name, entries, items)
        else:
            reply = self._add(name, entries, items)

        if not entries:
            lists.pop(name)
        self._store.save(_DOCUMENT, document)
        return reply

    @staticmethod
    def _read(name: str, entries: Sequence[str]) -> str:
        if not entries:
            return f"Your {name} list is empty."
        return f"Your {name} list has {plural(len(entries), 'item')}: {and_list(list(entries))}."

    @staticmethod
    def _add(name: str, entries: list[str], items: Sequence[str]) -> str:
        if not items:
            return f"Sorry, I didn't catch what to add to your {name} list."
        existing = {e.lower() for e in entries}
        added = [i for i in items if i.lower() not in existing]
        entries.extend(added)
        already = [i for i in items if i not in added]
        parts = []
        if added:
            parts.append(f"Added {and_list(added)} to your {name} list.")
        if already:
            verb = "is" if len(already) == 1 else "are"
            parts.append(f"{and_list(already)} {verb} already on it.")
        return " ".join(parts)

    @staticmethod
    def _remove(name: str, entries: list[str], items: Sequence[str]) -> str:
        if not items:
            return f"Sorry, I didn't catch what to take off your {name} list."
        removed, missing = [], []
        for item in items:
            match = best_match(item, entries, _ITEM_MATCH)
            if match is None:
                missing.append(item)
            else:
                entries.remove(match)
                removed.append(match)
        parts = []
        if removed:
            parts.append(f"Removed {and_list(removed)} from your {name} list.")
        if missing:
            parts.append(f"I couldn't find {and_list(missing)} on it.")
        return " ".join(parts)
