"""Calendar writes as flows over the Google Calendar MCP server.

Creating needs an end time the user rarely says, and updating needs the event's id,
which only a lookup can supply. Both are deterministic steps, so they live here rather
than with the LLM.
"""

from __future__ import annotations

import json
import re
from datetime import date, datetime, timedelta
from typing import Any, Callable, Mapping

from assistant.flows.speech import when_phrase
from assistant.ports.flow import FlowContext
from assistant.ports.tools import ToolClient, ToolResult


class CreateEventFlow:
    def __init__(self, calendar: ToolClient, today: Callable[[], date], default_minutes: int = 60) -> None:
        self._calendar = calendar
        self._today = today
        self._default_minutes = default_minutes

    def describe(self, arguments: Mapping[str, Any]) -> str:
        return f"Add {arguments['summary']} {when_phrase(arguments['startTime'], self._today())}?"

    async def run(self, context: FlowContext) -> str:
        args = context.arguments
        title = str(args["summary"])
        title = title[:1].upper() + title[1:]
        start = datetime.fromisoformat(args["startTime"])
        end = args.get("endTime") or (start + timedelta(minutes=self._default_minutes)).isoformat(timespec="seconds")
        payload: dict[str, Any] = {"summary": title, "startTime": start.isoformat(timespec="seconds"), "endTime": end}
        if args.get("location"):
            payload["location"] = args["location"]
        result = await self._calendar.call_tool("create_event", payload)
        if result.is_error:
            return "Sorry, I couldn't add that to your calendar."
        return f"Done. {title} is on your calendar {when_phrase(payload['startTime'], self._today())}."


class UpdateEventFlow:
    def __init__(self, calendar: ToolClient, today: Callable[[], date]) -> None:
        self._calendar = calendar
        self._today = today

    def _describe(self, arguments: Mapping[str, Any]) -> str:
        event = arguments["event"]
        changes = []
        if arguments.get("newStartTime"):
            changes.append(f"move your {event} event to {when_phrase(arguments['newStartTime'], self._today(), with_on=False)}")
        if arguments.get("newSummary"):
            changes.append(f"rename your {event} event to {arguments['newSummary']}")
        return f"Should I {' and '.join(changes)}?" if changes else "Should I go ahead?"

    def describe(self, arguments: Mapping[str, Any]) -> str:
        return self._describe(_meaningful_changes(arguments))

    async def run(self, context: FlowContext) -> str:
        args = _meaningful_changes(context.arguments)
        event = args["event"]
        if not args.get("newStartTime") and not args.get("newSummary"):
            return f"Tell me the new time or name for your {event} event."

        found = await self._calendar.call_tool("list_events", {"fullText": event, "pageSize": 5, "orderBy": "startTime"})
        if found.is_error:
            return "Sorry, I couldn't read your calendar."
        # Never write to an event the user did not name, whatever the search returned.
        events = [e for e in _events(found) if _words(str(e.get("summary", ""))) & _words(event)]
        if not events:
            return f"I couldn't find an upcoming event matching {event}."
        if len(events) > 1:
            return f"I found {len(events)} upcoming events matching {event}. Can you be more specific?"

        event_id = events[0].get("id") or events[0].get("eventId")
        if not event_id:
            return "Sorry, I couldn't tell which event that is."
        payload: dict[str, Any] = {"eventId": event_id}
        if args.get("newStartTime"):
            payload["startTime"] = args["newStartTime"]  # the server keeps the duration
        if args.get("newSummary"):
            payload["summary"] = args["newSummary"]
        result = await self._calendar.call_tool("update_event", payload)
        return "Sorry, I couldn't change that event." if result.is_error else "Done, I've updated it."


_FILLER = frozenset({"the", "my", "a", "an", "with", "to", "at", "on", "for", "event", "appointment", "meeting"})


def _words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9']+", text.lower()) if w not in _FILLER}


def _meaningful_changes(arguments: Mapping[str, Any]) -> dict[str, Any]:
    """Drop a 'new name' that only restates the event's current name (small models add one unasked)."""
    args = dict(arguments)
    new_summary = args.get("newSummary")
    if new_summary:
        new_words, event_words = _words(str(new_summary)), _words(str(args["event"]))
        if not new_words or new_words <= event_words or event_words <= new_words:
            del args["newSummary"]
    return args


def _events(result: ToolResult) -> list[Mapping[str, Any]]:
    data: Any = result.data
    if data is None:
        try:
            data = json.loads(result.text)
        except (json.JSONDecodeError, TypeError):
            return []
    if not isinstance(data, Mapping):
        return []
    events = data.get("events") or data.get("items") or []
    return [e for e in events if isinstance(e, Mapping)]
