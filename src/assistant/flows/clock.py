"""Time, date, weekday and day-count questions, answered from the system clock."""

from __future__ import annotations

from datetime import date, datetime
from functools import lru_cache
from typing import Any, Callable, Optional
from zoneinfo import ZoneInfo, available_timezones

from assistant.flows.speech import day_phrase, plural, time_phrase
from assistant.ports.flow import FlowContext

# Big places the time-zone database names by a different city.
_PLACE_ZONES = {
    "san francisco": "America/Los_Angeles", "seattle": "America/Los_Angeles", "california": "America/Los_Angeles",
    "boston": "America/New_York", "washington": "America/New_York", "miami": "America/New_York",
    "dallas": "America/Chicago", "houston": "America/Chicago", "austin": "America/Chicago",
    "beijing": "Asia/Shanghai", "mumbai": "Asia/Kolkata", "delhi": "Asia/Kolkata", "india": "Asia/Kolkata",
    "england": "Europe/London", "uk": "Europe/London", "japan": "Asia/Tokyo", "china": "Asia/Shanghai",
    "hawaii": "Pacific/Honolulu", "germany": "Europe/Berlin", "france": "Europe/Paris",
}


@lru_cache(maxsize=1)
def _zones_by_city() -> dict[str, str]:
    """'new york' -> 'America/New_York', from the last part of each zone name."""
    return {zone.rsplit("/", 1)[-1].replace("_", " ").lower(): zone for zone in available_timezones() if "/" in zone}


def zone_for(place: str) -> Optional[ZoneInfo]:
    key = place.strip().lower()
    name = _PLACE_ZONES.get(key) or _zones_by_city().get(key)
    return ZoneInfo(name) if name else None


class ClockFlow:
    def __init__(self, clock: Callable[[], datetime]) -> None:
        self._clock = clock

    async def run(self, context: FlowContext) -> str:
        args = context.arguments
        question = args.get("question", "time")
        place = str(args.get("place") or "").strip()
        now = self._clock()
        if place:
            zone = zone_for(place)
            if zone is None:
                return f"I don't know the time zone for {place}."
            now = now.astimezone(zone)
        where = f" in {place}" if place else ""

        if question == "time":
            return f"It's {time_phrase(now.time())}{where}."
        if question == "date":
            return f"It's {now:%A}, {now:%B} {now.day}{where}."

        day = _parse_date(args.get("date"))
        if day is None:
            return "Sorry, I didn't catch which day you meant."
        today = self._clock().date()
        if question == "weekday":
            verb = "was" if day < today else "is"
            return f"{day:%B} {day.day} {verb} a {day:%A}."
        days = (day - today).days
        if days == 0:
            return "That's today."
        if days == 1:
            return "That's tomorrow."
        if days < 0:
            return f"That was {plural(-days, 'day')} ago."
        return f"It's in {plural(days, 'day')}, {day_phrase(day, today)}."


def _parse_date(value: Any) -> Optional[date]:
    if not isinstance(value, str):
        return None
    try:
        return date.fromisoformat(value[:10])
    except ValueError:
        return None
