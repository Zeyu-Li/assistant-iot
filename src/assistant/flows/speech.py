"""Phrasing dates, times, durations and numbers the way people say them."""

from __future__ import annotations

import math
import re
from datetime import date, datetime, time, timedelta
from typing import Optional, Sequence

_ISO_DURATION = re.compile(
    r"^P(?:(?P<days>\d+(?:\.\d+)?)D)?(?:T(?:(?P<hours>\d+(?:\.\d+)?)H)?"
    r"(?:(?P<minutes>\d+(?:\.\d+)?)M)?(?:(?P<seconds>\d+(?:\.\d+)?)S)?)?$"
)


def parse_duration(value: str) -> Optional[int]:
    """Seconds in an ISO 8601 duration such as 'PT1H30M'; None if it is not one."""
    match = _ISO_DURATION.match(value.strip().upper())
    if not match or not any(match.groupdict().values()):
        return None
    parts = {k: float(v) for k, v in match.groupdict().items() if v}
    return round(parts.get("days", 0) * 86400 + parts.get("hours", 0) * 3600 + parts.get("minutes", 0) * 60 + parts.get("seconds", 0))


def plural(count: int, noun: str) -> str:
    return f"{count} {noun}" if count == 1 else f"{count} {noun}s"


def and_list(items: Sequence[str]) -> str:
    """'milk', 'milk and eggs', 'milk, eggs, and bread'."""
    if len(items) <= 2:
        return " and ".join(items)
    return f"{', '.join(items[:-1])}, and {items[-1]}"


def duration_phrase(seconds: int) -> str:
    """'1 hour and 30 minutes', '45 seconds'."""
    hours, rest = divmod(max(0, seconds), 3600)
    minutes, secs = divmod(rest, 60)
    parts = [plural(n, unit) for n, unit in ((hours, "hour"), (minutes, "minute"), (secs, "second")) if n]
    return and_list(parts) if parts else "0 seconds"


def remaining_phrase(seconds: float) -> str:
    """Time left, rounded the way a person would read a kitchen timer."""
    if seconds < 120:
        return duration_phrase(max(1, math.ceil(seconds)))
    return duration_phrase(round(seconds / 60) * 60)


def number_phrase(value: float) -> str:
    """Three significant figures for small values, whole numbers with commas for large ones."""
    if not math.isfinite(value):
        return "an impossibly large number"
    if abs(value) >= 1000:
        return f"{round(value):,}"
    if value == 0:
        return "0"
    rounded = round(value, 2 - int(math.floor(math.log10(abs(value)))))
    text = f"{rounded:f}".rstrip("0").rstrip(".")
    return "0" if text in ("-0", "") else text


def day_phrase(day: date, today: date, *, with_on: bool = True) -> str:
    """'today', 'tomorrow', 'on Friday', 'on December 3' (drop 'on' after words like 'to')."""
    if day == today:
        return "today"
    if day == today + timedelta(days=1):
        return "tomorrow"
    on = "on " if with_on else ""
    if today < day < today + timedelta(days=7):
        return f"{on}{day:%A}"
    return f"{on}{day:%B} {day.day}"


def time_phrase(moment: time) -> str:
    if moment.hour == 12 and moment.minute == 0:
        return "noon"
    if moment.hour == 0 and moment.minute == 0:
        return "midnight"
    hour = moment.hour % 12 or 12
    suffix = "AM" if moment.hour < 12 else "PM"
    return f"{hour} {suffix}" if moment.minute == 0 else f"{hour}:{moment.minute:02d} {suffix}"


def when_phrase(value: str, today: date, *, with_on: bool = True) -> str:
    """'2026-11-27T12:00:00' -> 'on Friday at noon'; a bare date -> just the day."""
    if "T" not in value:
        return day_phrase(date.fromisoformat(value), today, with_on=with_on)
    moment = datetime.fromisoformat(value)
    return f"{day_phrase(moment.date(), today, with_on=with_on)} at {time_phrase(moment.time())}"
