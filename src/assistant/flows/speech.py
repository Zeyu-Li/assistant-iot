"""Phrasing dates and times the way people say them."""

from __future__ import annotations

from datetime import date, datetime, time, timedelta


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
