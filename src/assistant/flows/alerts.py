"""Timers and alarms on top of the AlertScheduler."""

from __future__ import annotations

import re
from datetime import datetime, time, timedelta
from typing import Callable, Optional

from assistant.core.alerts import Alert, AlertScheduler
from assistant.flows.speech import and_list, day_phrase, duration_phrase, parse_duration, remaining_phrase, time_phrase
from assistant.ports.flow import FlowContext

MAX_TIMER_SECONDS = 24 * 3600
# Words that never make a name on their own: small models pass "timer", "5 minutes" or
# "wake up" as the label when the user gave none.
_NOT_A_NAME = frozenset(
    "a an the my for all every any of timer timers alarm alarms set wake waking me up get morning one two three four five "
    "six seven eight nine ten fifteen twenty thirty forty fifty sixty ninety half quarter second seconds "
    "minute minutes hour hours am pm".split()
)


def clean_label(raw: object) -> str:
    """The user's name for an alert, or '' if the "name" is only generic or timing words."""
    label = str(raw or "").strip()
    words = re.findall(r"[a-z]+|\d+", label.lower())
    if not words or all(w in _NOT_A_NAME or w.isdigit() for w in words):
        return ""
    return label


def announcement(alert: Alert, now: datetime) -> str:
    """What the assistant says when an alert goes off."""
    if alert.kind == "timer":
        if alert.label:
            return f"Your {alert.label} timer is done."
        if alert.duration_seconds:
            return f"Your timer for {duration_phrase(alert.duration_seconds)} is done."
        return "Your timer is done."
    sentence = f"It's {time_phrase(now.time())}."
    return f"{sentence} {_capitalised(alert.label)}." if alert.label else f"{sentence} This is your alarm."


class TimerFlow:
    def __init__(self, scheduler: AlertScheduler, clock: Callable[[], datetime]) -> None:
        self._scheduler = scheduler
        self._clock = clock

    async def run(self, context: FlowContext) -> str:
        action = context.arguments.get("action", "set")
        label = clean_label(context.arguments.get("label"))
        if action == "check":
            return self._check(label)
        if action == "cancel":
            return self._cancel(label)
        return self._set(str(context.arguments.get("duration") or ""), label)

    def _set(self, duration: str, label: str) -> str:
        seconds = parse_duration(duration)
        if not seconds:
            return "Sorry, I didn't catch how long the timer should be."
        if seconds > MAX_TIMER_SECONDS:
            return "Timers can be up to 24 hours. For longer, set an alarm."
        self._scheduler.add("timer", self._clock() + timedelta(seconds=seconds), label, seconds)
        length = duration_phrase(seconds)
        return f"Okay, {label} timer for {length}, starting now." if label else f"Okay, {length}, starting now."

    def _check(self, label: str) -> str:
        timers = self._scheduler.matching("timer", label or None)
        if not timers:
            return f"You don't have a {label} timer." if label else "You don't have any timers."
        now = self._clock()
        if len(timers) == 1:
            return _capitalised(f"{_left(timers[0], now)} left.")
        return _capitalised(and_list([f"{t.label or 'one'} has {_left(t, now)}" for t in timers]) + " left.")

    def _cancel(self, label: str) -> str:
        timers = self._scheduler.matching("timer", label or None)
        if not timers:
            return f"You don't have a {label} timer." if label else "You don't have any timers."
        self._scheduler.cancel(timers)
        if len(timers) == 1:
            name = f"the {timers[0].label} timer" if timers[0].label else "your timer"
            return f"Okay, I cancelled {name}."
        return f"Okay, I cancelled {len(timers)} timers."


class AlarmFlow:
    def __init__(self, scheduler: AlertScheduler, clock: Callable[[], datetime]) -> None:
        self._scheduler = scheduler
        self._clock = clock

    async def run(self, context: FlowContext) -> str:
        action = context.arguments.get("action", "set")
        label = clean_label(context.arguments.get("label"))
        at = self._parse_time(context.arguments.get("time"))
        if action == "list":
            return self._list()
        if action == "cancel":
            return self._cancel(label, at)
        if at is None:
            return "Sorry, I didn't catch what time to set the alarm for."
        return self._set(at, label)

    def _parse_time(self, value: object) -> Optional[datetime]:
        if not isinstance(value, str):
            return None
        try:
            moment = datetime.fromisoformat(value)
        except ValueError:
            return None
        # Spoken times are local wall-clock times; ignore any offset rather than converting it.
        return moment.replace(tzinfo=self._clock().tzinfo)

    def _set(self, at: datetime, label: str) -> str:
        now = self._clock()
        if at <= now and at.date() == now.date():
            at += timedelta(days=1)  # "7 AM" said in the evening means tomorrow morning
        if at <= now:
            return "That time has already passed."
        self._scheduler.add("alarm", at, label)
        return f"Okay, your alarm is set for {self._when(at, with_on=False)}."

    def _list(self) -> str:
        alarms = self._scheduler.pending("alarm")
        if not alarms:
            return "You don't have any alarms."
        described = [f"{self._when(a.due)}{', ' + a.label if a.label else ''}" for a in alarms]
        noun = "an alarm" if len(alarms) == 1 else f"{len(alarms)} alarms:"
        return f"You have {noun} {and_list(described)}."

    def _cancel(self, label: str, at: Optional[datetime]) -> str:
        alarms = self._scheduler.matching("alarm", label or None)
        if at is not None:
            alarms = [a for a in alarms if _same_minute(a.due.time(), at.time())]
        if not alarms:
            return "I couldn't find that alarm." if (label or at) else "You don't have any alarms."
        self._scheduler.cancel(alarms)
        if len(alarms) == 1:
            return f"Okay, I cancelled your alarm {self._when(alarms[0].due)}."
        return f"Okay, I cancelled {len(alarms)} alarms."

    def _when(self, moment: datetime, *, with_on: bool = True) -> str:
        return f"{day_phrase(moment.date(), self._clock().date(), with_on=with_on)} at {time_phrase(moment.time())}"


def _left(timer: Alert, now: datetime) -> str:
    seconds = (timer.due - now).total_seconds()
    phrase = remaining_phrase(seconds)
    return f"about {phrase}" if seconds >= 120 else phrase


def _capitalised(text: str) -> str:
    return text[:1].upper() + text[1:]


def _same_minute(a: time, b: time) -> bool:
    return (a.hour, a.minute) == (b.hour, b.minute)
