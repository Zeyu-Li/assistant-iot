"""Alert scheduler, timer and alarm flows."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

from assistant.adapters.fakes import MemoryStore
from assistant.core.alerts import Alert, AlertScheduler
from assistant.flows.alerts import AlarmFlow, TimerFlow, announcement, clean_label
from assistant.ports.flow import FlowContext

START = datetime(2026, 11, 25, 18, 0, tzinfo=timezone.utc)  # Wednesday 6 PM


class Clock:
    def __init__(self, now: datetime = START) -> None:
        self.now = now

    def __call__(self) -> datetime:
        return self.now


def run(flow: TimerFlow | AlarmFlow, **arguments: object) -> str:
    return asyncio.run(flow.run(FlowContext("", arguments)))


# --- scheduler ------------------------------------------------------------------------


def test_alerts_persist_and_stale_ones_are_dropped_on_load() -> None:
    store, clock = MemoryStore(), Clock()
    scheduler = AlertScheduler(store, clock)
    scheduler.add("alarm", START + timedelta(hours=12), "gym")
    scheduler.add("timer", START + timedelta(minutes=1), duration_seconds=60)

    clock.now = START + timedelta(minutes=10)  # the timer went off 9 minutes ago while we were "off"
    reloaded = AlertScheduler(store, clock)

    assert [(a.kind, a.label) for a in reloaded.pending()] == [("alarm", "gym")]


def test_matching_by_label_word_or_all() -> None:
    scheduler = AlertScheduler(MemoryStore(), Clock())
    pasta = scheduler.add("timer", START + timedelta(minutes=5), "pasta")
    eggs = scheduler.add("timer", START + timedelta(minutes=9), "boiled eggs")

    assert scheduler.matching("timer", "the pasta timer") == [pasta]
    assert scheduler.matching("timer", "eggs") == [eggs]
    assert scheduler.matching("timer", None) == [pasta, eggs]
    assert scheduler.matching("alarm", None) == []


async def test_run_rings_due_alerts_and_dismiss_stops_the_ring() -> None:
    clock = Clock()
    scheduler = AlertScheduler(MemoryStore(), clock)
    rung: list[Alert] = []
    dismissed: list[bool] = []

    async def ring(alert: Alert) -> None:
        rung.append(alert)
        dismissed.append(await scheduler.wait_dismissed(2.0))

    scheduler.add("timer", START - timedelta(seconds=1), "tea")
    task = asyncio.create_task(scheduler.run(ring))
    while scheduler.ringing is None:
        await asyncio.sleep(0.01)
    assert scheduler.dismiss() and not scheduler.dismiss()  # a second dismiss is a no-op
    while scheduler.ringing is not None:
        await asyncio.sleep(0.01)
    task.cancel()

    assert [a.label for a in rung] == ["tea"] and dismissed == [True]
    assert scheduler.pending() == []


async def test_adding_an_alert_wakes_a_sleeping_scheduler() -> None:
    clock = Clock()
    scheduler = AlertScheduler(MemoryStore(), clock)
    rung: list[str] = []

    async def ring(alert: Alert) -> None:
        rung.append(alert.label)

    task = asyncio.create_task(scheduler.run(ring))
    await asyncio.sleep(0.05)  # now sleeping with nothing pending (up to 30 s)
    scheduler.add("timer", START, "now")
    for _ in range(100):
        if rung:
            break
        await asyncio.sleep(0.01)
    task.cancel()

    assert rung == ["now"]


# --- timers ---------------------------------------------------------------------------


def test_timer_set_check_cancel() -> None:
    clock = Clock()
    scheduler = AlertScheduler(MemoryStore(), clock)
    timers = TimerFlow(scheduler, clock)

    assert run(timers, action="set", duration="PT10M") == "Okay, 10 minutes, starting now."
    assert run(timers, action="set", duration="PT1H30M", label="pasta") == "Okay, pasta timer for 1 hour and 30 minutes, starting now."
    clock.now += timedelta(minutes=3, seconds=20)
    assert run(timers, action="check") == "One has about 7 minutes and pasta has about 1 hour and 27 minutes left."
    assert run(timers, action="check", label="pasta") == "About 1 hour and 27 minutes left."
    assert run(timers, action="cancel", label="pasta") == "Okay, I cancelled the pasta timer."
    clock.now += timedelta(minutes=6, seconds=15)
    assert run(timers, action="check") == "25 seconds left."
    assert run(timers, action="cancel", label="all") == "Okay, I cancelled your timer."
    assert run(timers, action="check") == "You don't have any timers."


def test_timer_rejects_missing_and_huge_durations() -> None:
    timers = TimerFlow(AlertScheduler(MemoryStore(), Clock()), Clock())
    assert run(timers, action="set", duration="ten minutes").startswith("Sorry, I didn't catch")
    assert run(timers, action="set", duration="PT25H").startswith("Timers can be up to 24 hours")


def test_generic_words_are_not_names() -> None:
    assert clean_label("timer") == "" and clean_label("5 minutes") == "" and clean_label("wake me up") == ""
    assert clean_label("all") == "" and clean_label("pasta") == "pasta"
    assert clean_label("take my medicine") == "take my medicine"


# --- alarms ---------------------------------------------------------------------------


def test_alarm_time_already_passed_today_means_tomorrow() -> None:
    clock = Clock()
    alarms = AlarmFlow(AlertScheduler(MemoryStore(), clock), clock)

    assert run(alarms, action="set", time="2026-11-25T07:00:00") == "Okay, your alarm is set for tomorrow at 7 AM."
    assert run(alarms, action="set", time="2026-11-24T07:00:00") == "That time has already passed."


def test_alarm_ignores_offsets_the_model_adds() -> None:
    clock = Clock(datetime(2026, 11, 25, 18, 0, tzinfo=timezone(timedelta(hours=-5))))
    scheduler = AlertScheduler(MemoryStore(), clock)

    run(AlarmFlow(scheduler, clock), action="set", time="2026-11-26T07:00:00Z")

    assert scheduler.pending()[0].due.hour == 7


def test_alarm_list_and_cancel_by_time() -> None:
    clock = Clock()
    scheduler = AlertScheduler(MemoryStore(), clock)
    alarms = AlarmFlow(scheduler, clock)
    run(alarms, action="set", time="2026-11-26T07:00:00")
    run(alarms, action="set", time="2026-11-27T06:30:00", label="take the medicine")

    assert run(alarms, action="list") == "You have 2 alarms: tomorrow at 7 AM and on Friday at 6:30 AM, take the medicine."
    assert run(alarms, action="cancel", time="2026-11-26T09:00:00") == "I couldn't find that alarm."
    assert run(alarms, action="cancel", time="2026-11-26T07:00:00") == "Okay, I cancelled your alarm tomorrow at 7 AM."
    assert [a.label for a in scheduler.pending()] == ["take the medicine"]


def test_announcements() -> None:
    timer = Alert("1", "timer", START, "", 600)
    assert announcement(timer, START) == "Your timer for 10 minutes is done."
    assert announcement(Alert("2", "timer", START, "pasta"), START) == "Your pasta timer is done."
    assert announcement(Alert("3", "alarm", START, "take the medicine"), START) == "It's 6 PM. Take the medicine."
    assert announcement(Alert("4", "alarm", START), START) == "It's 6 PM. This is your alarm."
