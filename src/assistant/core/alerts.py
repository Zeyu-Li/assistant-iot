"""Timers and alarms: the one place the assistant speaks without being spoken to.

The scheduler keeps alerts in the document store, so alarms survive a restart, and wakes
when the next one is due. Ringing is delegated to a callback (the voice loop chimes and
speaks; the text REPL prints). While an alert rings, any speech or the wake phrase
dismisses it.
"""

from __future__ import annotations

import asyncio
import logging
import re
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from typing import Awaitable, Callable, Literal, Optional

from assistant.ports.store import DocumentStore

log = logging.getLogger(__name__)

AlertKind = Literal["timer", "alarm"]
_DOCUMENT = "alerts"
_MAX_SLEEP_SECONDS = 30.0
"""Re-check at least this often, so a laptop waking from sleep or a clock change is noticed."""
_GENERIC_LABEL_WORDS = frozenset({"timer", "timers", "alarm", "alarms", "the", "my", "a", "an", "for"})


@dataclass(frozen=True)
class Alert:
    id: str
    kind: AlertKind
    due: datetime
    label: str = ""
    duration_seconds: Optional[int] = None
    """Timers only: the length that was asked for, so the alert can say "your 10 minute timer"."""


class AlertScheduler:
    def __init__(
        self,
        store: DocumentStore,
        clock: Callable[[], datetime],
        *,
        stale_after: timedelta = timedelta(minutes=5),
    ) -> None:
        self._store = store
        self._clock = clock
        self._alerts = self._load(stale_after)
        self._changed = asyncio.Event()
        self._dismissed = asyncio.Event()
        self.ringing: Optional[Alert] = None

    # --- managing alerts -----------------------------------------------------------------

    def add(self, kind: AlertKind, due: datetime, label: str = "", duration_seconds: Optional[int] = None) -> Alert:
        alert = Alert(uuid.uuid4().hex[:12], kind, due, label.strip(), duration_seconds)
        self._alerts.append(alert)
        self._save()
        return alert

    def pending(self, kind: Optional[AlertKind] = None) -> list[Alert]:
        return sorted((a for a in self._alerts if kind is None or a.kind == kind), key=lambda a: a.due)

    def matching(self, kind: AlertKind, label: Optional[str]) -> list[Alert]:
        """Alerts of ``kind`` whose label shares a word with ``label``; all of them when no label is given."""
        wanted = _label_words(label or "")
        candidates = self.pending(kind)
        if not wanted:
            return candidates
        return [a for a in candidates if wanted & _label_words(a.label)]

    def cancel(self, alerts: list[Alert]) -> None:
        ids = {a.id for a in alerts}
        self._alerts = [a for a in self._alerts if a.id not in ids]
        self._save()

    # --- ringing -----------------------------------------------------------------------

    def dismiss(self) -> bool:
        """Stop the ringing alert. Returns False when nothing was ringing (or it was already stopped)."""
        if self.ringing is None or self._dismissed.is_set():
            return False
        log.info("%s dismissed", self.ringing.kind)
        self._dismissed.set()
        return True

    async def wait_dismissed(self, timeout: float) -> bool:
        try:
            await asyncio.wait_for(self._dismissed.wait(), timeout)
            return True
        except asyncio.TimeoutError:
            return False

    async def run(self, ring: Callable[[Alert], Awaitable[None]]) -> None:
        """Ring each alert when it is due, forever."""
        while True:
            self._changed.clear()
            for alert in self._pop_due():
                log.info("%s due: %s", alert.kind, alert.label or alert.id)
                self.ringing = alert
                self._dismissed.clear()
                try:
                    await ring(alert)
                except Exception:  # a failed ring must not stop later alarms
                    log.exception("ringing %s failed", alert.id)
                finally:
                    self.ringing = None
            upcoming = self.pending()
            wait = _MAX_SLEEP_SECONDS
            if upcoming:
                wait = min(wait, max(0.0, (upcoming[0].due - self._clock()).total_seconds()))
            try:
                await asyncio.wait_for(self._changed.wait(), wait)
            except asyncio.TimeoutError:
                pass

    def _pop_due(self) -> list[Alert]:
        now = self._clock()
        due = [a for a in self.pending() if a.due <= now]
        if due:
            self.cancel(due)
        return due

    # --- persistence -------------------------------------------------------------------

    def _load(self, stale_after: timedelta) -> list[Alert]:
        now = self._clock()
        alerts = []
        for raw in self._store.load(_DOCUMENT).get("alerts", []):
            alert = Alert(**{**raw, "due": datetime.fromisoformat(raw["due"])})
            if alert.due < now - stale_after:
                log.warning("dropping %s for %s: it was due while the assistant was off", alert.kind, alert.due)
                continue
            alerts.append(alert)
        return alerts

    def _save(self) -> None:
        self._store.save(_DOCUMENT, {"alerts": [{**asdict(a), "due": a.due.isoformat()} for a in self._alerts]})
        self._changed.set()


def _label_words(label: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", label.lower())) - _GENERIC_LABEL_WORDS
