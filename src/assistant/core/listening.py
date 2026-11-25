"""Listening modes: muted, wake word, open mic."""

from __future__ import annotations

import logging
import time
from enum import Enum
from typing import Callable

log = logging.getLogger(__name__)


class ListeningMode(str, Enum):
    MUTED = "muted"
    """Microphone closed. Only the CLI or the local control endpoint can leave this mode."""
    WAKE_WORD = "wake_word"
    """Only the wake-word spotter runs, plus a short follow-up window after each reply."""
    OPEN_MIC = "open_mic"
    """Every speech segment is transcribed; the assistant answers only when addressed."""


class ListeningController:
    def __init__(
        self,
        mode: ListeningMode,
        follow_up_seconds: float,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._mode = mode
        self._follow_up_seconds = follow_up_seconds
        self._clock = clock
        self._follow_up_until = float("-inf")
        self._listeners: list[Callable[[ListeningMode], None]] = []

    @property
    def mode(self) -> ListeningMode:
        return self._mode

    def set_mode(self, mode: ListeningMode) -> None:
        if mode is self._mode:
            return
        log.info("listening mode %s -> %s", self._mode.value, mode.value)
        self._mode = mode
        self._follow_up_until = float("-inf")
        for listener in self._listeners:
            listener(mode)

    def on_change(self, listener: Callable[[ListeningMode], None]) -> None:
        """Notify ``listener`` on every mode change (the audio loop opens/closes the mic here)."""
        self._listeners.append(listener)

    def open_follow_up(self) -> None:
        self._follow_up_until = self._clock() + self._follow_up_seconds

    def in_follow_up(self) -> bool:
        return self._clock() < self._follow_up_until

    def accepts_speech(self, *, wake_word_detected: bool) -> bool:
        if self._mode is ListeningMode.MUTED:
            return False
        if self._mode is ListeningMode.OPEN_MIC:
            return True
        return wake_word_detected or self.in_follow_up()

    def needs_addressed_check(self, *, wake_word_detected: bool) -> bool:
        """Open-mic speech that was not preceded by the wake word may not be meant for us."""
        return self._mode is ListeningMode.OPEN_MIC and not wake_word_detected and not self.in_follow_up()
