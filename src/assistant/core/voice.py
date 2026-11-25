"""The listening loop: microphone frames in, spoken replies out.

Voice activity detection runs on every frame, but a finished utterance is only
transcribed when the listening mode accepts it: in wake-word mode that means the wake
phrase was heard during it, or the follow-up window is open. So "hey nova, what's the
weather" works in one breath, and so does pausing after the wake phrase.

The loop is half duplex: frames are ignored while the assistant talks, and the backlog is
flushed afterwards so it never transcribes its own voice. Replies and ringing alerts share
one speaking lock, so they never talk over each other.
"""

from __future__ import annotations

import asyncio
import logging
import re
from typing import Optional, Sequence

import numpy as np

from assistant.core.listening import ListeningMode
from assistant.core.pipeline import Assistant
from assistant.ports.speech import (
    Audio,
    Microphone,
    Speaker,
    SpeechSegmenter,
    SpeechToText,
    TextToSpeech,
    WakeWordDetector,
)

log = logging.getLogger(__name__)

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


class VoiceLoop:
    def __init__(
        self,
        assistant: Assistant,
        microphone: Microphone,
        wake: WakeWordDetector,
        segmenter: SpeechSegmenter,
        stt: SpeechToText,
        tts: TextToSpeech,
        speaker: Speaker,
        wake_phrases: Sequence[str],
        earcon: Optional[Audio] = None,
    ) -> None:
        self._assistant = assistant
        self._mic = microphone
        self._wake = wake
        self._segmenter = segmenter
        self._stt = stt
        self._tts = tts
        self._speaker = speaker
        self._wake_words = [phrase.lower().split() for phrase in wake_phrases]
        self._earcon = earcon
        self._chime = chime()
        self._wake_in_segment = False
        self._skip_segment = False
        self._speaking = asyncio.Lock()
        assistant.listening.on_change(self._on_mode_change)

    async def run(self) -> None:
        self._mic.set_active(self._assistant.listening.mode is not ListeningMode.MUTED)
        async for frame in self._mic.frames():
            await self.process(frame)

    async def process(self, frame: np.ndarray) -> None:
        if self._assistant.listening.mode is ListeningMode.MUTED or self._speaking.locked():
            return
        if await asyncio.to_thread(self._wake.accept, frame):
            log.info("wake phrase heard")
            if self._assistant.alerts.dismiss():
                self._skip_segment = True  # "hey nova, stop" - the rest of it is not a request
            else:
                self._wake_in_segment = True
                self._assistant.listening.open_follow_up()
                if self._earcon is not None:
                    async with self._speaking:
                        await self._play(self._earcon)
        for segment in await asyncio.to_thread(self._segmenter.accept, frame):
            await self._handle_segment(segment)

    async def ring(self, text: str, *, repeats: int = 6, gap_seconds: float = 4.0) -> None:
        """Chime and speak ``text`` until dismissed (wake phrase or any speech), at most ``repeats`` times."""
        alerts = self._assistant.alerts
        for _ in range(repeats):
            async with self._speaking:
                await self._play(self._chime)
                for sentence in _sentences(text):
                    await self._play(await asyncio.to_thread(self._tts.synthesize, sentence))
            self._assistant.listening.open_follow_up()  # so "stop" works without the wake phrase
            if await alerts.wait_dismissed(gap_seconds):
                return

    async def _handle_segment(self, segment: np.ndarray) -> None:
        wake_heard, self._wake_in_segment = self._wake_in_segment, False
        skip, self._skip_segment = self._skip_segment, False
        if skip:
            return
        listening = self._assistant.listening
        if not listening.accepts_speech(wake_word_detected=wake_heard):
            return  # never transcribed: nothing said without the wake phrase leaves the device's memory
        text = self.strip_wake_phrase(await asyncio.to_thread(self._stt.transcribe, segment))
        log.info("heard: %r (wake=%s)", text, wake_heard)
        if not text:
            return  # only the wake phrase; the follow-up window is open for the request
        reply = await self._assistant.handle_utterance(text, wake_word_detected=wake_heard)
        if reply is not None:
            await self.say(reply.text)

    async def say(self, text: str) -> None:
        async with self._speaking:
            for sentence in _sentences(text):
                await self._play(await asyncio.to_thread(self._tts.synthesize, sentence))

    async def _play(self, audio: Audio) -> None:
        """Play (callers hold the speaking lock), then drop what the mic heard meanwhile."""
        await self._speaker.play(audio)
        self._mic.flush()
        self._segmenter.reset()
        self._wake.reset()

    def strip_wake_phrase(self, text: str) -> str:
        """Drop a leading wake phrase from the transcript ("Hey nova, what's..." -> "what's...")."""
        words = text.split()
        normalised = [re.sub(r"[^a-z0-9']", "", w.lower()) for w in words]
        for phrase in self._wake_words:
            if normalised[: len(phrase)] == phrase:
                return " ".join(words[len(phrase):]).lstrip(" ,.!?").strip()
        return text.strip()

    def _on_mode_change(self, mode: ListeningMode) -> None:
        self._mic.set_active(mode is not ListeningMode.MUTED)
        self._segmenter.reset()
        self._wake.reset()
        self._wake_in_segment = False
        self._skip_segment = False


def _sentences(text: str) -> list[str]:
    return [s for s in _SENTENCE_END.split(text.strip()) if s]


def chime(sample_rate: int = 24_000) -> Audio:
    """Two rising tones announcing a timer or alarm."""
    parts = [beep(frequency, 0.18, sample_rate).samples for frequency in (660.0, 990.0)]
    gap = np.zeros(int(0.06 * sample_rate), dtype=np.float32)
    return Audio(np.concatenate([parts[0], gap, parts[1], gap]), sample_rate)


def beep(frequency: float = 880.0, seconds: float = 0.12, sample_rate: int = 24_000) -> Audio:
    """A short soft tone acknowledging the wake phrase."""
    t = np.arange(int(seconds * sample_rate)) / sample_rate
    envelope = np.minimum(1.0, np.minimum(t, seconds - t) / 0.02)
    return Audio((0.2 * envelope * np.sin(2 * np.pi * frequency * t)).astype(np.float32), sample_rate)
