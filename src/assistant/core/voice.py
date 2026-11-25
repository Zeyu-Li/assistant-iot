"""The listening loop: microphone frames in, spoken replies out.

Voice activity detection runs on every frame, but a finished utterance is only
transcribed when the listening mode accepts it: in wake-word mode that means the wake
phrase was heard during it, or the follow-up window is open. So "hey computer, what's the
weather" works in one breath, and so does pausing after the wake phrase.

The loop is half duplex: frames are ignored while the assistant talks, and the backlog is
flushed afterwards so it never transcribes its own voice.
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
        self._wake_in_segment = False
        assistant.listening.on_change(self._on_mode_change)

    async def run(self) -> None:
        self._mic.set_active(self._assistant.listening.mode is not ListeningMode.MUTED)
        async for frame in self._mic.frames():
            await self.process(frame)

    async def process(self, frame: np.ndarray) -> None:
        if self._assistant.listening.mode is ListeningMode.MUTED:
            return
        if await asyncio.to_thread(self._wake.accept, frame):
            log.info("wake phrase heard")
            self._wake_in_segment = True
            self._assistant.listening.open_follow_up()
            if self._earcon is not None:
                await self._play(self._earcon)
        for segment in await asyncio.to_thread(self._segmenter.accept, frame):
            await self._handle_segment(segment)

    async def _handle_segment(self, segment: np.ndarray) -> None:
        wake_heard, self._wake_in_segment = self._wake_in_segment, False
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
        for sentence in (s for s in _SENTENCE_END.split(text.strip()) if s):
            await self._play(await asyncio.to_thread(self._tts.synthesize, sentence))

    async def _play(self, audio: Audio) -> None:
        await self._speaker.play(audio)
        self._mic.flush()
        self._segmenter.reset()
        self._wake.reset()

    def strip_wake_phrase(self, text: str) -> str:
        """Drop a leading wake phrase from the transcript ("Hey computer, what's..." -> "what's...")."""
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


def beep(frequency: float = 880.0, seconds: float = 0.12, sample_rate: int = 24_000) -> Audio:
    """A short soft tone acknowledging the wake phrase."""
    t = np.arange(int(seconds * sample_rate)) / sample_rate
    envelope = np.minimum(1.0, np.minimum(t, seconds - t) / 0.02)
    return Audio((0.2 * envelope * np.sin(2 * np.pi * frequency * t)).astype(np.float32), sample_rate)
