"""Real speech models end to end without a microphone: the TTS speaks a request, and the
audio goes through the real wake-word spotter, VAD and speech-to-text inside VoiceLoop.

Skipped unless the models are present (poetry run python scripts/fetch_models.py).
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import AsyncIterator, Optional

import numpy as np
import pytest

from assistant.adapters.fakes import MemoryStore
from assistant.core.alerts import AlertScheduler
from assistant.core.dialogue import Reply
from assistant.core.listening import ListeningController, ListeningMode
from assistant.core.voice import VoiceLoop
from assistant.ports.speech import CAPTURE_RATE, Audio

MODELS = Path(__file__).resolve().parents[2] / "models"
pytestmark = pytest.mark.skipif(
    not all((MODELS / d).exists() for d in ("kws", "vad", "asr-moonshine", "tts-piper")),
    reason="speech models not downloaded",
)


@pytest.fixture(scope="module")
def models():  # type: ignore[no-untyped-def]
    from assistant.adapters.speech import sherpa

    return {
        "tts": sherpa.piper(sherpa.PiperSettings(dir=MODELS / "tts-piper", model_file="en_US-lessac-medium.onnx")),
        "stt": sherpa.moonshine(sherpa.MoonshineSettings(dir=MODELS / "asr-moonshine")),
        "make_wake": lambda: sherpa.SherpaWakeWord(sherpa.SherpaWakeWordSettings(dir=MODELS / "kws"), ["hey nova"]),
        "make_vad": lambda: sherpa.SileroVad(sherpa.SileroVadSettings(model=MODELS / "vad" / "silero_vad.onnx")),
    }


class Heard:
    def __init__(self) -> None:
        self.listening = ListeningController(ListeningMode.WAKE_WORD, follow_up_seconds=8)
        self.alerts = AlertScheduler(MemoryStore(), lambda: datetime.now(timezone.utc))
        self.utterances: list[tuple[str, bool]] = []

    async def handle_utterance(self, text: str, *, wake_word_detected: bool) -> Optional[Reply]:
        self.utterances.append((text, wake_word_detected))
        return None


class SilentMic:
    async def frames(self) -> AsyncIterator[np.ndarray]:  # pragma: no cover
        yield np.zeros(1600, dtype=np.float32)

    def set_active(self, active: bool) -> None: ...

    def flush(self) -> None: ...


class NoSpeaker:
    async def play(self, audio: Audio) -> None: ...


def to_capture_rate(audio: Audio) -> np.ndarray:
    duration = len(audio.samples) / audio.sample_rate
    t_out = np.arange(int(duration * CAPTURE_RATE)) / CAPTURE_RATE
    t_in = np.arange(len(audio.samples)) / audio.sample_rate
    speech = np.interp(t_out, t_in, audio.samples).astype(np.float32)
    silence = np.zeros(int(0.5 * CAPTURE_RATE), dtype=np.float32)
    return np.concatenate([silence, speech, silence, silence, silence])


async def run_through_loop(models, spoken: str) -> list[tuple[str, bool]]:  # type: ignore[no-untyped-def]
    heard = Heard()
    loop = VoiceLoop(
        heard, SilentMic(), models["make_wake"](), models["make_vad"](), models["stt"], models["tts"],  # type: ignore[arg-type]
        NoSpeaker(), ["hey nova"],
    )
    samples = to_capture_rate(models["tts"].synthesize(spoken))
    for start in range(0, len(samples), 1600):
        await loop.process(samples[start : start + 1600])
    return heard.utterances


async def test_wake_phrase_then_request_is_transcribed(models) -> None:  # type: ignore[no-untyped-def]
    utterances = await run_through_loop(models, "Hey nova, what's the weather tomorrow?")

    assert len(utterances) == 1
    text, wake = utterances[0]
    assert wake and "weather" in text.lower() and not text.lower().startswith("hey")


async def test_request_without_wake_phrase_is_ignored(models) -> None:  # type: ignore[no-untyped-def]
    assert await run_through_loop(models, "What's the weather tomorrow?") == []
