"""VoiceLoop logic with scripted wake/VAD/STT and a recording assistant."""

from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import AsyncIterator, Optional

import numpy as np

from assistant.adapters.control.unix_socket import ControlServer
from assistant.adapters.fakes import MemoryStore
from assistant.core.alerts import Alert, AlertScheduler
from assistant.core.dialogue import Reply
from assistant.core.listening import ListeningController, ListeningMode
from assistant.core.voice import VoiceLoop
from assistant.ports.speech import Audio

FRAME = np.zeros(1600, dtype=np.float32)


class Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


class RecordingAssistant:
    """Stands in for core.pipeline.Assistant: same listening gate, canned replies."""

    def __init__(self, listening: ListeningController) -> None:
        self.listening = listening
        self.alerts = AlertScheduler(MemoryStore(), lambda: datetime(2026, 11, 25, 9, 0, tzinfo=timezone.utc))
        self.heard: list[tuple[str, bool]] = []

    async def handle_utterance(self, text: str, *, wake_word_detected: bool) -> Optional[Reply]:
        self.heard.append((text, wake_word_detected))
        return Reply(f"You said {text}. Done.")


class Scripted:
    """Wake/VAD/STT whose answers are queued per frame (wake, segments) or per call (text)."""

    def __init__(self, wakes: list[bool], segments: list[int], texts: list[str]) -> None:
        self.wakes, self.segments, self.texts = deque(wakes), deque(segments), deque(texts)
        self.resets = 0

    def accept_wake(self, frame: np.ndarray) -> bool:
        return self.wakes.popleft() if self.wakes else False

    def accept_vad(self, frame: np.ndarray) -> list[np.ndarray]:
        n = self.segments.popleft() if self.segments else 0
        return [np.ones(8000, dtype=np.float32)] * n

    def transcribe(self, samples: np.ndarray) -> str:
        return self.texts.popleft()

    def reset(self) -> None:
        self.resets += 1


class Wake:
    def __init__(self, s: Scripted) -> None:
        self.accept, self.reset = s.accept_wake, s.reset


class Vad:
    def __init__(self, s: Scripted) -> None:
        self.accept, self.reset = s.accept_vad, s.reset


class Stt:
    def __init__(self, s: Scripted) -> None:
        self.transcribe = s.transcribe


class Tts:
    def __init__(self) -> None:
        self.spoken: list[str] = []

    def synthesize(self, text: str) -> Audio:
        self.spoken.append(text)
        return Audio(np.zeros(10, dtype=np.float32), 24000)


class Mic:
    def __init__(self) -> None:
        self.active: Optional[bool] = None
        self.flushes = 0

    async def frames(self) -> AsyncIterator[np.ndarray]:  # pragma: no cover - not used by these tests
        while True:
            yield FRAME

    def set_active(self, active: bool) -> None:
        self.active = active

    def flush(self) -> None:
        self.flushes += 1


class Speaker:
    def __init__(self) -> None:
        self.played = 0

    async def play(self, audio: Audio) -> None:
        self.played += 1


@dataclass
class Rig:
    loop: VoiceLoop
    assistant: RecordingAssistant
    script: Scripted
    tts: Tts
    mic: Mic
    clock: Clock


def make(mode: ListeningMode, wakes: list[bool], segments: list[int], texts: list[str]) -> Rig:
    clock = Clock()
    assistant = RecordingAssistant(ListeningController(mode, follow_up_seconds=8, clock=clock))
    script, tts, mic = Scripted(wakes, segments, texts), Tts(), Mic()
    # RecordingAssistant and the scripted parts satisfy the ports structurally.
    loop = VoiceLoop(assistant, mic, Wake(script), Vad(script), Stt(script), tts, Speaker(), ["hey computer"])  # type: ignore[arg-type]
    return Rig(loop, assistant, script, tts, mic, clock)


async def test_wake_phrase_and_request_in_one_breath() -> None:
    rig = make(ListeningMode.WAKE_WORD, [True], [0, 1], ["Hey computer, what's the weather?"])

    await rig.loop.process(FRAME)  # wake heard
    await rig.loop.process(FRAME)  # utterance ends

    assert rig.assistant.heard == [("what's the weather?", True)]
    assert rig.tts.spoken == ["You said what's the weather?.", "Done."]  # spoken sentence by sentence
    assert rig.mic.flushes == 2  # backlog dropped after each played sentence


async def test_speech_without_wake_phrase_is_never_transcribed() -> None:
    rig = make(ListeningMode.WAKE_WORD, [], [1], ["should not be read"])

    await rig.loop.process(FRAME)

    assert rig.assistant.heard == [] and list(rig.script.texts) == ["should not be read"]


async def test_pause_after_wake_phrase_uses_follow_up_window() -> None:
    rig = make(ListeningMode.WAKE_WORD, [True, False], [1, 0, 1], ["hey computer", "play some jazz"])

    await rig.loop.process(FRAME)  # "hey computer" alone: nothing to handle, window opens
    rig.clock.now += 3
    await rig.loop.process(FRAME)
    await rig.loop.process(FRAME)  # the request, no wake phrase this time

    assert rig.assistant.heard == [("play some jazz", False)]


async def test_muting_closes_mic_and_ignores_frames() -> None:
    rig = make(ListeningMode.WAKE_WORD, [True], [1], ["hey computer stop"])

    rig.assistant.listening.set_mode(ListeningMode.MUTED)
    await rig.loop.process(FRAME)

    assert rig.mic.active is False and rig.assistant.heard == []
    rig.assistant.listening.set_mode(ListeningMode.OPEN_MIC)
    assert rig.mic.active is True


async def test_open_mic_passes_overheard_speech_on() -> None:
    rig = make(ListeningMode.OPEN_MIC, [], [1], ["what's the weather"])
    await rig.loop.process(FRAME)
    assert rig.assistant.heard == [("what's the weather", False)]


def test_strip_wake_phrase_variants() -> None:
    loop = make(ListeningMode.WAKE_WORD, [], [], []).loop
    assert loop.strip_wake_phrase("Hey, computer. Pause the music") == "Pause the music"
    assert loop.strip_wake_phrase("hey computer") == ""
    assert loop.strip_wake_phrase("what's the weather") == "what's the weather"


def test_control_socket_protocol() -> None:
    listening = ListeningController(ListeningMode.MUTED, follow_up_seconds=8)
    control = ControlServer(Path("/nonexistent/control.sock"), listening)
    assert control.respond("mode") == "ok muted"
    assert control.respond("mode wake_word") == "ok wake_word" and listening.mode is ListeningMode.WAKE_WORD
    assert control.respond("mode loud").startswith("error")
    assert control.respond("rm -rf /").startswith("error")


def ringing(rig: Rig) -> None:
    rig.assistant.alerts.ringing = Alert("t1", "timer", datetime(2026, 11, 25, 9, 0, tzinfo=timezone.utc))


async def test_ring_repeats_until_its_limit_when_nobody_answers() -> None:
    rig = make(ListeningMode.WAKE_WORD, [], [], [])
    ringing(rig)

    await rig.loop.ring("Your timer is done.", repeats=3, gap_seconds=0.01)

    assert rig.tts.spoken == ["Your timer is done."] * 3
    assert rig.assistant.listening.in_follow_up()  # "stop" works without the wake phrase


async def test_wake_phrase_dismisses_a_ring_and_the_rest_of_that_sentence_is_dropped() -> None:
    rig = make(ListeningMode.WAKE_WORD, [True, False], [0, 1], ["hey computer stop"])
    ringing(rig)
    ring = asyncio.create_task(rig.loop.ring("Your timer is done.", repeats=5, gap_seconds=5))
    await asyncio.sleep(0.01)  # first ring played; now waiting to be dismissed

    await rig.loop.process(FRAME)  # wake phrase -> dismiss
    await rig.loop.process(FRAME)  # the segment "hey computer stop" ends
    await asyncio.wait_for(ring, 1)

    assert rig.tts.spoken == ["Your timer is done."]
    assert rig.assistant.heard == []  # neither transcribed as a request nor answered


async def test_frames_are_ignored_while_speaking() -> None:
    rig = make(ListeningMode.WAKE_WORD, [True], [1], ["hey computer what time is it"])
    async with rig.loop._speaking:  # e.g. an alarm is mid-sentence
        await rig.loop.process(FRAME)
    assert rig.script.wakes  # the wake detector never saw the frame
