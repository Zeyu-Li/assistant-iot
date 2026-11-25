"""Speech and audio ports. Audio is mono float32 in [-1, 1]; capture is 16 kHz.

The model calls are synchronous and CPU/GPU bound; the voice loop runs them in a worker
thread so the event loop keeps reading the microphone.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import AsyncIterator, Protocol

import numpy as np

CAPTURE_RATE = 16_000


@dataclass(frozen=True)
class Audio:
    samples: np.ndarray
    sample_rate: int

    @property
    def seconds(self) -> float:
        return len(self.samples) / self.sample_rate


class WakeWordDetector(Protocol):
    def accept(self, frame: np.ndarray) -> bool:
        """Feed one capture frame; True when the wake phrase has just been heard."""
        ...

    def reset(self) -> None: ...


class SpeechSegmenter(Protocol):
    """Voice activity detection: turns a frame stream into complete utterances."""

    def accept(self, frame: np.ndarray) -> list[np.ndarray]:
        """Feed one capture frame; return any utterances that just ended."""
        ...

    def reset(self) -> None: ...


class SpeechToText(Protocol):
    def transcribe(self, samples: np.ndarray) -> str:
        """Transcribe one utterance captured at CAPTURE_RATE."""
        ...


class TextToSpeech(Protocol):
    def synthesize(self, text: str) -> Audio: ...


class Microphone(Protocol):
    def frames(self) -> AsyncIterator[np.ndarray]:
        """Capture frames at CAPTURE_RATE, forever."""
        ...

    def set_active(self, active: bool) -> None:
        """Open or close the capture stream (closed while muted)."""
        ...

    def flush(self) -> None:
        """Drop frames buffered while the loop was busy (they may hold the assistant's own voice)."""
        ...


class Speaker(Protocol):
    async def play(self, audio: Audio) -> None:
        """Play to the end."""
        ...
