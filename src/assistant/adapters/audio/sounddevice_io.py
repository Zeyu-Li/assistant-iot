"""Microphone and speaker on PortAudio (sounddevice): CoreAudio on macOS, ALSA on Linux/Jetson."""

from __future__ import annotations

import asyncio
import logging
from typing import Any, AsyncIterator, Optional, Union

import numpy as np
import sounddevice as sd

from assistant.ports.speech import CAPTURE_RATE, Audio

log = logging.getLogger(__name__)

Device = Optional[Union[int, str]]


class SoundDeviceMicrophone:
    def __init__(self, device: Device = None, frame_ms: int = 100) -> None:
        self._device = device
        self._blocksize = CAPTURE_RATE * frame_ms // 1000
        self._queue: asyncio.Queue[np.ndarray] = asyncio.Queue(maxsize=200)
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._stream: Optional[sd.InputStream] = None
        self._active = False

    async def frames(self) -> AsyncIterator[np.ndarray]:
        self._loop = asyncio.get_running_loop()
        self._apply()
        try:
            while True:
                yield await self._queue.get()
        finally:
            self._close()

    def set_active(self, active: bool) -> None:
        self._active = active
        if self._loop is not None:
            self._apply()

    def flush(self) -> None:
        while not self._queue.empty():
            self._queue.get_nowait()

    def _apply(self) -> None:
        if self._active and self._stream is None:
            self._stream = sd.InputStream(
                samplerate=CAPTURE_RATE,
                channels=1,
                dtype="float32",
                blocksize=self._blocksize,
                device=self._device,
                callback=self._on_audio,
            )
            self._stream.start()
            log.info("microphone open")
        elif not self._active and self._stream is not None:
            self._close()
            self.flush()
            log.info("microphone closed")

    def _close(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    def _on_audio(self, data: np.ndarray, frames: int, time: Any, status: sd.CallbackFlags) -> None:
        if status:
            log.debug("input status: %s", status)
        assert self._loop is not None
        frame = data[:, 0].copy()
        self._loop.call_soon_threadsafe(self._enqueue, frame)

    def _enqueue(self, frame: np.ndarray) -> None:
        if self._queue.full():  # the loop fell behind; old audio is worthless
            self._queue.get_nowait()
        self._queue.put_nowait(frame)


class SoundDeviceSpeaker:
    def __init__(self, device: Device = None) -> None:
        self._device = device

    async def play(self, audio: Audio) -> None:
        await asyncio.to_thread(self._play_blocking, audio)

    def _play_blocking(self, audio: Audio) -> None:
        sd.play(audio.samples, audio.sample_rate, device=self._device)
        sd.wait()
