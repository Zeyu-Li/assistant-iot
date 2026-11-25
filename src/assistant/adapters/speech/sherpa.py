"""Wake word, voice activity detection, speech-to-text and text-to-speech on sherpa-onnx.

One runtime for all four keeps the Jetson build simple (CUDA wheels) while the same code
runs on macOS. Model files come from scripts/fetch_models.py; each settings class names
the files inside its model directory, so swapping a model is a profile edit.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Literal, Sequence

import numpy as np
import sherpa_onnx
from pydantic import BaseModel, ConfigDict

from assistant.ports.speech import CAPTURE_RATE, Audio


class _Settings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dir: Path
    threads: int = 2
    provider: Literal["cpu", "cuda", "coreml"] = "cpu"


# --- wake word -------------------------------------------------------------------------


class SherpaWakeWordSettings(_Settings):
    threshold: float = 0.25
    """Lower fires more readily (more false wakes); raise it if it wakes on its own."""
    score: float = 1.0
    """Boost for the wake phrase's tokens during decoding."""


class SherpaWakeWord:
    """Open-vocabulary keyword spotting: the wake phrase is plain text, no training."""

    def __init__(self, settings: SherpaWakeWordSettings, phrases: Sequence[str]) -> None:
        d = settings.dir
        keyword_lines = [f"{' '.join(_bpe_tokens(d, p))} @{p.replace(' ', '_')}" for p in phrases]
        self._keywords_file = tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False)
        self._keywords_file.write("\n".join(keyword_lines) + "\n")
        self._keywords_file.close()
        self._spotter = sherpa_onnx.KeywordSpotter(
            tokens=str(d / "tokens.txt"),
            encoder=str(d / "encoder-epoch-12-avg-2-chunk-16-left-64.int8.onnx"),
            decoder=str(d / "decoder-epoch-12-avg-2-chunk-16-left-64.int8.onnx"),
            joiner=str(d / "joiner-epoch-12-avg-2-chunk-16-left-64.int8.onnx"),
            keywords_file=self._keywords_file.name,
            num_threads=settings.threads,
            keywords_score=settings.score,
            keywords_threshold=settings.threshold,
            provider=settings.provider,
        )
        self._stream = self._spotter.create_stream()

    def accept(self, frame: np.ndarray) -> bool:
        self._stream.accept_waveform(CAPTURE_RATE, frame)
        heard = False
        while self._spotter.is_ready(self._stream):
            self._spotter.decode_stream(self._stream)
            if self._spotter.get_result(self._stream):
                heard = True
                self._spotter.reset_stream(self._stream)
        return heard

    def reset(self) -> None:
        self._stream = self._spotter.create_stream()


def _bpe_tokens(model_dir: Path, phrase: str) -> list[str]:
    """The model's BPE pieces for a phrase. (sherpa_onnx.text2token does the same but also
    imports a Chinese pinyin library, which an English model never needs.)"""
    import sentencepiece

    pieces = sentencepiece.SentencePieceProcessor(model_file=str(model_dir / "bpe.model")).encode(phrase.upper(), out_type=str)
    vocabulary = {line.split()[0] for line in (model_dir / "tokens.txt").read_text().splitlines() if line.strip()}
    unknown = [p for p in pieces if p not in vocabulary]
    if unknown:
        raise ValueError(f"wake phrase {phrase!r} has pieces the model does not know: {unknown}")
    return pieces


# --- voice activity detection ----------------------------------------------------------


class SileroVadSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: Path
    threshold: float = 0.5
    min_silence_seconds: float = 0.6
    """Pause that ends an utterance. Shorter answers faster but cuts off slow speakers."""
    min_speech_seconds: float = 0.25
    max_speech_seconds: float = 15.0


class SileroVad:
    _WINDOW = 512  # samples per Silero step at 16 kHz

    def __init__(self, settings: SileroVadSettings) -> None:
        config = sherpa_onnx.VadModelConfig(
            silero_vad=sherpa_onnx.SileroVadModelConfig(
                model=str(settings.model),
                threshold=settings.threshold,
                min_silence_duration=settings.min_silence_seconds,
                min_speech_duration=settings.min_speech_seconds,
                window_size=self._WINDOW,
                max_speech_duration=settings.max_speech_seconds,
            ),
            sample_rate=CAPTURE_RATE,
        )
        self._vad = sherpa_onnx.VoiceActivityDetector(config, buffer_size_in_seconds=30)
        self._pending = np.zeros(0, dtype=np.float32)

    def accept(self, frame: np.ndarray) -> list[np.ndarray]:
        self._pending = np.concatenate([self._pending, frame.astype(np.float32, copy=False)])
        while len(self._pending) >= self._WINDOW:
            self._vad.accept_waveform(self._pending[: self._WINDOW])
            self._pending = self._pending[self._WINDOW :]
        segments = []
        while not self._vad.empty():
            segments.append(np.asarray(self._vad.front.samples, dtype=np.float32))
            self._vad.pop()
        return segments

    def reset(self) -> None:
        self._vad.reset()
        self._pending = np.zeros(0, dtype=np.float32)


# --- speech-to-text --------------------------------------------------------------------


class MoonshineSettings(_Settings):
    pass


class ParakeetSettings(_Settings):
    """NVIDIA Parakeet TDT (transducer): larger and more accurate than Moonshine."""


class _OfflineStt:
    def __init__(self, recognizer: sherpa_onnx.OfflineRecognizer) -> None:
        self._recognizer = recognizer

    def transcribe(self, samples: np.ndarray) -> str:
        stream = self._recognizer.create_stream()
        stream.accept_waveform(CAPTURE_RATE, samples)
        self._recognizer.decode_stream(stream)
        return str(stream.result.text).strip()


def moonshine(settings: MoonshineSettings) -> _OfflineStt:
    d = settings.dir
    return _OfflineStt(
        sherpa_onnx.OfflineRecognizer.from_moonshine_v2(
            encoder=str(d / "encoder_model.ort"),
            decoder=str(d / "decoder_model_merged.ort"),
            tokens=str(d / "tokens.txt"),
            num_threads=settings.threads,
            provider=settings.provider,
        )
    )


def parakeet(settings: ParakeetSettings) -> _OfflineStt:
    d = settings.dir
    return _OfflineStt(
        sherpa_onnx.OfflineRecognizer.from_transducer(
            encoder=str(d / "encoder.int8.onnx"),
            decoder=str(d / "decoder.int8.onnx"),
            joiner=str(d / "joiner.int8.onnx"),
            tokens=str(d / "tokens.txt"),
            num_threads=settings.threads,
            provider=settings.provider,
            model_type="nemo_transducer",
        )
    )


# --- text-to-speech --------------------------------------------------------------------


class KokoroSettings(_Settings):
    speaker_id: int = 0
    speed: float = 1.0


class PiperSettings(_Settings):
    model_file: str
    """The .onnx file inside dir, for example en_US-lessac-medium.onnx."""
    speed: float = 1.0


class _OfflineTts:
    def __init__(self, tts: sherpa_onnx.OfflineTts, speaker_id: int, speed: float) -> None:
        self._tts = tts
        self._speaker_id = speaker_id
        self._speed = speed

    def synthesize(self, text: str) -> Audio:
        generated = self._tts.generate(text, sid=self._speaker_id, speed=self._speed)
        return Audio(np.asarray(generated.samples, dtype=np.float32), int(generated.sample_rate))


def kokoro(settings: KokoroSettings) -> _OfflineTts:
    d = settings.dir
    model = sherpa_onnx.OfflineTtsModelConfig(
        kokoro=sherpa_onnx.OfflineTtsKokoroModelConfig(
            model=str(next(d.glob("model*.onnx"))),
            voices=str(d / "voices.bin"),
            tokens=str(d / "tokens.txt"),
            data_dir=str(d / "espeak-ng-data"),
        ),
        num_threads=settings.threads,
        provider=settings.provider,
    )
    return _OfflineTts(sherpa_onnx.OfflineTts(sherpa_onnx.OfflineTtsConfig(model=model)), settings.speaker_id, settings.speed)


def piper(settings: PiperSettings) -> _OfflineTts:
    d = settings.dir
    model = sherpa_onnx.OfflineTtsModelConfig(
        vits=sherpa_onnx.OfflineTtsVitsModelConfig(
            model=str(d / settings.model_file), tokens=str(d / "tokens.txt"), data_dir=str(d / "espeak-ng-data")
        ),
        num_threads=settings.threads,
        provider=settings.provider,
    )
    return _OfflineTts(sherpa_onnx.OfflineTts(sherpa_onnx.OfflineTtsConfig(model=model)), 0, settings.speed)
