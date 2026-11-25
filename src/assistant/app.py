"""Composition root: turn a profile into a wired-up Assistant.

Adding an adapter = one factory function plus one entry in the matching table below
(MCP servers are built by ``build_tool_clients``).
"""

from __future__ import annotations

import os
from datetime import date, datetime
from pathlib import Path
from typing import Callable, Mapping, Optional, TypeVar
from zoneinfo import ZoneInfo

import httpx

from assistant.adapters import fakes
from assistant.adapters.decision.systemone_http import SystemOneDecisionModel, SystemOneSettings
from assistant.adapters.google.oauth import GoogleTokenProvider, make_store
from assistant.adapters.llm.ollama import OllamaLanguageModel, OllamaSettings
from assistant.adapters.llm.openai_compat import OpenAICompatLanguageModel, OpenAICompatSettings
from assistant.adapters.mcp.client import McpHttpSettings, McpHttpToolClient
from assistant.adapters.music.brave_youtube import BraveYouTubePlayer, BraveYouTubeSettings
from assistant.adapters.net.allowlist import HttpClientFactory
from assistant.adapters.store.json_file import JsonFileStore
from assistant.adapters.weather.open_meteo import OpenMeteoWeatherProvider
from assistant.adapters.control.unix_socket import ControlServer
from assistant.config import AdapterSettings, Profile
from assistant.core.alerts import AlertScheduler
from assistant.core.dialogue import DialogueManager
from assistant.core.executor import CapabilityExecutor
from assistant.core.listening import ListeningController, ListeningMode
from assistant.core.pipeline import Assistant
from assistant.core.registry import Registry
from assistant.core.render import ResultRenderer
from assistant.core.router import Router, RouterThresholds
from assistant.core.slots import SlotExtractor
from assistant.core.voice import VoiceLoop, beep
from assistant.flows.alerts import AlarmFlow, TimerFlow
from assistant.flows.calculator import CalculatorFlow
from assistant.flows.calendar import CreateEventFlow, UpdateEventFlow
from assistant.flows.chat import ChatFlow
from assistant.flows.clock import ClockFlow
from assistant.flows.convert import ConvertFlow
from assistant.flows.listening_mode import ListeningModeFlow
from assistant.flows.lists import ListsFlow
from assistant.flows.memory import MemoryFlow
from assistant.flows.music import MusicControlFlow, PlayMusicFlow
from assistant.flows.weather import WeatherFlow
from assistant.ports.decision import DecisionModel
from assistant.ports.flow import FlowHandler
from assistant.ports.language import LanguageModel, Preloadable
from assistant.ports.music import MusicPlayer
from assistant.ports.speech import SpeechToText, TextToSpeech
from assistant.ports.store import DocumentStore
from assistant.ports.tools import ToolClient, ToolSpec
from assistant.ports.weather import WeatherProvider

FAILURE_TEXT = "Sorry, something went wrong with that."

T = TypeVar("T")
Factory = Callable[[AdapterSettings, Profile], T]


def _http(profile: Profile, *, base_url: str, timeout: float, api_key_env: Optional[str]) -> httpx.AsyncClient:
    headers = {}
    if api_key_env:
        key = os.environ.get(api_key_env)
        if not key:
            raise ValueError(f"environment variable {api_key_env} is not set")
        headers["Authorization"] = f"Bearer {key}"
    return HttpClientFactory(profile.network.allowed_hosts).create(base_url=base_url, timeout=timeout, headers=headers)


def _systemone(settings: AdapterSettings, profile: Profile) -> DecisionModel:
    s = SystemOneSettings.model_validate(settings.options)
    client = _http(profile, base_url=s.base_url, timeout=s.timeout_seconds, api_key_env=s.api_key_env)
    return SystemOneDecisionModel(client, s.model, s.yes_no_as)


def _openai_compat(settings: AdapterSettings, profile: Profile) -> LanguageModel:
    s = OpenAICompatSettings.model_validate(settings.options)
    client = _http(profile, base_url=s.base_url, timeout=s.timeout_seconds, api_key_env=s.api_key_env)
    return OpenAICompatLanguageModel(client, s)


def _ollama(settings: AdapterSettings, profile: Profile) -> LanguageModel:
    s = OllamaSettings.model_validate(settings.options)
    client = _http(profile, base_url=s.base_url, timeout=s.timeout_seconds, api_key_env=s.api_key_env)
    return OllamaLanguageModel(client, s)


def _open_meteo(settings: AdapterSettings, profile: Profile) -> WeatherProvider:
    timeout = float(settings.options.get("timeout_seconds", 10.0))
    home = profile.assistant
    pinned = {home.home_location: home.home_coordinates} if home.home_coordinates else {}
    client = _http(profile, base_url="", timeout=timeout, api_key_env=None)
    return OpenMeteoWeatherProvider(client, home.units, pinned)


def _brave_youtube(settings: AdapterSettings, _: Profile) -> MusicPlayer:
    return BraveYouTubePlayer(BraveYouTubeSettings.model_validate(settings.options))


def _fake_tool_client(settings: AdapterSettings, _: Profile) -> ToolClient:
    tools = settings.options.get("tools", {})
    specs = [ToolSpec(name, spec.get("description", ""), spec.get("input_schema", {})) for name, spec in tools.items()]
    return fakes.FakeToolClient(specs, {name: spec.get("response", f"{name} ok") for name, spec in tools.items()})


DECISION_ADAPTERS: dict[str, Factory[DecisionModel]] = {
    "fake": lambda s, p: fakes.KeywordDecisionModel(),
    "systemone_http": _systemone,
}
LLM_ADAPTERS: dict[str, Factory[LanguageModel]] = {
    "fake": lambda s, p: fakes.ScriptedLanguageModel(),
    "openai_compat": _openai_compat,
    "ollama": _ollama,
}
WEATHER_ADAPTERS: dict[str, Factory[WeatherProvider]] = {
    "fake": lambda s, p: fakes.FakeWeatherProvider(),
    "open_meteo": _open_meteo,
}
MUSIC_ADAPTERS: dict[str, Factory[MusicPlayer]] = {
    "fake": lambda s, p: fakes.FakeMusicPlayer(),
    "brave_youtube": _brave_youtube,
}
STORAGE_ADAPTERS: dict[str, Factory[DocumentStore]] = {
    "memory": lambda s, p: fakes.MemoryStore(),
    "json_file": lambda s, p: JsonFileStore(Path(s.options["dir"])),
}


def build(table: Mapping[str, Factory[T]], settings: AdapterSettings, profile: Profile, port: str) -> T:
    factory = table.get(settings.adapter)
    if factory is None:
        raise ValueError(f"unknown {port} adapter '{settings.adapter}'; available: {', '.join(sorted(table))}")
    return factory(settings, profile)


def build_tool_clients(profile: Profile) -> dict[str, ToolClient]:
    """One client per MCP server; servers with ``auth: google`` share one token provider."""
    http = HttpClientFactory(profile.network.allowed_hosts)
    google_tokens: Optional[GoogleTokenProvider] = None
    clients: dict[str, ToolClient] = {}
    for name, settings in profile.mcp_servers.items():
        if settings.adapter == "fake":
            clients[name] = _fake_tool_client(settings, profile)
        elif settings.adapter == "mcp_http":
            s = McpHttpSettings.model_validate(settings.options)
            token_provider: Optional[GoogleTokenProvider] = None
            if s.auth == "google":
                if google_tokens is None:
                    google_tokens = GoogleTokenProvider(make_store(profile.google_oauth), http.create(timeout=15))
                token_provider = google_tokens
            elif s.auth is not None:
                raise ValueError(f"mcp server '{name}': unknown auth '{s.auth}'")
            clients[name] = McpHttpToolClient(s.url, http, token_provider, s.timeout_seconds)
        else:
            raise ValueError(f"unknown adapter '{settings.adapter}' for mcp server '{name}'; available: fake, mcp_http")
    return clients


def build_tts(settings: AdapterSettings) -> TextToSpeech:
    # sherpa-onnx is imported lazily: the text-only REPL and tests need none of it.
    from assistant.adapters.speech import sherpa

    tables: dict[str, Callable[[AdapterSettings], TextToSpeech]] = {
        "kokoro": lambda s: sherpa.kokoro(sherpa.KokoroSettings.model_validate(s.options)),
        "piper": lambda s: sherpa.piper(sherpa.PiperSettings.model_validate(s.options)),
    }
    if settings.adapter not in tables:
        raise ValueError(f"unknown tts adapter '{settings.adapter}'; available: {', '.join(tables)}")
    return tables[settings.adapter](settings)


def build_voice_loop(profile: Profile, assistant: Assistant) -> tuple[VoiceLoop, ControlServer]:
    from assistant.adapters.audio.sounddevice_io import SoundDeviceMicrophone, SoundDeviceSpeaker
    from assistant.adapters.speech import sherpa

    voice = profile.voice
    if voice is None:
        raise ValueError("this profile has no 'voice' section")
    if voice.wake.adapter != "sherpa_kws" or voice.vad.adapter != "silero":
        raise ValueError("supported: wake adapter 'sherpa_kws', vad adapter 'silero'")
    stt_tables: dict[str, Callable[[AdapterSettings], SpeechToText]] = {
        "moonshine": lambda s: sherpa.moonshine(sherpa.MoonshineSettings.model_validate(s.options)),
        "parakeet": lambda s: sherpa.parakeet(sherpa.ParakeetSettings.model_validate(s.options)),
    }
    if voice.stt.adapter not in stt_tables:
        raise ValueError(f"unknown stt adapter '{voice.stt.adapter}'; available: {', '.join(stt_tables)}")

    loop = VoiceLoop(
        assistant,
        SoundDeviceMicrophone(voice.input_device),
        sherpa.SherpaWakeWord(sherpa.SherpaWakeWordSettings.model_validate(voice.wake.options), voice.wake_phrases),
        sherpa.SileroVad(sherpa.SileroVadSettings.model_validate(voice.vad.options)),
        stt_tables[voice.stt.adapter](voice.stt),
        build_tts(voice.tts),
        SoundDeviceSpeaker(voice.output_device),
        voice.wake_phrases,
        earcon=beep() if voice.earcon else None,
    )
    return loop, ControlServer(voice.control_socket, assistant.listening)


def make_clock(timezone: Optional[str]) -> Callable[[], datetime]:
    zone = ZoneInfo(timezone) if timezone else None
    return lambda: datetime.now(zone).astimezone(zone)


def build_router(profile: Profile, registry: Registry, decider: DecisionModel) -> Router:
    thresholds = RouterThresholds(route=profile.router.route_threshold, addressed=profile.router.addressed_threshold)
    return Router(decider, registry, thresholds, include_examples=profile.router.include_examples)


async def build_assistant(profile: Profile, capabilities_path: Path) -> Assistant:
    registry = Registry.load(capabilities_path, disabled=profile.disabled_capabilities)
    clock = make_clock(profile.assistant.timezone)
    today: Callable[[], date] = lambda: clock().date()

    decider = build(DECISION_ADAPTERS, profile.decision, profile, "decision")
    llm = build(LLM_ADAPTERS, profile.llm, profile, "llm")
    if isinstance(llm, Preloadable):
        await llm.preload()
    weather = build(WEATHER_ADAPTERS, profile.weather, profile, "weather")
    music = build(MUSIC_ADAPTERS, profile.music, profile, "music")
    store = build(STORAGE_ADAPTERS, profile.storage, profile, "storage")
    tool_clients = build_tool_clients(profile)

    listening = ListeningController(
        ListeningMode(profile.dialogue.initial_mode), follow_up_seconds=profile.dialogue.follow_up_seconds
    )
    alerts = AlertScheduler(store, clock)
    flows: dict[str, FlowHandler] = {
        "weather": WeatherFlow(weather, profile.assistant.home_location, today),
        "timer": TimerFlow(alerts, clock),
        "alarm": AlarmFlow(alerts, clock),
        "lists": ListsFlow(store),
        "memory": MemoryFlow(store, clock),
        "date_time": ClockFlow(clock),
        "unit_conversion": ConvertFlow(),
        "math": CalculatorFlow(),
        "play_music": PlayMusicFlow(music),
        "music_control": MusicControlFlow(music),
        "listening_mode": ListeningModeFlow(listening),
        "chat": ChatFlow(llm),
    }
    calendar = tool_clients.get("calendar")
    if calendar is not None:
        flows["create_calendar_event"] = CreateEventFlow(calendar, today)
        flows["update_calendar_event"] = UpdateEventFlow(calendar, today)
    executor = CapabilityExecutor(flows, tool_clients, ResultRenderer(llm, FAILURE_TEXT, clock), FAILURE_TEXT)
    await executor.verify(registry)

    router = build_router(profile, registry, decider)
    dialogue = DialogueManager(
        registry,
        router,
        SlotExtractor(llm, clock),
        executor,
        decider,
        confirm_threshold=profile.router.confirm_threshold,
        max_reprompts=profile.dialogue.max_reprompts,
        follow_up_threshold=profile.router.follow_up_threshold,
        follow_up_seconds=profile.dialogue.context_seconds,
    )
    return Assistant(dialogue, listening, alerts)
