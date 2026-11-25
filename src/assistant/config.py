"""Profiles: which adapter and model each port uses on a given machine."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, Optional, Union

import yaml
from pydantic import BaseModel, ConfigDict, Field

from assistant.adapters.google.oauth import GoogleOAuthSettings


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class AdapterSettings(BaseModel):
    """``adapter`` picks the implementation; any other keys are that adapter's options."""

    model_config = ConfigDict(extra="allow")

    adapter: str

    @property
    def options(self) -> dict[str, Any]:
        return dict(self.model_extra or {})


class AssistantSettings(_Strict):
    home_location: str
    home_coordinates: Optional[tuple[float, float]] = None
    """Latitude, longitude of home, so an ambiguous name ("Mountain View") is never geocoded wrongly."""
    units: Literal["celsius", "fahrenheit"] = "fahrenheit"
    timezone: Optional[str] = None
    """IANA name such as America/Los_Angeles; defaults to the system time zone."""


class RouterSettings(_Strict):
    route_threshold: float = Field(0.5, ge=0, le=1)
    addressed_threshold: float = Field(0.7, ge=0, le=1)
    confirm_threshold: float = Field(0.6, ge=0, le=1)
    include_examples: bool = False
    """Show example phrasings next to each option. Small encoders such as Laya have a tight
    option-token budget, so leave this off for them; LLM-based deciders benefit from it."""


class NetworkSettings(_Strict):
    allowed_hosts: tuple[str, ...] = ()
    """Remote hosts adapters may call. Loopback is always allowed; everything else is blocked."""


class DialogueSettings(_Strict):
    max_reprompts: int = Field(2, ge=0)
    follow_up_seconds: float = Field(8.0, ge=0)
    initial_mode: Literal["muted", "wake_word", "open_mic"] = "wake_word"


class VoiceSettings(_Strict):
    wake_phrases: tuple[str, ...] = ("hey computer",)
    """Plain text; any English phrase works (two or more words wake less by accident)."""
    wake: AdapterSettings
    vad: AdapterSettings
    stt: AdapterSettings
    tts: AdapterSettings
    input_device: Optional[Union[int, str]] = None
    """sounddevice device index or name substring; default is the system input."""
    output_device: Optional[Union[int, str]] = None
    earcon: bool = True
    """Play a short tone when the wake phrase is heard."""
    control_socket: Path = Path("~/.local/state/assistant-iot/control.sock")


class Profile(_Strict):
    assistant: AssistantSettings
    decision: AdapterSettings
    llm: AdapterSettings
    weather: AdapterSettings
    music: AdapterSettings
    mcp_servers: dict[str, AdapterSettings] = Field(default_factory=dict)
    router: RouterSettings = RouterSettings()
    dialogue: DialogueSettings = DialogueSettings()
    network: NetworkSettings = NetworkSettings()
    google_oauth: GoogleOAuthSettings = GoogleOAuthSettings()
    """Used by MCP servers with ``auth: google`` and by ``assistant google-login``."""
    voice: Optional[VoiceSettings] = None
    """Microphone, speaker and speech models; needed by ``assistant listen``."""
    disabled_capabilities: tuple[str, ...] = ()


def load_profile(path: Path) -> Profile:
    return Profile.model_validate(yaml.safe_load(path.read_text()))


def resolve_profile_path(name_or_path: str, config_dir: Path) -> Path:
    candidate = Path(name_or_path)
    if candidate.suffix in (".yaml", ".yml"):
        return candidate
    return config_dir / "profiles" / f"{name_or_path}.yaml"
