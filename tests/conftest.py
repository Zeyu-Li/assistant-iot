from __future__ import annotations

from datetime import date, datetime, timezone
from pathlib import Path
from typing import Callable, Iterable, Mapping, Optional

import pytest

from assistant.adapters.fakes import (
    FakeMusicPlayer,
    FakeToolClient,
    FakeWeatherProvider,
    ScriptedDecisionModel,
    ScriptedLanguageModel,
)
from assistant.core.dialogue import DialogueManager
from assistant.core.executor import CapabilityExecutor
from assistant.core.registry import Capability, Registry
from assistant.core.render import ResultRenderer
from assistant.core.router import Router, RouterThresholds
from assistant.core.slots import SlotExtractor
from assistant.flows.music import MusicControlFlow, PlayMusicFlow
from assistant.flows.weather import WeatherFlow
from assistant.ports.decision import Answer, ChoiceAnswer
from assistant.ports.tools import ToolSpec

REPO_ROOT = Path(__file__).resolve().parents[1]
TODAY = date(2026, 11, 25)  # a Wednesday
NOW = datetime(2026, 11, 25, 14, 5, tzinfo=timezone.utc)


def route(label: str, confidence: float = 0.9, others: Iterable[str] = ("none",)) -> ChoiceAnswer:
    rest = [o for o in others if o != label]
    share = (1 - confidence) / len(rest) if rest else 0.0
    return ChoiceAnswer({label: confidence, **{o: share for o in rest}})


CREATE_EVENT_SCHEMA = {
    "type": "object",
    "properties": {"summary": {"type": "string"}, "start": {"type": "string"}},
    "required": ["summary", "start"],
}

TEST_CAPABILITIES = [
    Capability.model_validate(c)
    for c in [
        {
            "id": "weather",
            "kind": "flow",
            "description": "Weather forecast.",
            "slots": {
                "type": "object",
                "properties": {"location": {"type": "string"}, "date": {"type": "string", "format": "date"}},
            },
        },
        {
            "id": "play_music",
            "kind": "flow",
            "description": "Play music.",
            "slots": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
            "prompts": {"query": "What would you like me to play?"},
        },
        {
            "id": "music_control",
            "kind": "flow",
            "description": "Control playback.",
            "slots": {
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": ["pause", "resume", "stop", "volume_up", "volume_down", "set_volume"]},
                    "level": {"type": "integer", "minimum": 0, "maximum": 100},
                },
                "required": ["action"],
            },
        },
        {
            "id": "create_calendar_event",
            "kind": "mcp",
            "description": "Add a calendar event.",
            "mcp": {"server": "calendar", "tool": "create_event"},
            "confirm": True,
            "confirm_prompt": "Add {summary} at {start}?",
            "render": {"mode": "fixed", "text": "Done, it's on your calendar."},
        },
        {
            "id": "read_email",
            "kind": "mcp",
            "description": "Read email.",
            "mcp": {"server": "gmail", "tool": "search_threads"},
        },
    ]
]


class Harness:
    """A DialogueManager wired to fakes, with handles on every fake."""

    def __init__(
        self,
        decisions: Mapping[str, Iterable[Answer]],
        json_responses: Iterable[dict] = (),
        text_responses: Iterable[str] = (),
        max_reprompts: int = 2,
        route_threshold: float = 0.5,
        calendar_tools: Optional[list[ToolSpec]] = None,
        capabilities: Optional[list[Capability]] = None,
        clock: Callable[[], float] = lambda: 0.0,
    ) -> None:
        self.registry = Registry(capabilities if capabilities is not None else TEST_CAPABILITIES)
        self.decider = ScriptedDecisionModel(decisions)
        self.llm = ScriptedLanguageModel(json_responses, text_responses)
        self.music = FakeMusicPlayer()
        self.weather = FakeWeatherProvider()
        self.calendar = FakeToolClient(
            calendar_tools if calendar_tools is not None else [ToolSpec("create_event", "", CREATE_EVENT_SCHEMA)]
        )
        self.gmail = FakeToolClient([ToolSpec("search_threads", "", {"type": "object", "properties": {"q": {"type": "string"}}})],
                                    {"search_threads": "From: Sam. Subject: Dinner"})
        flows = {
            "weather": WeatherFlow(self.weather, "Mountain View", lambda: TODAY),
            "play_music": PlayMusicFlow(self.music),
            "music_control": MusicControlFlow(self.music),
        }
        self.executor = CapabilityExecutor(
            flows, {"calendar": self.calendar, "gmail": self.gmail}, ResultRenderer(self.llm, "error", lambda: NOW), "failed"
        )
        self.dialogue = DialogueManager(
            self.registry,
            Router(self.decider, self.registry, RouterThresholds(route=route_threshold, addressed=0.7)),
            SlotExtractor(self.llm, lambda: NOW),
            self.executor,
            self.decider,
            confirm_threshold=0.6,
            max_reprompts=max_reprompts,
            follow_up_seconds=60,
            clock=clock,
        )


@pytest.fixture
def capabilities_path() -> Path:
    return REPO_ROOT / "config" / "capabilities.yaml"
