"""In-memory fakes for every port: used by tests and by the model-free ``fake`` profile."""

from __future__ import annotations

import math
import re
from collections import deque
from datetime import date
from typing import Any, Iterable, Mapping, Optional

from assistant.ports.decision import Answer, ChoiceAnswer, ChoiceQuestion, Question, YesNoAnswer
from assistant.ports.tools import ToolResult, ToolSpec
from assistant.ports.weather import DailyForecast, LocationNotFound

_STOPWORDS = frozenset(
    "a an the is it to of for on in at my me i you what whats how do does can could will be "
    "and or with by this that some any user said examples".split()
)
_YES = frozenset("yes yeah yep sure ok okay go ahead please do it correct right".split())
_NO = frozenset("no nope don't dont not cancel stop wait".split())


def _tokens(text: str) -> set[str]:
    return {w[:5] for w in re.findall(r"[a-z']+", text.lower()) if w not in _STOPWORDS}


class KeywordDecisionModel:
    """Crude word-overlap stand-in for a real decision model, so the REPL runs with no models."""

    def __init__(self, none_label: str = "other", none_score: float = 0.5) -> None:
        self._none_label = none_label
        self._none_score = none_score

    async def decide(self, state: str, questions: Mapping[str, Question]) -> Mapping[str, Answer]:
        return {name: self._answer(state, q) for name, q in questions.items()}

    def _answer(self, state: str, question: Question) -> Answer:
        if isinstance(question, ChoiceQuestion):
            words = _tokens(state)
            scores = {
                label: self._none_score if label == self._none_label else float(len(words & _tokens(text)))
                for label, text in question.options.items()
            }
            peak = max(scores.values())
            weights = {label: math.exp(2.0 * (s - peak)) for label, s in scores.items()}
            total = sum(weights.values())
            return ChoiceAnswer({label: w / total for label, w in weights.items()})
        reply = state.rsplit(":", 1)[-1].lower()
        words = set(re.findall(r"[a-z']+", reply))
        if words & _NO:
            return YesNoAnswer(0.05)
        return YesNoAnswer(0.95 if words & _YES else 0.5)


class ScriptedDecisionModel:
    """Returns pre-set answers per question name; records every state it was asked about."""

    def __init__(self, answers: Mapping[str, Iterable[Answer]]) -> None:
        self._answers = {name: deque(queue) for name, queue in answers.items()}
        self.states: list[str] = []

    async def decide(self, state: str, questions: Mapping[str, Question]) -> Mapping[str, Answer]:
        self.states.append(state)
        return {name: self._answers[name].popleft() for name in questions}


class ScriptedLanguageModel:
    """Pops queued JSON objects / texts; returns {} and a canned sentence when the queues run dry."""

    def __init__(self, json_responses: Iterable[dict[str, Any]] = (), text_responses: Iterable[str] = ()) -> None:
        self._json = deque(json_responses)
        self._text = deque(text_responses)
        self.json_calls: list[tuple[str, str, Mapping[str, Any]]] = []
        self.text_calls: list[tuple[str, str]] = []

    async def generate_json(self, system: str, user: str, schema: Mapping[str, Any]) -> dict[str, Any]:
        self.json_calls.append((system, user, schema))
        return self._json.popleft() if self._json else {}

    async def generate_text(self, system: str, user: str) -> str:
        self.text_calls.append((system, user))
        return self._text.popleft() if self._text else "This is a placeholder answer from the fake model."


class FakeToolClient:
    def __init__(self, tools: Iterable[ToolSpec], responses: Optional[Mapping[str, str]] = None) -> None:
        self._tools = list(tools)
        self._responses = dict(responses or {})
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def list_tools(self) -> list[ToolSpec]:
        return list(self._tools)

    async def call_tool(self, name: str, arguments: Mapping[str, Any]) -> ToolResult:
        self.calls.append((name, dict(arguments)))
        return ToolResult(self._responses.get(name, f"{name} ok"))


class FakeMusicPlayer:
    def __init__(self, catalogue_miss: Iterable[str] = ()) -> None:
        self._misses = {q.lower() for q in catalogue_miss}
        self.actions: list[str] = []
        self.volume = 50

    async def play(self, query: str) -> Optional[str]:
        self.actions.append(f"play:{query}")
        return None if query.lower() in self._misses else query

    async def pause(self) -> None:
        self.actions.append("pause")

    async def resume(self) -> None:
        self.actions.append("resume")

    async def stop(self) -> None:
        self.actions.append("stop")

    async def set_volume(self, level: int) -> int:
        self.volume = max(0, min(100, level))
        self.actions.append(f"volume:{self.volume}")
        return self.volume

    async def adjust_volume(self, delta: int) -> int:
        return await self.set_volume(self.volume + delta)


class FakeWeatherProvider:
    def __init__(self, unknown_locations: Iterable[str] = ()) -> None:
        self._unknown = {loc.lower() for loc in unknown_locations}
        self.requests: list[tuple[str, date]] = []

    async def daily_forecast(self, location: str, day: date) -> DailyForecast:
        self.requests.append((location, day))
        if location.lower() in self._unknown:
            raise LocationNotFound(location)
        return DailyForecast(location=location, day=day, summary="partly cloudy", high=61.4, low=47.6, precipitation_chance=20)
