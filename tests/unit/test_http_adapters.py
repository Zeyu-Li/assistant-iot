from __future__ import annotations

import json
from typing import Any, Callable

import httpx
import pytest

from assistant.adapters.decision.systemone_http import SystemOneDecisionModel
from assistant.adapters.llm.ollama import OllamaLanguageModel, OllamaSettings
from assistant.adapters.llm.openai_compat import OpenAICompatLanguageModel, OpenAICompatSettings
from assistant.adapters.net.allowlist import EgressBlocked, HttpClientFactory
from assistant.ports.decision import ChoiceQuestion, YesNoQuestion
from assistant.ports.language import Preloadable

Handler = Callable[[httpx.Request], httpx.Response]


def client(handler: Handler, base_url: str = "http://127.0.0.1:8000", allowed: tuple[str, ...] = ()) -> httpx.AsyncClient:
    return HttpClientFactory(allowed).create(base_url=base_url, timeout=5, transport=httpx.MockTransport(handler))


class Recorder:
    def __init__(self, response: dict[str, Any]) -> None:
        self.response = response
        self.requests: list[httpx.Request] = []

    @property
    def bodies(self) -> list[dict[str, Any]]:
        return [json.loads(r.content) for r in self.requests]

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return httpx.Response(200, json=self.response)


ROUTE = ChoiceQuestion("Which?", {"weather": "Weather.", "none": "Nothing."})


async def test_systemone_request_and_choice_decoding() -> None:
    server = Recorder({"answers": {"route": {"type": "choice", "choice": "weather", "probabilities": {"weather": 0.8, "none": 0.2}}}})
    decider = SystemOneDecisionModel(client(server), "laya")

    answers = await decider.decide("The user said: rain?", {"route": ROUTE})

    assert server.bodies == [
        {
            "model": "laya",
            "state": "The user said: rain?",
            "questions": {"route": {"type": "choice", "instructions": "Which?", "criteria": {"weather": "Weather.", "none": "Nothing."}}},
        }
    ]
    assert answers["route"].best == "weather" and answers["route"].confidence == 0.8  # type: ignore[union-attr]


async def test_systemone_refusal_means_no_confidence() -> None:
    server = Recorder({"answers": {"route": {"type": "refusal"}, "ok": {"type": "refusal"}}})
    answers = await SystemOneDecisionModel(client(server), "m").decide("s", {"route": ROUTE, "ok": YesNoQuestion("Ok?")})

    assert answers["route"].probabilities == {"weather": 0.0, "none": 0.0}  # type: ignore[union-attr]
    assert answers["ok"].p_yes == 0.0  # type: ignore[union-attr]


async def test_systemone_yes_no_as_noul_or_choice() -> None:
    noul_server = Recorder({"answers": {"ok": {"type": "noul", "noul": 0.9}}})
    noul = await SystemOneDecisionModel(client(noul_server), "m").decide("s", {"ok": YesNoQuestion("Ok?")})
    assert noul_server.bodies[0]["questions"]["ok"] == {"type": "noul", "instructions": "Ok?"}
    assert noul["ok"].p_yes == 0.9  # type: ignore[union-attr]

    choice_server = Recorder({"answers": {"ok": {"type": "choice", "probabilities": {"affirmative": 0.7, "negative": 0.3}}}})
    as_choice = await SystemOneDecisionModel(client(choice_server), "m", yes_no_as="choice").decide("s", {"ok": YesNoQuestion("Ok?")})
    assert choice_server.bodies[0]["questions"]["ok"]["type"] == "choice"
    assert as_choice["ok"].p_yes == 0.7  # type: ignore[union-attr]


def chat_response(content: str) -> dict[str, Any]:
    return {"choices": [{"message": {"role": "assistant", "content": content}}]}


async def test_openai_compat_constrains_json_and_strips_thinking() -> None:
    server = Recorder(chat_response('<think>hmm</think>{"query": "jazz"}'))
    llm = OpenAICompatLanguageModel(
        client(server, base_url="http://127.0.0.1:11434/v1"), OpenAICompatSettings(base_url="unused", model="gemma")
    )
    schema = {"type": "object", "properties": {"query": {"type": "string"}}}

    assert await llm.generate_json("sys", "user", schema) == {"query": "jazz"}
    body = server.bodies[0]
    assert body["response_format"]["json_schema"]["schema"] == schema
    assert body["messages"][0] == {"role": "system", "content": "sys"}


async def test_openai_compat_invalid_json_returns_empty() -> None:
    llm = OpenAICompatLanguageModel(client(Recorder(chat_response("not json"))), OpenAICompatSettings(base_url="x", model="m"))
    assert await llm.generate_json("s", "u", {"type": "object"}) == {}


async def test_ollama_preload_pins_model_through_native_api() -> None:
    server = Recorder({"done": True})
    base_url = "http://127.0.0.1:11434/v1"
    llm = OllamaLanguageModel(client(server, base_url=base_url), OllamaSettings(base_url=base_url, model="gemma3:12b"))

    assert isinstance(llm, Preloadable)
    await llm.preload()

    assert str(server.requests[0].url) == "http://127.0.0.1:11434/api/generate"
    assert server.bodies == [{"model": "gemma3:12b", "keep_alive": -1}]


async def test_allowlist_blocks_unlisted_hosts() -> None:
    ok = Recorder({"fine": True})
    async with client(ok, base_url="https://api.open-meteo.com", allowed=("api.open-meteo.com",)) as allowed:
        assert (await allowed.get("/v1/forecast")).status_code == 200
    async with client(ok, base_url="https://example.com", allowed=("api.open-meteo.com",)) as blocked:
        with pytest.raises(EgressBlocked):
            await blocked.get("/")
    assert [r.url.host for r in ok.requests] == ["api.open-meteo.com"]
