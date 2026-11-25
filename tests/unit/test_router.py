from __future__ import annotations

from conftest import TEST_CAPABILITIES, route

from assistant.adapters.fakes import ScriptedDecisionModel
from assistant.core.registry import NONE_ID, Registry
from assistant.core.router import OTHER_LABEL, Router, RouterThresholds
from assistant.ports.decision import ChoiceQuestion, YesNoAnswer

THRESHOLDS = RouterThresholds(route=0.6, addressed=0.7)


def make_router(decider: ScriptedDecisionModel) -> Router:
    return Router(decider, Registry(TEST_CAPABILITIES), THRESHOLDS)


async def test_options_cover_every_capability_plus_none() -> None:
    seen: list[ChoiceQuestion] = []

    class Spy(ScriptedDecisionModel):
        async def decide(self, state, questions):  # type: ignore[no-untyped-def]
            seen.append(questions["route"])
            return await super().decide(state, questions)

    await make_router(Spy({"route": [route("weather")]})).route("weather?", require_addressed=False)

    assert set(seen[0].options) == {c.id for c in TEST_CAPABILITIES} | {OTHER_LABEL}


async def test_other_label_maps_to_none() -> None:
    decision = await make_router(ScriptedDecisionModel({"route": [route(OTHER_LABEL, 0.9, others=("weather",))]})).route(
        "x", require_addressed=False
    )
    assert decision.capability_id is None and decision.best_label == NONE_ID


async def test_accepts_confident_route() -> None:
    decision = await make_router(ScriptedDecisionModel({"route": [route("weather", 0.8)]})).route("x", require_addressed=False)
    assert decision.capability_id == "weather"


async def test_rejects_below_threshold_and_none() -> None:
    low = await make_router(ScriptedDecisionModel({"route": [route("weather", 0.59)]})).route("x", require_addressed=False)
    none = await make_router(ScriptedDecisionModel({"route": [route(NONE_ID, 0.99, others=("weather",))]})).route(
        "x", require_addressed=False
    )
    assert low.capability_id is None and low.best_label == "weather"
    assert none.capability_id is None


async def test_addressed_gate_only_when_required() -> None:
    gated = ScriptedDecisionModel({"route": [route("weather")], "addressed": [YesNoAnswer(0.69)]})
    decision = await make_router(gated).route("x", require_addressed=True)
    assert decision.capability_id is None and decision.p_addressed == 0.69

    ungated = ScriptedDecisionModel({"route": [route("weather")]})
    assert (await make_router(ungated).route("x", require_addressed=False)).p_addressed is None
