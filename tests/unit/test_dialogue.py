from __future__ import annotations

from conftest import TODAY, Harness, route

from assistant.ports.decision import ChoiceAnswer, YesNoAnswer


def agree(p: float) -> ChoiceAnswer:
    return route("agree", p, others=("decline",))


async def test_flow_runs_with_extracted_slots() -> None:
    h = Harness({"route": [route("weather")]}, json_responses=[{"location": "Denver", "date": "2026-11-26"}])

    reply = await h.dialogue.handle("will it snow in Denver tomorrow")

    assert reply is not None and not reply.expects_response
    assert reply.text.startswith("Tomorrow in Denver: partly cloudy")
    assert h.weather.requests == [("Denver", TODAY.replace(day=26))]
    assert not h.dialogue.in_progress


async def test_extraction_schema_never_forces_required_fields() -> None:
    h = Harness({"route": [route("play_music")]}, json_responses=[{"query": "jazz"}])

    await h.dialogue.handle("put on some jazz")

    _, _, schema = h.llm.json_calls[0]
    assert "required" not in schema and schema["additionalProperties"] is False


async def test_missing_required_slot_is_asked_then_filled() -> None:
    h = Harness({"route": [route("play_music")]}, json_responses=[{}, {"query": "Hotel California"}])

    first = await h.dialogue.handle("play something")
    second = await h.dialogue.handle("Hotel California")

    assert first is not None and first.expects_response and first.text == "What would you like me to play?"
    assert second is not None and second.text == "Playing Hotel California."
    _, user_prompt, _ = h.llm.json_calls[1]
    assert "The assistant asked: What would you like me to play?" in user_prompt


async def test_gives_up_after_max_reprompts() -> None:
    h = Harness({"route": [route("play_music")]}, max_reprompts=1)

    asks = [await h.dialogue.handle(text) for text in ("play", "um", "hmm")]

    assert [r.expects_response for r in asks if r] == [True, True, False]
    assert not h.dialogue.in_progress
    assert h.music.actions == []


async def test_cancel_abandons_pending_task() -> None:
    h = Harness({"route": [route("play_music")]})

    await h.dialogue.handle("play")
    reply = await h.dialogue.handle("Never mind.")

    assert reply is not None and reply.text == "Okay, cancelled."
    assert not h.dialogue.in_progress and h.music.actions == []


async def test_invalid_enum_value_is_dropped_and_reasked() -> None:
    h = Harness({"route": [route("music_control")]}, json_responses=[{"action": "explode"}, {"action": "pause"}])

    first = await h.dialogue.handle("do the thing to the music")
    second = await h.dialogue.handle("pause")

    assert first is not None and first.expects_response
    assert second is not None and second.text == "Paused."


async def test_unknown_fields_from_the_model_are_ignored() -> None:
    h = Harness({"route": [route("play_music")]}, json_responses=[{"query": "jazz", "shell": "rm -rf /"}])

    await h.dialogue.handle("play jazz")

    assert h.music.actions == ["play:jazz"]


async def test_write_waits_for_confirmation_and_runs_on_yes() -> None:
    h = Harness(
        {"route": [route("create_calendar_event")], "confirm": [agree(0.95)]},
        json_responses=[{"summary": "Lunch with Sam", "start": "2026-11-27T12:00:00"}],
    )

    ask = await h.dialogue.handle("add lunch with Sam Friday at noon")
    assert ask is not None and ask.expects_response
    assert ask.text == "Add Lunch with Sam at 2026-11-27T12:00:00?"
    assert h.calendar.calls == []

    done = await h.dialogue.handle("yes please")

    assert done is not None and done.text == "Done, it's on your calendar."
    assert h.calendar.calls == [("create_event", {"summary": "Lunch with Sam", "start": "2026-11-27T12:00:00"})]
    assert "The user replied: yes please" in h.decider.states[-1]


async def test_write_is_skipped_when_confirmation_is_unclear() -> None:
    h = Harness(
        {"route": [route("create_calendar_event")], "confirm": [agree(0.55)]},
        json_responses=[{"summary": "Lunch", "start": "2026-11-27T12:00:00"}],
    )

    await h.dialogue.handle("add lunch Friday at noon")
    reply = await h.dialogue.handle("hmm maybe")

    assert reply is not None and reply.text == "Okay, I won't."
    assert h.calendar.calls == []


async def test_mcp_read_is_summarised_with_result_marked_untrusted() -> None:
    h = Harness({"route": [route("read_email")]}, json_responses=[{"q": "is:unread"}], text_responses=["Sam asked about dinner."])

    reply = await h.dialogue.handle("read my email")

    assert reply is not None and reply.text == "Sam asked about dinner."
    assert h.gmail.calls == [("search_threads", {"q": "is:unread"})]
    system, user = h.llm.text_calls[0]
    assert "untrusted" in system and "<tool_result>" in user


async def test_low_confidence_route_declines() -> None:
    h = Harness({"route": [route("weather", confidence=0.3)]})

    reply = await h.dialogue.handle("blorp")

    assert reply is not None and reply.text == "Sorry, I can't help with that."


async def test_overheard_speech_stays_silent_when_not_addressed() -> None:
    h = Harness({"route": [route("weather")], "addressed": [YesNoAnswer(0.2)]})

    assert await h.dialogue.handle("the weather was awful on our trip", require_addressed=True) is None
    assert h.weather.requests == []
