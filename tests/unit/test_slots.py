from __future__ import annotations

from conftest import NOW

from assistant.adapters.fakes import ScriptedLanguageModel
from assistant.core.slots import PreviousRequest, SlotExtractor, invalid_fields, missing_required

SCHEMA = {
    "type": "object",
    "properties": {
        "location": {"type": "string", "description": "Place the user named.", "x-verbatim": True},
        "date": {"type": "string", "format": "date"},
        "mode": {"type": "string", "enum": ["a", "b"]},
    },
    "required": ["mode"],
}


async def test_prompt_spells_out_fields_and_upcoming_days() -> None:
    llm = ScriptedLanguageModel([{}])
    await SlotExtractor(llm, lambda: NOW).extract("hi", SCHEMA)

    system, _, sent_schema = llm.json_calls[0]
    assert "- location (string): Place the user named." in system
    assert "- mode (one of a, b)" in system
    assert "Saturday 2026-11-28" in system and "Wednesday 2026-11-25 (today)" in system
    assert "x-verbatim" not in str(sent_schema)


async def test_verbatim_values_must_come_from_the_user() -> None:
    llm = ScriptedLanguageModel([{"location": "EST", "date": "2026-11-25"}, {"location": "lake tahoe"}])
    extractor = SlotExtractor(llm, lambda: NOW)

    assert await extractor.extract("is it going to rain today", SCHEMA) == {"date": "2026-11-25"}
    assert await extractor.extract("weather at Tahoe", SCHEMA) == {"location": "lake tahoe"}


async def test_verbatim_date_needs_a_time_mentioned() -> None:
    schema = {"type": "object", "properties": {"start": {"type": "string", "format": "date-time", "x-verbatim": True}}}
    llm = ScriptedLanguageModel([{"start": "2026-11-25T14:29:00"}, {"start": "2026-11-28T14:00:00"}])
    extractor = SlotExtractor(llm, lambda: NOW)

    assert await extractor.extract("put a dentist appointment on my calendar", schema) == {}
    assert await extractor.extract("Saturday at 2pm", schema) == {"start": "2026-11-28T14:00:00"}


async def test_placeholder_strings_count_as_missing() -> None:
    llm = ScriptedLanguageModel([{"location": "null", "mode": "None"}])
    assert await SlotExtractor(llm, lambda: NOW).extract("hi", SCHEMA) == {}


def test_validation_helpers() -> None:
    assert missing_required(SCHEMA, {"location": "x"}) == ["mode"]
    assert missing_required(SCHEMA, {"mode": ""}) == ["mode"]
    assert invalid_fields(SCHEMA, {"mode": "c", "date": "tomorrow", "location": "ok"}) == {"mode", "date"}


ACTION_SCHEMA = {
    "type": "object",
    "properties": {"action": {"type": "string", "enum": ["set", "check"]}, "duration": {"type": "string"}},
    "allOf": [{"if": {"properties": {"action": {"const": "set"}}}, "then": {"required": ["duration"]}}],
}


def test_required_only_for_some_actions() -> None:
    assert missing_required(ACTION_SCHEMA, {"action": "set"}) == ["duration"]
    assert missing_required(ACTION_SCHEMA, {"action": "check"}) == []
    assert missing_required(ACTION_SCHEMA, {"action": "set", "duration": "PT5M"}) == []
    assert invalid_fields(ACTION_SCHEMA, {"action": "set"}) == set()  # a missing field is not an invalid one


async def test_named_weekday_overrides_wrong_arithmetic_and_offsets_are_dropped() -> None:
    schema = {"type": "object", "properties": {"day": {"type": "string", "format": "date"},
                                               "at": {"type": "string", "format": "date-time"}}}
    llm = ScriptedLanguageModel([
        {"day": "2026-11-27"},  # NOW is Wednesday 25th; the model said Friday for Saturday
        {"at": "2026-11-27T09:00:00Z"},
        {"day": "2026-11-26"},  # two weekdays named: leave it alone
        {"day": "2026-11-23"},
    ])
    extractor = SlotExtractor(llm, lambda: NOW)

    assert await extractor.extract("what about Saturday", schema) == {"day": "2026-11-28"}
    assert await extractor.extract("move it to Thursday at 9", schema) == {"at": "2026-11-26T09:00:00"}
    assert await extractor.extract("Thursday or Friday", schema) == {"day": "2026-11-26"}
    assert await extractor.extract("what did I have last Monday", schema) == {"day": "2026-11-23"}


async def test_verbatim_duration_needs_a_length_mentioned() -> None:
    schema = {"type": "object", "properties": {"duration": {"type": "string", "format": "duration", "x-verbatim": True}}}
    llm = ScriptedLanguageModel([{"duration": "PT10M"}, {"duration": "PT30M"}])
    extractor = SlotExtractor(llm, lambda: NOW)

    assert await extractor.extract("set a timer", schema) == {}
    assert await extractor.extract("half an hour", schema) == {"duration": "PT30M"}


async def test_follow_up_prompt_carries_the_previous_request() -> None:
    llm = ScriptedLanguageModel([{"date": "2026-11-28"}])
    previous = PreviousRequest("what's the weather tomorrow", {"date": "2026-11-26"})

    await SlotExtractor(llm, lambda: NOW).extract("what about Saturday", SCHEMA, previous=previous)

    _, user, _ = llm.json_calls[0]
    assert "Earlier the user said: what's the weather tomorrow" in user and '"date": "2026-11-26"' in user
    assert "Now the user follows up: what about Saturday" in user
