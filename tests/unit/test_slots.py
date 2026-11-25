from __future__ import annotations

from conftest import NOW

from assistant.adapters.fakes import ScriptedLanguageModel
from assistant.core.slots import SlotExtractor, invalid_fields, missing_required

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
