"""Argument extraction and validation against a capability's JSON Schema."""

from __future__ import annotations

import logging
import re
from datetime import datetime, timedelta
from typing import Any, Callable, Mapping, Optional

from jsonschema import Draft202012Validator

from assistant.ports.language import LanguageModel

log = logging.getLogger(__name__)

_EMPTY = (None, "", [], {})

VERBATIM = "x-verbatim"
"""Schema keyword: the value must come from what the user actually said. For text, its words
must appear in the utterance; for a date or date-time, the utterance must mention a time."""


_PLACEHOLDERS = frozenset({"null", "none", "n/a", "unknown", "unspecified"})


def _blank(value: Any) -> bool:
    """Empty, or a placeholder string small models write instead of leaving a field out."""
    return value in _EMPTY or (isinstance(value, str) and value.strip().lower() in _PLACEHOLDERS)


def _standard_keys(spec: Mapping[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in spec.items() if not k.startswith("x-")}


def _words(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


_TEMPORAL = re.compile(
    r"\d|\b(today|tonight|tomorrow|yesterday|noon|midnight|morning|afternoon|evening|night|week|weekend|"
    r"month|monday|tuesday|wednesday|thursday|friday|saturday|sunday|january|february|march|april|may|"
    r"june|july|august|september|october|november|december|hour|minute|later|next|one|two|three|four|"
    r"five|six|seven|eight|nine|ten|eleven|twelve)\b",
    re.IGNORECASE,
)


def _grounded(value: Any, utterance: str, spec: Mapping[str, Any]) -> bool:
    if spec.get("format") in ("date", "date-time"):
        # A date the user never mentioned is the model filling in "now".
        return bool(_TEMPORAL.search(utterance))
    value_words = _words(str(value))
    return bool(value_words) and len(value_words & _words(utterance)) * 2 >= len(value_words)


class SlotExtractor:
    def __init__(self, llm: LanguageModel, clock: Callable[[], datetime]) -> None:
        self._llm = llm
        self._clock = clock

    async def extract(
        self, utterance: str, schema: Mapping[str, Any], *, question: Optional[str] = None
    ) -> dict[str, Any]:
        """Return the arguments the user stated, keyed by schema property; unknown keys are dropped."""
        properties: Mapping[str, Any] = schema.get("properties") or {}
        if not properties:
            return {}
        # Nothing is required here: forcing required fields makes small models invent values.
        extraction_schema = {
            "type": "object",
            "properties": {name: _standard_keys(spec) for name, spec in properties.items()},
            "additionalProperties": False,
        }
        raw = await self._llm.generate_json(
            f"{self._system_prompt()}\nFields:\n{_describe_fields(properties)}",
            _user_prompt(utterance, question),
            extraction_schema,
        )
        values = {k: v for k, v in raw.items() if k in properties and not _blank(v)}
        for name in [n for n in values if properties[n].get(VERBATIM) and not _grounded(values[n], utterance, properties[n])]:
            log.info("dropping %s=%r: not something the user said", name, values[name])
            del values[name]
        return values

    def _system_prompt(self) -> str:
        now = self._clock()
        # Small models get weekday arithmetic wrong, so give them a lookup table instead.
        upcoming = "\n".join(
            f"- {day:%A} {day:%Y-%m-%d}{label}"
            for offset, label in enumerate([" (today)", " (tomorrow)", "", "", "", "", ""])
            for day in [now.date() + timedelta(days=offset)]
        )
        return (
            "You fill in the arguments of a voice assistant command.\n"
            f"Current local time: {now:%Y-%m-%d %H:%M} ({now.tzname()}). The next seven days:\n{upcoming}\n"
            "Rules: copy every value the user stated, such as names and places. Leave out a field only "
            "when the user gave nothing for it; never invent a value. Write dates as YYYY-MM-DD and "
            "date-times as YYYY-MM-DDTHH:MM:SS, using the table above for weekday names."
        )


def _describe_fields(properties: Mapping[str, Any]) -> str:
    """Spell the schema out in the prompt: constrained decoding enforces it but never shows it to the model."""
    lines = []
    for name, spec in properties.items():
        kind = spec.get("type", "any")
        if "enum" in spec:
            kind = "one of " + ", ".join(map(str, spec["enum"]))
        elif "format" in spec:
            kind = f"{kind}, {spec['format']}"
        description = spec.get("description", "")
        lines.append(f"- {name} ({kind}){': ' + description if description else ''}")
    return "\n".join(lines)


def _user_prompt(utterance: str, question: Optional[str]) -> str:
    if question:
        return f"The assistant asked: {question}\nThe user answered: {utterance}"
    return f"The user said: {utterance}"


def missing_required(schema: Mapping[str, Any], arguments: Mapping[str, Any]) -> list[str]:
    return [name for name in schema.get("required", ()) if arguments.get(name) in _EMPTY]


def invalid_fields(schema: Mapping[str, Any], arguments: Mapping[str, Any]) -> set[str]:
    """Top-level fields whose present value breaks the schema (wrong type, not in enum, ...)."""
    present_only = {k: v for k, v in schema.items() if k != "required"}
    validator = Draft202012Validator(present_only, format_checker=Draft202012Validator.FORMAT_CHECKER)
    return {str(error.absolute_path[0]) for error in validator.iter_errors(dict(arguments)) if error.absolute_path}
