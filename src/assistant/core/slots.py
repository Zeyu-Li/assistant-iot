"""Argument extraction and validation against a capability's JSON Schema."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any, Callable, Mapping, Optional

from jsonschema import Draft202012Validator

from assistant.ports.language import LanguageModel

log = logging.getLogger(__name__)

_EMPTY = (None, "", [], {})

VERBATIM = "x-verbatim"
"""Schema keyword: the value must come from what the user actually said. For text, its words
must appear in the utterance; for a date, date-time or duration, the utterance must mention a time."""
_TIME_FORMATS = frozenset({"date", "date-time", "duration"})


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
    r"five|six|seven|eight|nine|ten|eleven|twelve|twenty|thirty|forty|fifty|sixty|ninety|half|quarter|dozen|"
    r"seconds?|minutes?|hours?|days?|christmas|halloween|thanksgiving|easter|valentine'?s|new year'?s?)\b",
    re.IGNORECASE,
)


def _grounded(value: Any, utterance: str, spec: Mapping[str, Any]) -> bool:
    if spec.get("format") in _TIME_FORMATS:
        # A date the user never mentioned is the model filling in "now".
        return bool(_TEMPORAL.search(utterance))
    value_words = _words(str(value))
    return bool(value_words) and len(value_words & _words(utterance)) * 2 >= len(value_words)


class SlotExtractor:
    def __init__(self, llm: LanguageModel, clock: Callable[[], datetime]) -> None:
        self._llm = llm
        self._clock = clock

    async def extract(
        self,
        utterance: str,
        schema: Mapping[str, Any],
        *,
        question: Optional[str] = None,
        previous: Optional[PreviousRequest] = None,
    ) -> dict[str, Any]:
        """Return the arguments the user stated, keyed by schema property; unknown keys are dropped.

        ``question`` is what the assistant just asked; ``previous`` is the request this
        utterance follows up on ("what about Saturday?"), so only changed fields come back.
        """
        properties: Mapping[str, Any] = schema.get("properties") or {}
        if not properties:
            return {}
        # Nothing is required here: forcing required fields makes small models invent values.
        extraction_schema = {
            "type": "object",
            "properties": {name: _standard_keys(spec) for name, spec in properties.items()},
            "additionalProperties": False,
        }
        now = self._clock()
        # The clock goes in the user turn: a system prompt that changes every minute would
        # stop the LLM server from reusing its cached prompt, costing ~1 s per turn.
        raw = await self._llm.generate_json(
            f"{_system_prompt(now.date())}\nFields:\n{_describe_fields(properties)}",
            # No time-zone abbreviation: models copied "EST" into place fields.
            f"Current local time: {now:%A %Y-%m-%d %H:%M}.\n{_user_prompt(utterance, question, previous)}",
            extraction_schema,
        )
        values = {k: v for k, v in raw.items() if k in properties and not _blank(v)}
        for name in [n for n in values if properties[n].get(VERBATIM) and not _grounded(values[n], utterance, properties[n])]:
            log.info("dropping %s=%r: not something the user said", name, values[name])
            del values[name]
        for name, value in values.items():
            if properties[name].get("format") in ("date", "date-time") and isinstance(value, str):
                fixed = _match_named_weekday(_strip_offset(value), utterance, now.date())
                if fixed != value:
                    log.info("normalising %s=%r to %r (local time, the weekday the user named)", name, value, fixed)
                    values[name] = fixed
        return values


_TZ_SUFFIX = re.compile(r"(Z|[+-]\d\d:?\d\d)$", re.IGNORECASE)


def _strip_offset(value: str) -> str:
    """A spoken time is local wall-clock time, so an offset the model appends ("Z") is noise."""
    return _TZ_SUFFIX.sub("", value) if "T" in value else value


_WEEKDAYS = ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")
_WEEKDAY_MENTION = re.compile(r"\b(last\s+)?(" + "|".join(_WEEKDAYS) + r")\b", re.IGNORECASE)


def _match_named_weekday(value: str, utterance: str, today: date) -> str:
    """Move a date onto the weekday the user named, when the model got the arithmetic wrong.

    Small models reliably turn "what about Saturday" into Friday. Only acts when exactly one
    weekday is named: the coming one (today counts), or the previous one after "last".
    """
    mentions = {(bool(m.group(1)), m.group(2).lower()) for m in _WEEKDAY_MENTION.finditer(utterance)}
    if len(mentions) != 1:
        return value
    (past, name), = mentions
    try:
        day = date.fromisoformat(value[:10])
    except ValueError:
        return value
    wanted = _WEEKDAYS.index(name)
    if day.weekday() == wanted:
        return value
    if past:
        target = today - timedelta(days=(today.weekday() - wanted) % 7 or 7)
    else:
        target = today + timedelta(days=(wanted - today.weekday()) % 7)
    return target.isoformat() + value[10:]


def _system_prompt(today: date) -> str:
    # Small models get weekday arithmetic wrong, so give them a lookup table instead.
    upcoming = "\n".join(
        f"- {day:%A} {day:%Y-%m-%d}{label}"
        for offset, label in enumerate([" (today)", " (tomorrow)", "", "", "", "", ""])
        for day in [today + timedelta(days=offset)]
    )
    return (
        "You fill in the arguments of a voice assistant command.\n"
        f"The next seven days:\n{upcoming}\n"
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


@dataclass(frozen=True)
class PreviousRequest:
    request: str
    arguments: Mapping[str, Any]


def _user_prompt(utterance: str, question: Optional[str], previous: Optional[PreviousRequest]) -> str:
    if question:
        return f"The assistant asked: {question}\nThe user answered: {utterance}"
    if previous:
        return (
            f"Earlier the user said: {previous.request}\n"
            f"That was handled with these fields: {json.dumps(dict(previous.arguments), default=str)}\n"
            f"Now the user follows up: {utterance}\n"
            "Give only the fields the follow-up changes. If it changes the day or time, give every "
            "date and time field for the new day or time."
        )
    return f"The user said: {utterance}"


def missing_required(schema: Mapping[str, Any], arguments: Mapping[str, Any]) -> list[str]:
    """Required fields that are still empty, in order.

    Besides top-level ``required``, entries of ``allOf`` shaped ``{if: ..., then: {required: [...]}}``
    add fields that are only required in some cases, for example a timer's duration only
    when the action is "set".
    """
    required = list(schema.get("required", ()))
    present = {k: v for k, v in arguments.items() if v not in _EMPTY}
    for rule in schema.get("allOf", ()):
        if "if" in rule and Draft202012Validator(rule["if"]).is_valid(present):
            required += [n for n in rule.get("then", {}).get("required", ()) if n not in required]
    return [name for name in required if arguments.get(name) in _EMPTY]


def invalid_fields(schema: Mapping[str, Any], arguments: Mapping[str, Any]) -> set[str]:
    """Top-level fields whose present value breaks the schema (wrong type, not in enum, ...)."""
    present_only = {k: v for k, v in schema.items() if k != "required"}
    validator = Draft202012Validator(present_only, format_checker=Draft202012Validator.FORMAT_CHECKER)
    return {str(error.absolute_path[0]) for error in validator.iter_errors(dict(arguments)) if error.absolute_path}
