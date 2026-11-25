"""Date/time questions, unit conversion, the calculator and number/duration phrasing."""

from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

import pytest

from assistant.flows.calculator import CalculationError, CalculatorFlow, evaluate
from assistant.flows.clock import ClockFlow
from assistant.flows.convert import ConvertFlow, unit_phrase
from assistant.flows.speech import duration_phrase, number_phrase, parse_duration
from assistant.ports.flow import FlowContext

NOW = datetime(2026, 11, 25, 17, 42, tzinfo=ZoneInfo("America/Los_Angeles"))  # Wednesday


def run(flow: Any, **arguments: object) -> str:
    return asyncio.run(flow.run(FlowContext("", arguments)))


def test_clock_answers() -> None:
    clock = ClockFlow(lambda: NOW)
    assert run(clock, question="time") == "It's 5:42 PM."
    assert run(clock, question="date") == "It's Wednesday, November 25."
    assert run(clock, question="time", place="Tokyo") == "It's 10:42 AM in Tokyo."
    assert run(clock, question="time", place="new york") == "It's 8:42 PM in new york."
    assert run(clock, question="time", place="Atlantis") == "I don't know the time zone for Atlantis."
    assert run(clock, question="weekday", date="2026-07-04") == "July 4 was a Saturday."
    assert run(clock, question="weekday", date="2026-12-25") == "December 25 is a Friday."
    assert run(clock, question="days_until", date="2026-12-25") == "It's in 30 days, on December 25."
    assert run(clock, question="days_until", date="2026-11-26") == "That's tomorrow."
    assert run(clock, question="days_until", date="2026-11-20") == "That was 5 days ago."


def test_unit_conversions() -> None:
    convert = ConvertFlow()
    assert run(convert, value=1, from_unit="cup", to_unit="tablespoon") == "1 cup is 16 tablespoons."
    assert run(convert, value=70, from_unit="fahrenheit", to_unit="celsius") == "70 degrees Fahrenheit is 21.1 degrees Celsius."
    assert run(convert, value=1, from_unit="mile", to_unit="foot") == "1 mile is 5,280 feet."
    assert run(convert, value=60, from_unit="miles per hour", to_unit="kilometers per hour") == (
        "60 miles per hour is 96.6 kilometers per hour."
    )
    assert run(convert, value=1, from_unit="cup", to_unit="kilometers") == "I can't convert cup to kilometers."
    assert run(convert, value=1, from_unit="smoot", to_unit="feet").startswith("I don't know the unit")


def test_unit_phrase_plurals() -> None:
    assert unit_phrase(2, "inch") == "2 inches" and unit_phrase(3, "km") == "3 km"
    assert unit_phrase(1, "cup") == "1 cup" and unit_phrase(5, "pounds") == "5 pounds"
    assert unit_phrase(300, "kelvin") == "300 kelvin"


def test_calculator_evaluates_arithmetic() -> None:
    assert evaluate("0.15 * 80") == 12
    assert evaluate("(45 + 30) / 3") == 25
    assert evaluate("sqrt(144)") == 12
    assert evaluate("2^10") == 1024  # spoken "to the power of", never XOR
    assert run(CalculatorFlow(), expression="12 * 17") == "It's 204."
    assert run(CalculatorFlow(), expression="1 / 3") == "It's 0.333."
    assert run(CalculatorFlow(), expression="5 / 0") == "That divides by zero, so there's no answer."


@pytest.mark.parametrize(
    "expression",
    ["__import__('os').system('true')", "().__class__", "open('x')", "9 ** 9 ** 9", "factorial(10000)", "x + 1", "lambda: 1"],
)
def test_calculator_refuses_anything_but_arithmetic(expression: str) -> None:
    with pytest.raises(CalculationError):
        evaluate(expression)
    assert run(CalculatorFlow(), expression=expression) == "Sorry, I couldn't work that out."


def test_durations_and_numbers() -> None:
    assert parse_duration("PT10M") == 600 and parse_duration("PT1H30M") == 5400 and parse_duration("pt45s") == 45
    assert parse_duration("P1DT2H") == 93600 and parse_duration("PT1.5H") == 5400
    assert parse_duration("10 minutes") is None and parse_duration("P") is None
    assert duration_phrase(5400) == "1 hour and 30 minutes" and duration_phrase(3725) == "1 hour, 2 minutes, and 5 seconds"
    assert number_phrase(21.1111) == "21.1" and number_phrase(0.90718) == "0.907" and number_phrase(43560.17) == "43,560"
    assert number_phrase(16.0) == "16" and number_phrase(0) == "0" and number_phrase(-3.14159) == "-3.14"
