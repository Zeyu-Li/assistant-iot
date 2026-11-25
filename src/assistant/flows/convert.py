"""Unit conversions with pint: lengths, weights, volumes, temperatures, speeds, areas..."""

from __future__ import annotations

import pint

from assistant.flows.speech import number_phrase
from assistant.ports.flow import FlowContext

_TEMPERATURES = {"fahrenheit": "Fahrenheit", "celsius": "Celsius", "centigrade": "Celsius", "kelvin": "kelvin"}
_IRREGULAR = {"foot": "feet", "inch": "inches"}
_PLURAL_ENDINGS = ("s", "ch", "sh", "x")


def unit_phrase(amount: float, unit: str) -> str:
    """'16 tablespoons', '1 cup', '21.1 degrees Celsius', '5 feet' - from whatever form the unit came in."""
    words = unit.lower().replace("degrees", "").replace("degree", "").split()
    number = number_phrase(amount)
    if len(words) == 1 and words[0] in _TEMPERATURES:
        name = _TEMPERATURES[words[0]]
        return f"{number} {name}" if name == "kelvin" else f"{number} degrees {name}"
    if amount == 1 or not words:
        return f"{number} {unit}"
    # Pluralise the noun before "per" ("miles per hour"), otherwise the last word.
    index = words.index("per") - 1 if "per" in words[1:] else len(words) - 1
    words[index] = _plural(words[index])
    return f"{number} {' '.join(words)}"


def _plural(word: str) -> str:
    if word in _IRREGULAR:
        return _IRREGULAR[word]
    if word in _IRREGULAR.values() or word.endswith("s") or not any(v in word for v in "aeiou"):
        return word  # already plural, or an abbreviation such as km or mph
    return word + "es" if word.endswith(_PLURAL_ENDINGS) else word + "s"


class ConvertFlow:
    def __init__(self) -> None:
        self._units = pint.UnitRegistry()

    async def run(self, context: FlowContext) -> str:
        args = context.arguments
        value = float(args.get("value", 1))
        source = str(args["from_unit"]).strip()
        target = str(args["to_unit"]).strip()
        try:
            # Quantity(value, unit), not value * unit: temperatures have offsets.
            result = self._units.Quantity(value, source).to(target).magnitude
        except pint.UndefinedUnitError as error:
            return f"I don't know the unit {error.unit_names[0] if error.unit_names else source}."
        except pint.DimensionalityError:
            return f"I can't convert {source} to {target}."
        return f"{unit_phrase(value, source)} is {unit_phrase(result, target)}."
