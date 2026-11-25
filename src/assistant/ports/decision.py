"""Decision model port: a one-pass classifier in the style of Jev / "System One" models.

It answers typed questions about a piece of state with probabilities and never
generates free text, which is what makes routing deterministic and auditable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Protocol, Union


@dataclass(frozen=True)
class ChoiceQuestion:
    """Pick exactly one option. ``options`` maps a label to the text the model judges."""

    prompt: str
    options: Mapping[str, str]


@dataclass(frozen=True)
class YesNoQuestion:
    prompt: str


Question = Union[ChoiceQuestion, YesNoQuestion]


@dataclass(frozen=True)
class ChoiceAnswer:
    probabilities: Mapping[str, float]

    @property
    def best(self) -> str:
        return max(self.probabilities, key=self.probabilities.__getitem__)

    @property
    def confidence(self) -> float:
        return self.probabilities[self.best]


@dataclass(frozen=True)
class YesNoAnswer:
    p_yes: float


Answer = Union[ChoiceAnswer, YesNoAnswer]


class DecisionModel(Protocol):
    async def decide(self, state: str, questions: Mapping[str, Question]) -> Mapping[str, Answer]:
        """Answer every question about ``state``; the result is keyed like ``questions``."""
        ...
