"""Pick a capability for an utterance with one decision-model pass."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from assistant.core.registry import NONE_ID, Registry
from assistant.ports.decision import (
    ChoiceAnswer,
    ChoiceQuestion,
    DecisionModel,
    Question,
    YesNoAnswer,
    YesNoQuestion,
)

ROUTE_PROMPT = "Which capability of a home voice assistant should handle what the user just said?"
# The label and wording of the "nothing fits" option matter a lot: an abstract "none of the
# other options" absorbed most in-scope requests in evaluation, while concrete examples of
# out-of-scope requests under a neutral label did not.
OTHER_LABEL = "other"
OTHER_OPTION = (
    "Something else the assistant cannot do, such as smart-home control, shopping, sending "
    "messages, news or prices, or the speaker talking to another person."
)
ADDRESSED_PROMPT = (
    "Is the speaker talking to the voice assistant, giving it a command or asking it a question, "
    "rather than talking to someone else in the room?"
)


@dataclass(frozen=True)
class RouterThresholds:
    route: float
    addressed: float


@dataclass(frozen=True)
class RouteDecision:
    capability_id: Optional[str]
    """The chosen capability, or None when nothing cleared the thresholds."""
    best_label: str
    confidence: float
    p_addressed: Optional[float] = None


class Router:
    def __init__(
        self,
        decider: DecisionModel,
        registry: Registry,
        thresholds: RouterThresholds,
        *,
        include_examples: bool = True,
    ) -> None:
        self._decider = decider
        self._thresholds = thresholds
        options = {c.id: c.option_text if include_examples else c.description for c in registry}
        options[OTHER_LABEL] = OTHER_OPTION
        self._route_question = ChoiceQuestion(ROUTE_PROMPT, options)

    async def route(self, utterance: str, *, require_addressed: bool) -> RouteDecision:
        questions: dict[str, Question] = {"route": self._route_question}
        if require_addressed:
            questions["addressed"] = YesNoQuestion(ADDRESSED_PROMPT)
        answers = await self._decider.decide(f"The user said: {utterance}", questions)

        route = answers["route"]
        assert isinstance(route, ChoiceAnswer)
        p_addressed: Optional[float] = None
        if require_addressed:
            addressed = answers["addressed"]
            assert isinstance(addressed, YesNoAnswer)
            p_addressed = addressed.p_yes

        best = NONE_ID if route.best == OTHER_LABEL else route.best
        accepted = (
            best != NONE_ID
            and route.confidence >= self._thresholds.route
            and (p_addressed is None or p_addressed >= self._thresholds.addressed)
        )
        return RouteDecision(
            capability_id=best if accepted else None,
            best_label=best,
            confidence=route.confidence,
            p_addressed=p_addressed,
        )
