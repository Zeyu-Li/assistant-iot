"""Turn-by-turn dialogue: route, fill slots, confirm writes, execute.

The small LLM only extracts arguments and phrases results. Every decision that
changes what happens (which capability, did the user agree) goes through the
decision model or deterministic code.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Optional

from assistant.core.executor import CapabilityExecutor
from assistant.core.registry import Capability, Registry
from assistant.core.router import Router
from assistant.core.slots import PreviousRequest, SlotExtractor, invalid_fields, missing_required
from assistant.ports.decision import ChoiceAnswer, ChoiceQuestion, DecisionModel

log = logging.getLogger(__name__)

CANCEL_PHRASES = frozenset({"cancel", "never mind", "nevermind", "forget it", "stop", "forget about it"})
# Asked as an explicit two-way choice rather than a yes/no: spelling out that hesitation
# counts as declining separated every "yes" (>= 0.70) from every "no"/"um" (<= 0.32) on
# OpenJev, where a bare yes/no question scored a plain "yes" at 0.37.
CONFIRM_QUESTION = ChoiceQuestion(
    "Did the user agree to what the assistant asked?",
    {
        "agree": "Yes - the user clearly agreed or told it to go ahead.",
        "decline": "No - the user refused, corrected it, hesitated, or was unsure.",
    },
)


@dataclass(frozen=True)
class DialogueTexts:
    cannot_help: str = "Sorry, I can't help with that."
    cancelled: str = "Okay, cancelled."
    not_done: str = "Okay, I won't."
    gave_up: str = "Sorry, I didn't catch that. Let's try again later."
    confirm_fallback: str = "Should I go ahead?"


@dataclass(frozen=True)
class Reply:
    text: str
    expects_response: bool = False
    """True when the assistant asked a question and is waiting for the answer."""


@dataclass
class _Task:
    capability: Capability
    schema: Mapping[str, Any]
    request: str
    arguments: dict[str, Any] = field(default_factory=dict)
    question: Optional[str] = None
    """The last question put to the user, given to the extractor as context."""
    asking_for: Optional[str] = None
    reprompts: int = 0
    awaiting_confirmation: bool = False
    origin: Optional[str] = None
    """For a follow-up, the full request it follows up on."""


@dataclass(frozen=True)
class _Done:
    """The last completed request of a follow-up capability, kept briefly for "what about ...?"."""

    capability: Capability
    request: str
    arguments: Mapping[str, Any]
    at: float


class DialogueManager:
    def __init__(
        self,
        registry: Registry,
        router: Router,
        extractor: SlotExtractor,
        executor: CapabilityExecutor,
        decider: DecisionModel,
        *,
        confirm_threshold: float,
        max_reprompts: int,
        follow_up_threshold: float = 0.6,
        follow_up_seconds: float = 60.0,
        clock: Callable[[], float] = time.monotonic,
        texts: DialogueTexts = DialogueTexts(),
    ) -> None:
        self._registry = registry
        self._router = router
        self._extractor = extractor
        self._executor = executor
        self._decider = decider
        self._confirm_threshold = confirm_threshold
        self._max_reprompts = max_reprompts
        self._follow_up_threshold = follow_up_threshold
        self._follow_up_seconds = follow_up_seconds
        self._clock = clock
        self._texts = texts
        self._pending: Optional[_Task] = None
        self._last_done: Optional[_Done] = None

    @property
    def in_progress(self) -> bool:
        return self._pending is not None

    def reset(self) -> None:
        """Drop any half-finished task and the follow-up context."""
        self._pending = None
        self._last_done = None

    async def handle(
        self, utterance: str, *, require_addressed: bool = False, quiet_if_unmatched: bool = False
    ) -> Optional[Reply]:
        """Return what to say, or None to stay silent.

        ``require_addressed`` also asks the decision model whether the speech was meant for
        the assistant; ``quiet_if_unmatched`` stays silent instead of apologising when no
        capability fits (open-mic speech that was probably not a request).
        """
        text = utterance.strip()
        if not text:
            return None
        if self._pending is not None:
            return await self._continue(self._pending, text)
        return await self._start(text, require_addressed, quiet_if_unmatched or require_addressed)

    async def _start(self, text: str, require_addressed: bool, quiet_if_unmatched: bool) -> Optional[Reply]:
        follow_up = await self._as_follow_up(text)
        if follow_up is not None:
            return await self._advance(follow_up)

        decision = await self._router.route(text, require_addressed=require_addressed)
        log.info(
            "route=%s confidence=%.2f addressed=%s -> %s",
            decision.best_label, decision.confidence, decision.p_addressed, decision.capability_id,
        )
        if decision.capability_id is None:
            return None if quiet_if_unmatched else Reply(self._texts.cannot_help)

        capability = self._registry.get(decision.capability_id)
        schema = await self._executor.schema_for(capability)
        extracted = await self._extractor.extract(text, schema)
        task = _Task(capability, schema, request=text, arguments={**capability.defaults, **extracted})
        return await self._advance(task)

    async def _as_follow_up(self, text: str) -> Optional[_Task]:
        """The previous request with this utterance's changes, if it only changes a detail of it."""
        last = self._last_done
        if last is None or self._clock() - last.at > self._follow_up_seconds:
            return None
        p_follow_up = await self._router.follow_up_probability(last.request, text)
        log.info("follow-up of %s: p=%.2f", last.capability.id, p_follow_up)
        if p_follow_up < self._follow_up_threshold:
            return None
        schema = await self._executor.schema_for(last.capability)
        changes = await self._extractor.extract(text, schema, previous=PreviousRequest(last.request, last.arguments))
        if not changes:
            return None  # nothing to change, so it was not a follow-up after all ("thanks")
        return _Task(last.capability, schema, request=text, arguments={**last.arguments, **changes}, origin=last.request)

    async def _continue(self, task: _Task, text: str) -> Reply:
        if _normalise(text) in CANCEL_PHRASES:
            self._pending = None
            return Reply(self._texts.cancelled)
        if task.awaiting_confirmation:
            self._pending = None
            if not await self._agreed(task, text):
                return Reply(self._texts.not_done)
            return Reply(await self._execute(task))
        task.arguments.update(await self._extractor.extract(text, task.schema, question=task.question))
        return await self._advance(task)

    async def _execute(self, task: _Task) -> str:
        result = await self._executor.execute(task.capability, task.arguments, task.request)
        self._last_done = (
            _Done(task.capability, task.origin or task.request, dict(task.arguments), self._clock())
            if task.capability.follow_up
            else None
        )
        return result

    async def _advance(self, task: _Task) -> Reply:
        for name in invalid_fields(task.schema, task.arguments):
            log.info("dropping invalid value for %s: %r", name, task.arguments[name])
            del task.arguments[name]

        missing = missing_required(task.schema, task.arguments)
        if missing:
            return self._ask_for(task, missing[0])

        if task.capability.needs_confirmation(task.arguments):
            task.awaiting_confirmation = True
            task.question = self._confirm_prompt(task)
            self._pending = task
            return Reply(task.question, expects_response=True)

        self._pending = None
        return Reply(await self._execute(task))

    def _ask_for(self, task: _Task, slot: str) -> Reply:
        task.reprompts = task.reprompts + 1 if task.asking_for == slot else 0
        if task.reprompts > self._max_reprompts:
            self._pending = None
            return Reply(self._texts.gave_up)
        task.asking_for = slot
        task.question = task.capability.prompts.get(slot) or f"What should the {slot.replace('_', ' ')} be?"
        self._pending = task
        return Reply(task.question, expects_response=True)

    async def _agreed(self, task: _Task, text: str) -> bool:
        answers = await self._decider.decide(
            f"The assistant asked: {task.question}\nThe user replied: {text}", {"confirm": CONFIRM_QUESTION}
        )
        answer = answers["confirm"]
        assert isinstance(answer, ChoiceAnswer)
        p_agree = answer.probabilities.get("agree", 0.0)
        log.info("confirm p_agree=%.2f", p_agree)
        return p_agree >= self._confirm_threshold

    def _confirm_prompt(self, task: _Task) -> str:
        described = self._executor.describe(task.capability, task.arguments)
        if described:
            return described
        template = task.capability.confirm_prompt
        if not template:
            return self._texts.confirm_fallback
        try:
            return template.format_map(_BlankDefault(task.arguments))
        except (KeyError, IndexError, AttributeError, ValueError):
            return self._texts.confirm_fallback


class _BlankDefault(dict):  # type: ignore[type-arg]
    def __missing__(self, key: str) -> str:
        return ""


def _normalise(text: str) -> str:
    return re.sub(r"[^a-z' ]", "", text.lower()).strip()
