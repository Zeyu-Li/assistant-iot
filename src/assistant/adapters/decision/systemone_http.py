"""DecisionModel over the Jev-compatible ``POST /v1/systemone`` API.

Works with laya-serve, TypeSafe, OpenRouter, or any server that speaks the same shapes:
request ``{model, state, questions: {name: {type, instructions, criteria}}}``,
response ``{answers: {name: {type, choice, probabilities} | {type, noul}}}``.
"""

from __future__ import annotations

import logging
from typing import Any, Literal, Mapping, Optional

import httpx
from pydantic import BaseModel, ConfigDict

from assistant.ports.decision import Answer, ChoiceAnswer, ChoiceQuestion, Question, YesNoAnswer

log = logging.getLogger(__name__)

# Some models over-weight the literal "true"/"false" labels of noul questions;
# asking yes/no as a two-way choice with neutral keys avoids that.
_AFFIRM, _DENY = "affirmative", "negative"


class SystemOneSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    base_url: str
    model: str
    api_key_env: Optional[str] = None
    timeout_seconds: float = 10.0
    yes_no_as: Literal["noul", "choice"] = "noul"


class SystemOneDecisionModel:
    def __init__(self, client: httpx.AsyncClient, model: str, yes_no_as: Literal["noul", "choice"] = "noul") -> None:
        self._client = client
        self._model = model
        self._yes_no_as = yes_no_as

    async def decide(self, state: str, questions: Mapping[str, Question]) -> Mapping[str, Answer]:
        body = {
            "model": self._model,
            "state": state,
            "questions": {name: self._encode(q) for name, q in questions.items()},
        }
        response = await self._client.post("/v1/systemone", json=body)
        response.raise_for_status()
        answers: Mapping[str, Any] = response.json().get("answers", {})
        return {name: self._decode(q, answers.get(name) or {}) for name, q in questions.items()}

    def _encode(self, question: Question) -> dict[str, Any]:
        if isinstance(question, ChoiceQuestion):
            return {"type": "choice", "instructions": question.prompt, "criteria": dict(question.options)}
        if self._yes_no_as == "choice":
            return {"type": "choice", "instructions": question.prompt, "criteria": {_AFFIRM: "Yes.", _DENY: "No."}}
        return {"type": "noul", "instructions": question.prompt}

    def _decode(self, question: Question, answer: Mapping[str, Any]) -> Answer:
        # A missing answer or a refusal decodes to "no confidence in anything", which every
        # caller treats as the safe outcome (no route, not addressed, not confirmed).
        if isinstance(question, ChoiceQuestion):
            probabilities = _choice_probabilities(answer)
            return ChoiceAnswer({label: probabilities.get(label, 0.0) for label in question.options})
        if self._yes_no_as == "choice":
            return YesNoAnswer(_choice_probabilities(answer).get(_AFFIRM, 0.0))
        value = answer.get("noul")
        if not isinstance(value, (int, float)):
            log.warning("no yes/no answer in %r; treating as no", answer)
            return YesNoAnswer(0.0)
        return YesNoAnswer(float(value))


def _choice_probabilities(answer: Mapping[str, Any]) -> dict[str, float]:
    probabilities = answer.get("probabilities")
    if isinstance(probabilities, Mapping) and probabilities:
        return {str(k): float(v) for k, v in probabilities.items()}
    if isinstance(answer.get("choice"), str):
        return {answer["choice"]: float(answer.get("confidence", 1.0))}
    log.warning("no choice in %r; treating as no confidence", answer)
    return {}
