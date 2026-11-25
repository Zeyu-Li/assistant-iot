"""Measure routing accuracy on a labelled set and sweep the acceptance threshold."""

from __future__ import annotations

import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import yaml

from assistant.core.registry import NONE_ID, Registry
from assistant.core.router import Router, RouterThresholds
from assistant.ports.decision import DecisionModel


@dataclass(frozen=True)
class Example:
    text: str
    expect: str
    also_ok: frozenset[str] = frozenset()


@dataclass(frozen=True)
class Scored:
    example: Example
    best: str
    confidence: float
    seconds: float


def load_examples(path: Path, registry: Registry) -> list[Example]:
    raw: dict[str, list[Any]] = yaml.safe_load(path.read_text())
    enabled = {c.id for c in registry}
    examples = []
    for label, items in raw.items():
        # Utterances for a capability this profile disables are out of scope: they must route to none.
        expect = label if label in enabled else NONE_ID
        for item in items:
            text, also_ok = (item, ()) if isinstance(item, str) else (item["text"], item.get("also_ok", ()))
            examples.append(Example(text, expect, frozenset(also_ok) & enabled))
    return examples


async def score(decider: DecisionModel, registry: Registry, examples: Iterable[Example], include_examples: bool) -> list[Scored]:
    # Threshold 0 so we see the raw best label and confidence; thresholds are applied afterwards.
    router = Router(decider, registry, RouterThresholds(route=0.0, addressed=0.0), include_examples=include_examples)
    results = []
    for example in examples:
        start = time.perf_counter()
        decision = await router.route(example.text, require_addressed=False)
        results.append(Scored(example, decision.best_label, decision.confidence, time.perf_counter() - start))
    return results


def outcome(s: Scored, threshold: float) -> str:
    chosen = s.best if s.best != NONE_ID and s.confidence >= threshold else NONE_ID
    if chosen == s.example.expect or chosen in s.example.also_ok:
        return "correct"
    if s.example.expect == NONE_ID:
        return "false_accept"  # acted on something it should have declined
    if chosen == NONE_ID:
        return "missed"  # declined something it could do
    return "wrong_route"


def report(results: Sequence[Scored], current_threshold: float) -> str:
    lines = []
    in_scope = [s for s in results if s.example.expect != NONE_ID]
    out_scope = [s for s in results if s.example.expect == NONE_ID]
    latencies = sorted(s.seconds for s in results)
    lines.append(
        f"{len(results)} utterances ({len(in_scope)} in scope, {len(out_scope)} out of scope); "
        f"latency p50 {latencies[len(latencies) // 2] * 1000:.0f} ms, max {latencies[-1] * 1000:.0f} ms"
    )
    lines.append("")
    lines.append("threshold  correct  wrong_route  missed  false_accept")
    for threshold in sorted({round(t / 20, 2) for t in range(0, 20)} | {current_threshold}):
        counts = Counter(outcome(s, threshold) for s in results)
        marker = "  <- profile" if threshold == current_threshold else ""
        lines.append(
            f"   {threshold:4.2f}   {counts['correct'] / len(results):6.1%}  {counts['wrong_route']:>11}  "
            f"{counts['missed']:>6}  {counts['false_accept']:>12}{marker}"
        )
    lines.append("")
    lines.append(f"errors at the profile threshold ({current_threshold}):")
    for s in results:
        result = outcome(s, current_threshold)
        if result != "correct":
            lines.append(f"  {result:<12} expected {s.example.expect:<22} got {s.best:<22} {s.confidence:.2f}  {s.example.text}")
    return "\n".join(lines)
