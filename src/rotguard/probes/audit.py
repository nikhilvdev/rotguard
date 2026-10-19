"""Compaction audit: ask the same model the same questions with the old and the new context."""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

from rotguard.core import Message
from rotguard.ledger import Entry

from .caller import SideCaller
from .quiz import (
    Fact,
    build_prompt,
    cloze_facts,
    drop_leaky,
    extract_facts,
    grade,
    ledger_facts,
    norm,
)


@dataclass
class AuditPolicy:
    """When to audit a compaction. Each audit costs about three side calls."""

    mode: Literal["off", "sampled", "every", "ci"] = "sampled"
    sample_rate: float = 0.5
    max_facts: int = 5
    cloze: int = 2  # fallback facts built from the text when extraction finds fewer than max_facts
    max_audits: int = 20  # per session
    ci_max_loss: float = 0.3  # used by rotguard.testing.assert_audit_loss
    seed: int | None = None
    background: bool = True  # "ci" mode always runs synchronously

    def gate(self) -> AuditGate:
        return AuditGate(self)


class AuditGate:
    def __init__(self, policy: AuditPolicy):
        self.policy, self._rng, self.started = policy, random.Random(policy.seed), 0

    def allow(self) -> bool:
        p = self.policy
        if p.mode == "off" or self.started >= p.max_audits:
            return False
        if p.mode == "sampled" and self._rng.random() >= p.sample_rate:
            return False
        self.started += 1
        return True


@dataclass
class AuditReport:
    # lost / answerable-before over facts that did not come from tool output; None if the old
    # context answered none of them. Tool output is mostly noise (ids, listings) that no summary
    # keeps, so it dilutes the score; see `loss_all` and the `lost` list for those facts.
    loss: float | None
    facts_total: int
    answerable_before: int
    lost: list[dict[str, Any]] = field(default_factory=list)
    checked: list[dict[str, Any]] = field(default_factory=list)  # every fact, with before/after
    loss_all: float | None = None  # the same ratio over every fact, tool-origin included
    suggestions: list[str] = field(default_factory=list)
    side_calls: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def suggest(lost: list[Fact], loss: float | None) -> list[str]:
    out = []
    if any(f.source == "ledger" for f in lost):
        out.append(
            "A ledger entry was lost: turn on ledger injection, or pin it in the system prompt."
        )
    if any(f.origin_kind == "user" for f in lost):
        out.append(
            "User-stated facts were lost: record them with ledger_record, or change the "
            "summary prompt to preserve user instructions verbatim."
        )
    if any(f.origin_kind == "tool" for f in lost):
        out.append(
            "Tool-output details were lost: raise `keep`, or offload large tool outputs "
            "and fetch them back on demand."
        )
    if loss is not None and loss >= 0.5:
        out.append(
            "The summary dropped most checked facts: raise `keep` or use a stricter summary prompt."
        )
    return out


def run_audit(
    *,
    old: list[Message],
    new: list[Message],
    removed: list[Message],
    entries: list[Entry],
    caller: SideCaller,
    max_facts: int = 5,
    cloze: int = 0,
    prefix: str = "",
) -> AuditReport:
    """Quiz the model on `old` and on `new`; a fact counts as lost if only `old` answers it.

    `prefix` is text the model sees on every call (the injected ledger). It is put in front of the
    quiz for both contexts, so the report shows the loss that is left after rotguard's own help.
    Ledger entries are then guaranteed by construction and are not checked; without a prefix they
    are checked like any other fact.

    The report compares one compaction with the context just before it. A summary of an
    already-lossy summary has little left to lose, so repeated compactions report a low loss
    (or None) even though the cumulative loss is high.
    """
    span = extract_facts(removed, caller, max_facts)
    # Small models fill the list with tool-output listings, which `loss` ignores anyway and which
    # crowd out the user-stated facts a later step depends on. Keep one so tool loss still shows.
    tool = [f for f in span if f.origin_kind == "tool"]
    span = [f for f in span if f.origin_kind != "tool"] + tool[:1]
    calls = 1 if removed and max_facts > 0 else 0
    if cloze and len(span) < max_facts:
        span += cloze_facts(
            removed, min(cloze, max_facts - len(span)), {norm(f.answer) for f in span}
        )
    span = drop_leaky(span)
    led = [] if prefix else ledger_facts(entries)
    facts = [*led, *span]
    if not facts:
        return AuditReport(
            None,
            0,
            0,
            suggestions=["Nothing checkable was found in the removed span."],
            side_calls=calls,
        )
    with_ledger = bool(led)
    prompt = build_prompt(span, with_ledger)
    prompt = f"{prefix}\n\n{prompt}" if prefix else prompt
    before = grade(facts, with_ledger, caller.ask(old, prompt))
    after = grade(facts, with_ledger, caller.ask(new, prompt))
    answerable = [f for f in facts if before[f.id]]
    lost = [f for f in answerable if not after[f.id]]
    loss_all = len(lost) / len(answerable) if answerable else None
    scored = [f for f in answerable if f.origin_kind != "tool"]
    scored_lost = [f for f in scored if not after[f.id]]
    loss = len(scored_lost) / len(scored) if scored else None

    def row(f: Fact) -> dict[str, Any]:
        return {
            "id": f.id,
            "source": f.source,
            "answer": f.answer,
            "origin_id": f.origin_id,
            "origin_kind": f.origin_kind,
            "before": before[f.id],
            "after": after[f.id],
        }

    checked = [row(f) for f in facts]
    return AuditReport(
        loss,
        len(facts),
        len(answerable),
        [r for r in checked if r["before"] and not r["after"]],
        checked,
        loss_all,
        suggest(lost, loss),
        calls + 2,
    )
