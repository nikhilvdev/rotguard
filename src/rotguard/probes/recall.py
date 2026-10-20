"""Recall probes (tier 3): every N turns, ask the model pointed questions about facts it was given.

Each question is a fill-in-the-blank about one fact (a ledger entry, or a fact from the
conversation that the ledger does not cover), all in one numbered side call over the context the
agent is about to use, so it never enters history. One miss is noise; an alert needs
`alert_after` failed probes in a row.
"""

from __future__ import annotations

import random
import re
from collections import deque
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

from rotguard.core import Message
from rotguard.ledger import Entry

from .caller import SideCaller
from .quiz import (
    Fact,
    build_prompt,
    cued_ledger_facts,
    drop_leaky,
    grade,
    is_correct,
    ledger_facts,
    norm,
    reply_segments,
)


@dataclass
class RecallPolicy:
    """When to probe. Each probe costs one side call."""

    mode: Literal["off", "every"] = "off"
    every: int = 5  # turns between probes
    sample: int = 3  # facts asked per probe
    style: Literal["cued", "list"] = "cued"  # one pointed question per fact, or one "list all"
    # What to ask about. "ledger": active ledger entries. "history": facts from the conversation
    # that the ledger does not cover. "auto": history when the ledger is shown to the model on
    # every call (it can always answer from it), otherwise the ledger when it has entries.
    target: Literal["auto", "ledger", "history"] = "auto"
    max_probes: int = 20  # per session
    pass_at: float = 0.5  # share of sampled entries that must come back for a pass
    alert_after: int = 2  # consecutive failed probes before an alert
    window: int = 5  # probes in the rolling score
    seed: int | None = None
    background: bool = True
    action: Literal["log", "handoff", "compact"] = "log"  # on alert: log, restart, or clean up
    handoff_keep: int = 4  # messages kept verbatim by a handoff
    signals: bool = True  # tier 1: passive signals force a probe when they flag
    marker: str | None = None  # tier 2: ask the model to begin replies with this; a miss forces one


@dataclass
class RecallResult:
    turn: int
    score: float | None  # share of sampled entries recalled; None if nothing was checkable
    asked: list[str] = field(default_factory=list)
    missed: list[str] = field(default_factory=list)
    stale: list[str] = field(default_factory=list)  # answered with a value that was replaced
    side_calls: int = 1

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _first(text: str, tokens: list[str]) -> int | None:
    """Position of the earliest whole-token occurrence of any of `tokens`, or None."""
    spots = [
        m.start()
        for t in tokens
        if (m := re.search(rf"(?<!\w){re.escape(t)}(?!\w)", text)) is not None
    ]
    return min(spots) if spots else None


def _is_stale(f: Fact, answer: str, correct: bool) -> bool:
    """The answer gives a replaced value instead of, or ahead of, the current one."""
    text = norm(answer)
    old = _first(text, f.stale)
    if old is None:
        return False
    now = _first(text, f.tokens or [norm(f.answer)]) if correct else None
    return now is None or old < now


def run_recall(
    *,
    context: list[Message],
    entries: list[Entry],
    caller: SideCaller,
    sample: int = 3,
    rng: random.Random | None = None,
    turn: int = 0,
    style: Literal["cued", "list"] = "cued",
    facts: list[Fact] | None = None,
) -> RecallResult:
    """Ask about `sample` facts and grade the reply per fact.

    `facts` are pre-built cued facts to draw from (the history target); without them the facts
    come from the ledger `entries`, which should be the whole ledger so replaced entries can be
    used to spot stale answers. `style="list"` asks the old single "list everything" question.
    """
    rng = rng or random.Random()
    if facts is None and style == "list":
        led = ledger_facts([e for e in entries if e.status == "active"])
        if sample < len(led):
            led = rng.sample(led, sample)
        if not led:
            return RecallResult(turn, None, side_calls=0)
        got = grade(led, True, caller.ask(context, build_prompt([], True)))
        missed = [f.id for f in led if not got[f.id]]
        return RecallResult(turn, 1 - len(missed) / len(led), [f.id for f in led], missed)
    pool = list(facts) if facts is not None else cued_ledger_facts(entries)
    if sample < len(pool):
        pool = rng.sample(pool, sample)
    chosen = drop_leaky(pool)  # a question that contains another's answer would give it away
    if not chosen:
        return RecallResult(turn, None, side_calls=0)
    text = reply_segments(chosen, False, caller.ask(context, build_prompt(chosen, False)))
    right = {f.id: is_correct(f, text[f.id]) for f in chosen}
    stale = [f.id for f in chosen if _is_stale(f, text[f.id], right[f.id])]
    missed = [f.id for f in chosen if not right[f.id] or f.id in stale]
    return RecallResult(turn, 1 - len(missed) / len(chosen), [f.id for f in chosen], missed, stale)


class RecallMonitor:
    """Decides when a probe is due and when failures add up to an alert."""

    def __init__(self, policy: RecallPolicy):
        self.policy = policy
        self.rng = random.Random(policy.seed)
        self.started, self.streak, self._last = 0, 0, -policy.every
        self._forced = False
        self._scores: deque[float] = deque(maxlen=max(1, policy.window))

    def due(self, turn: int) -> bool:
        p = self.policy
        if p.mode == "off" or self.started >= p.max_probes:
            return False
        if turn - self._last < p.every and not self._forced:
            return False
        self._last, self.started, self._forced = turn, self.started + 1, False
        return True

    def flag(self) -> None:
        """A free tier saw something: run the next probe now (the budget still applies)."""
        self._forced = True

    @property
    def rolling(self) -> float | None:
        return sum(self._scores) / len(self._scores) if self._scores else None

    def record(self, r: RecallResult) -> bool:
        """Add a result; True when this one completes a run of `alert_after` failures."""
        if r.score is None:
            return False
        self._scores.append(r.score)
        failed = r.score < self.policy.pass_at
        self.streak = self.streak + 1 if failed else 0
        return failed and self.streak == self.policy.alert_after


def calibrate(clean_scores: list[float], floor: float = 0.3, ceiling: float = 1.0) -> float:
    """A `pass_at` for one model from probe scores taken on contexts known to be rot-free.

    A model that restates only 70% of entries even with a fresh context would fail every probe at
    `pass_at=0.9`; set the bar just under its clean performance (the lower quartile), within
    [floor, ceiling]. Needs a handful of clean probes to mean anything.
    """
    if not clean_scores:
        return 0.5
    ordered = sorted(clean_scores)
    low = ordered[len(ordered) // 4]
    return min(ceiling, max(floor, low - 1e-9))
