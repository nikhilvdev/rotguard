"""Recall probes (tier 3): every N turns, ask the model to restate sampled ledger entries.

The probe is a side call over the context the agent is about to use, so it never enters history.
One miss is noise; an alert needs `alert_after` failed probes in a row.
"""

from __future__ import annotations

import random
from collections import deque
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

from rotguard.core import Message
from rotguard.ledger import Entry

from .caller import SideCaller
from .quiz import build_prompt, grade, ledger_facts


@dataclass
class RecallPolicy:
    """When to probe. Each probe costs one side call."""

    mode: Literal["off", "every"] = "off"
    every: int = 5  # turns between probes
    sample: int = 3  # ledger entries asked per probe
    max_probes: int = 20  # per session
    pass_at: float = 0.5  # share of sampled entries that must come back for a pass
    alert_after: int = 2  # consecutive failed probes before an alert
    window: int = 5  # probes in the rolling score
    seed: int | None = None
    background: bool = True
    action: Literal["log", "handoff"] = "log"  # on alert: log only, or restart the context
    handoff_keep: int = 4  # messages kept verbatim by a handoff
    signals: bool = True  # tier 1: passive signals force a probe when they flag
    marker: str | None = None  # tier 2: ask the model to begin replies with this; a miss forces one


@dataclass
class RecallResult:
    turn: int
    score: float | None  # share of sampled entries recalled; None if nothing was checkable
    asked: list[str] = field(default_factory=list)
    missed: list[str] = field(default_factory=list)
    side_calls: int = 1

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def run_recall(
    *,
    context: list[Message],
    entries: list[Entry],
    caller: SideCaller,
    sample: int = 3,
    rng: random.Random | None = None,
    turn: int = 0,
) -> RecallResult:
    """Ask for `sample` random active ledger entries and grade the reply per entry."""
    facts = ledger_facts([e for e in entries if e.status == "active"])
    if sample < len(facts):
        facts = (rng or random.Random()).sample(facts, sample)
    if not facts:
        return RecallResult(turn, None, side_calls=0)
    got = grade(facts, True, caller.ask(context, build_prompt([], True)))
    missed = [f.id for f in facts if not got[f.id]]
    return RecallResult(turn, 1 - len(missed) / len(facts), [f.id for f in facts], missed)


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
