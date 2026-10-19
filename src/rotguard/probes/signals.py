"""Tier 1 recall check: passive signals that cost no model call.

Each signal is a cheap hint that the agent may be drifting. A flag does not mean rot; it makes the
paid recall probe (tier 3) run now instead of waiting for its schedule.

shortcut: goal overlap is lexical (shared words), not embedding similarity, so it needs no extra
dependency; swap in embeddings if goals and replies are routinely phrased differently.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from rotguard.core import Message
from rotguard.ledger import Entry

from .quiz import norm

_WORD = re.compile(r"[a-z0-9_]{4,}")
_ERROR = re.compile(r"\b(error|exception|traceback|failed|failure|timed? ?out)\b", re.I)


@dataclass
class Signals:
    goal_overlap: float | None = None  # share of goal words seen in recent messages
    repetition: float = 0.0  # share of recent assistant replies that repeat an earlier one
    tool_error_rate: float = 0.0  # share of recent tool results that look like errors
    reasons: list[str] = field(default_factory=list)

    @property
    def flag(self) -> bool:
        return bool(self.reasons)


def _kind(m: Message) -> str:
    return m.provenance.kind if m.provenance else m.role


def passive_signals(
    context: list[Message],
    entries: list[Entry],
    *,
    window: int = 6,
    min_goal_overlap: float = 0.15,
    max_repetition: float = 0.5,
    max_tool_errors: float = 0.5,
) -> Signals:
    """Look at the last `window` messages of each kind. Needs a few messages before it flags."""
    out = Signals()
    talk = [m for m in context if _kind(m) in ("user", "assistant")][-window:]
    goals = {
        w
        for e in entries
        if e.status == "active" and e.kind == "goal"
        for w in _WORD.findall(norm(e.text))
    }
    if goals and len(talk) >= window:
        seen = set(_WORD.findall(norm(" ".join(m.text for m in talk))))
        out.goal_overlap = len(goals & seen) / len(goals)
        if out.goal_overlap < min_goal_overlap:
            out.reasons.append("goal_drift")
    replies = [norm(m.text) for m in context if _kind(m) == "assistant"][-window:]
    if len(replies) >= 3:
        out.repetition = 1 - len(set(replies)) / len(replies)
        if out.repetition >= max_repetition:
            out.reasons.append("repetition")
    tools = [m for m in context if _kind(m) == "tool"][-window:]
    if len(tools) >= 3:
        out.tool_error_rate = sum(bool(_ERROR.search(m.text)) for m in tools) / len(tools)
        if out.tool_error_rate >= max_tool_errors:
            out.reasons.append("tool_errors")
    return out
