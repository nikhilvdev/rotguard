"""Observer: diffs consecutive contexts, assigns provenance, spots compactions by any tool."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from .messages import Message, default_provenance


@dataclass
class CompactionEvent:
    channel: str
    turn: int
    removed_ids: list[str]
    rewritten_ids: list[str]
    summary_ids: list[str]
    chars_before: int
    chars_after: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Observation:
    turn: int
    added: list[Message] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    rewritten: list[str] = field(default_factory=list)
    compaction: CompactionEvent | None = None


class Observer:
    """Watches one view of the context (`channel`): the stored history, or the final request.

    A compaction is any call where items vanished (summarizing, trimming) or kept their id but
    changed content (clearing tool output). Summary items added in the same step get the
    removed items as parents.
    """

    def __init__(self, channel: str = "history"):
        self.channel = channel
        self._hashes: dict[str, str] = {}
        self._chars = 0

    def observe(self, messages: list[Message], turn: int) -> Observation:
        cur = {m.id: m.content_hash for m in messages}
        removed = [i for i in self._hashes if i not in cur]
        rewritten = [i for i, h in cur.items() if i in self._hashes and self._hashes[i] != h]
        added = [m for m in messages if m.id not in self._hashes]

        summary_ids = []
        for m in added:
            m.provenance = m.provenance or default_provenance(m, turn)
            m.provenance.turn = turn
            if m.provenance.kind == "summary":
                summary_ids.append(m.id)
                m.provenance.parents = list(removed)

        chars = sum(len(m.text) for m in messages)
        compaction = None
        if removed or rewritten:
            compaction = CompactionEvent(
                self.channel, turn, removed, rewritten, summary_ids, self._chars, chars
            )
        self._hashes, self._chars = cur, chars
        return Observation(turn, added, removed, rewritten, compaction)
