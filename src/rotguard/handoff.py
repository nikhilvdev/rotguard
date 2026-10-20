"""Fresh-context handoff: restart the context from the system prompt, the ledger and the tail.

Used when recall probes keep failing and compaction has not helped. Unlike the per-call ledger
injection, this replaces the history, so it is always an explicit, opt-in action.
"""

from __future__ import annotations

from rotguard.core import Message, Provenance, TextBlock, ToolCallBlock, ToolResultBlock
from rotguard.ledger import Ledger

NOTE = (
    "[rotguard handoff: the earlier conversation was dropped to give you a fresh context. "
    "The ledger below is authoritative for what was decided so far.]"
)


def _call_ids(m: Message) -> set[str]:
    return {b.id for b in m.blocks if isinstance(b, ToolCallBlock)}


def _result_ids(m: Message) -> set[str]:
    return {b.call_id for b in m.blocks if isinstance(b, ToolResultBlock)}


def tail_start(body: list[Message], keep: int, user_first: bool = True) -> int:
    """Index where the last `keep` messages of `body` start, widened backwards until the tail
    holds the tool call for every tool result it contains (providers reject a result with no
    call) and, with `user_first`, starts on a user message. With no user message before the tail,
    it starts where it is. Pass `user_first=False` when a user-role message will go in front."""
    start = max(0, len(body) - max(keep, 1))
    while True:
        moved = start
        while user_first and start > 0 and body[start].role != "user":
            if any(b.role == "user" for b in body[:start]):
                start -= 1
            else:
                break
        needed = set().union(*(_result_ids(m) for m in body[start:]))
        for i in range(start):
            if _call_ids(body[i]) & needed:
                start = min(start, i)
        if start == moved:
            return start


def pinned_closure(body: list[Message], pinned: set[str] | frozenset[str]) -> set[str]:
    """Ids of the pinned messages plus the tool call or result each one is paired with."""
    keep = {m.id for m in body if m.id in pinned}
    calls = {cid: m.id for m in body for cid in _call_ids(m)}
    results = {cid: m.id for m in body for cid in _result_ids(m)}
    for m in body:
        if m.id not in pinned:
            continue
        keep |= {calls[c] for c in _result_ids(m) if c in calls}
        keep |= {results[c] for c in _call_ids(m) if c in results}
    return keep


def handoff(
    messages: list[Message],
    ledger: Ledger,
    *,
    keep: int = 4,
    pinned: set[str] | frozenset[str] = frozenset(),
) -> list[Message]:
    """System messages, a ledger note, pinned messages, then at least the last `keep` others.

    The tail is widened as in `tail_start`. Pinned messages (and the tool call or result each one
    is paired with) from before the tail are kept, in order, ahead of it.
    """
    system = [m for m in messages if m.role == "system"]
    body = [m for m in messages if m.role != "system"]
    start = tail_start(body, keep)
    kept = pinned_closure(body[:start], pinned)
    text = ledger.render()
    note = (
        [
            Message(
                id="handoff-ledger",
                role="user",
                blocks=[TextBlock(text=f"{NOTE}\n{text}")],
                provenance=Provenance(kind="ledger"),
            )
        ]
        if text
        else []
    )
    return [*system, *note, *[m for m in body[:start] if m.id in kept], *body[start:]]
