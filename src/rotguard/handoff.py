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


def handoff(messages: list[Message], ledger: Ledger, *, keep: int = 4) -> list[Message]:
    """System messages, a ledger note, then at least the last `keep` other messages.

    The tail is widened backwards until it starts on a user message and holds the tool call for
    every tool result it contains, because providers reject a result with no call. If the history
    has no user message before the tail, the tail starts where it is.
    """
    system = [m for m in messages if m.role == "system"]
    body = [m for m in messages if m.role != "system"]
    start = max(0, len(body) - max(keep, 1))
    while True:
        moved = start
        while start > 0 and body[start].role != "user":
            if any(b.role == "user" for b in body[:start]):
                start -= 1
            else:
                break
        needed = set().union(*(_result_ids(m) for m in body[start:]))
        for i in range(start):
            if _call_ids(body[i]) & needed:
                start = min(start, i)
        if start == moved:
            break
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
    return [*system, *note, *body[start:]]
