"""CI helpers: fail a test run when a compaction lost too much or broke the history."""

from __future__ import annotations

from typing import Any

from rotguard.core import Message, ToolCallBlock, ToolResultBlock
from rotguard.store import Store


def audit_reports(store: Store, session_id: str) -> list[dict[str, Any]]:
    return store.events(session_id, "audit")


def assert_audit_loss(
    store: Store, session_id: str, max_loss: float = 0.3, require_audit: bool = True
) -> None:
    """Raise AssertionError if any audited compaction lost more than `max_loss` of the facts the
    model could answer before it, if an audit crashed, or (by default) if no audit ran at all."""
    problems = [f"audit failed: {e['error']}" for e in store.events(session_id, "audit_error")]
    reports = audit_reports(store, session_id)
    if require_audit and not reports and not problems:
        problems.append("no compaction was audited (is the policy mode 'ci' or 'every'?)")
    for r in reports:
        if r["loss"] is not None and r["loss"] > max_loss:
            lost = ", ".join(f"{x['answer']!r} ({x['origin_kind']})" for x in r["lost"])
            problems.append(
                f"turn {r['turn']}: loss {r['loss']:.2f} > {max_loss:.2f}; lost: {lost}"
            )
    if problems:
        raise AssertionError("\n".join(problems))


def broken_pairs(messages: list[Message]) -> list[str]:
    """Tool results with no earlier call, and tool calls with no result (a call in the final
    message is fine: its result has not been produced yet). Providers reject both."""
    calls: dict[str, int] = {}
    problems: list[str] = []
    answered: set[str] = set()
    for i, m in enumerate(messages):
        for b in m.blocks:
            if isinstance(b, ToolCallBlock):
                calls[b.id] = i
            elif isinstance(b, ToolResultBlock):
                if b.call_id not in calls:
                    problems.append(f"tool result {b.call_id!r} has no earlier tool call")
                answered.add(b.call_id)
    last = len(messages) - 1
    problems += [
        f"tool call {cid!r} has no result"
        for cid, i in calls.items()
        if cid not in answered and i < last
    ]
    return problems


def assert_history_integrity(messages: list[Message]) -> None:
    """Raise AssertionError if a tool call and its result are not paired."""
    problems = broken_pairs(messages)
    if problems:
        raise AssertionError("\n".join(problems))
