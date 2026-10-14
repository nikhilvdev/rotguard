"""CI helpers: fail a test run when a compaction lost too much."""

from __future__ import annotations

from typing import Any

from rotguard.store import SQLiteStore


def audit_reports(store: SQLiteStore, session_id: str) -> list[dict[str, Any]]:
    return store.events(session_id, "audit")


def assert_audit_loss(
    store: SQLiteStore, session_id: str, max_loss: float = 0.3, require_audit: bool = True
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
