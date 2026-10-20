"""Replay a saved session against a model with different audit and recall settings.

The contexts are rebuilt from the source store's snapshots, one per turn, and fed through a fresh
`Session`. The agent's own calls are not re-run: only rotguard's side calls are, so the same trace
can be scored under another model, policy or ledger setting. Results go to a new store.
"""

from __future__ import annotations

import uuid
from collections.abc import Callable
from dataclasses import replace

from rotguard.core import Message, Provenance, TextBlock
from rotguard.ledger import Entry, Ledger
from rotguard.probes import AuditPolicy, RecallPolicy, RecallResult, SideCaller
from rotguard.session import Session
from rotguard.store import SQLiteStore, Store


def _message(row: dict) -> Message:  # type: ignore[type-arg]
    return Message(
        id=row["id"],
        role=row["role"],
        blocks=[TextBlock(text=row["text"])],
        provenance=Provenance(
            kind=row["kind"],
            turn=row["turn"],
            tool_name=row["tool_name"],
            call_id=row["call_id"],
            parents=row["parents"],
        ),
    )


def replay(
    src: Store,
    session: str,
    caller: SideCaller,
    *,
    audit: AuditPolicy | None = None,
    recall: RecallPolicy | None = None,
    inject_ledger: bool = False,
    dst: Store | None = None,
    on_alert: Callable[[RecallResult], None] | None = None,
) -> tuple[Store, str]:
    """Replay `session`; returns the store holding the new events and the new session id.

    Audits always run inline here (a replay is a batch job), whatever the policy says.
    """
    dst = dst or SQLiteStore()
    new_id = f"replay-{session}-{uuid.uuid4().hex[:6]}"
    rows = {r["id"]: r for r in src.items(session)}
    audit = replace(audit or AuditPolicy(mode="off"), background=False)
    recall = replace(recall, background=False) if recall else None
    ledger = Ledger(on_change=lambda e: dst.upsert_entry(new_id, e.model_dump()))
    entries = [Entry(**e) for e in src.entries(session)]
    ledger.load(entries)
    for e in entries:  # `load` is silent, so write them to the new store for its report
        dst.upsert_entry(new_id, e.model_dump())
    s = Session(dst, new_id, ledger, audit, recall, on_alert)
    last_turn = max([r["turn"] for r in rows.values()] + [0])
    previous: list[str] | None = None
    for turn in range(1, last_turn + 1):
        ids = src.snapshot_at(session, turn)
        if ids == previous or not ids:
            continue
        previous, s.turn = ids, turn
        s.observe_history([_message(rows[i]) for i in ids if i in rows])
        s.run_pending(caller, ledger.render() if inject_ledger else "")
    s.close()
    return dst, new_id
