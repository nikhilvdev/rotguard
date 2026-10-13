"""diff and blame over the provenance recorded in a store."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from rotguard.store import SQLiteStore


def _norm(s: str) -> str:
    return " ".join(s.lower().split())


@dataclass
class Diff:
    added: list[str]
    removed: list[str]
    rewritten: dict[str, list[str]] = field(default_factory=dict)  # summary id -> items it replaced


def diff(store: SQLiteStore, session: str, turn_a: int, turn_b: int) -> Diff:
    a, b = set(store.snapshot_at(session, turn_a)), store.snapshot_at(session, turn_b)
    items = {i["id"]: i for i in store.items(session)}
    added = [i for i in b if i not in a]
    removed = [i for i in a if i not in set(b)]
    rewritten = {
        i: [p for p in items[i]["parents"] if p in a]
        for i in added
        if items[i]["kind"] == "summary" and items[i]["parents"]
    }
    return Diff(added, removed, rewritten)


@dataclass
class Blame:
    origins: list[dict[str, Any]]  # earliest non-summary items containing the text
    carried_by: list[dict[str, Any]]  # summaries that still contain it
    ancestors: list[str]  # source items reached by walking summary parents


def blame(store: SQLiteStore, session: str, text: str) -> Blame:
    items = store.items(session)
    by_id = {i["id"]: i for i in items}
    q = _norm(text)
    hits = [i for i in items if q in _norm(i["text"])]
    originals = sorted((i for i in hits if i["kind"] != "summary"), key=lambda i: i["turn"])
    summaries = [i for i in hits if i["kind"] == "summary"]

    ancestors: list[str] = []
    stack = [p for s in summaries for p in s["parents"]]
    while stack:
        i = stack.pop()
        node = by_id.get(i)
        if node is None or i in ancestors:
            continue
        if node["kind"] == "summary":
            stack.extend(node["parents"])
        else:
            ancestors.append(i)
    origins = [o for o in originals if o["turn"] == originals[0]["turn"]] if originals else []
    return Blame(origins, summaries, ancestors)
