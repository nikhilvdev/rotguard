"""Ledger of goals, decisions, constraints and open questions, with supersession."""

from __future__ import annotations

from typing import Callable, Literal

from pydantic import BaseModel

EntryKind = Literal["goal", "decision", "constraint", "question"]


class Entry(BaseModel):
    id: str
    kind: EntryKind
    text: str
    key: str | None = None  # entries sharing a key replace each other
    status: Literal["active", "superseded"] = "active"
    superseded_by: str | None = None
    source_message_id: str | None = None
    turn: int = 0


HEADER = (
    "[rotguard ledger: the current goals, decisions and constraints for this task. "
    "Treat as authoritative; overridden items are omitted.]"
)


class Ledger:
    """Entries are never deleted; a newer one marks the older `superseded`."""

    def __init__(self, on_change: Callable[[Entry], None] | None = None):
        self._entries: dict[str, Entry] = {}
        self._on_change = on_change or (lambda e: None)

    def all(self) -> list[Entry]:
        return list(self._entries.values())

    def active(self) -> list[Entry]:
        return [e for e in self._entries.values() if e.status == "active"]

    def load(self, entries: list[Entry]) -> None:
        self._entries = {e.id: e for e in entries}

    def record(
        self,
        kind: EntryKind,
        text: str,
        *,
        key: str | None = None,
        supersedes: str | None = None,
        source_message_id: str | None = None,
        turn: int = 0,
    ) -> Entry:
        if supersedes is not None and supersedes not in self._entries:
            raise ValueError(f"unknown ledger entry to supersede: {supersedes}")
        entry = Entry(
            id=f"L{len(self._entries) + 1}",
            kind=kind,
            text=text,
            key=key,
            source_message_id=source_message_id,
            turn=turn,
        )
        for old in self.active():
            if old.id == supersedes or (key is not None and old.key == key):
                old.status, old.superseded_by = "superseded", entry.id
                self._on_change(old)
        self._entries[entry.id] = entry
        self._on_change(entry)
        return entry

    def render(self) -> str:
        active = self.active()
        if not active:
            return ""
        return "\n".join([HEADER, *(f"- {e.kind}: {e.text}" for e in active)])
