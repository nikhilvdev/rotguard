"""SQLite store (stdlib). Everything is keyed by session id."""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS items (
  session TEXT, id TEXT, turn INT, kind TEXT, role TEXT, tool_name TEXT, call_id TEXT,
  parents TEXT, hash TEXT, text TEXT, removed_turn INT, seq INTEGER,
  PRIMARY KEY (session, id));
CREATE TABLE IF NOT EXISTS snapshots (
  session TEXT, turn INT, ids TEXT, PRIMARY KEY (session, turn));
CREATE TABLE IF NOT EXISTS entries (session TEXT, id TEXT, data TEXT, PRIMARY KEY (session, id));
CREATE TABLE IF NOT EXISTS events (session TEXT, ts REAL, kind TEXT, data TEXT);
"""


class SQLiteStore:
    def __init__(self, path: str = ":memory:"):
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        with self._lock:
            self._db.executescript(SCHEMA)

    def _run(self, sql: str, args: tuple[Any, ...] = ()) -> sqlite3.Cursor:
        with self._lock:
            cur = self._db.execute(sql, args)
            self._db.commit()
            return cur

    def add_items(self, session: str, items: list[dict[str, Any]]) -> None:
        for it in items:
            self._run(
                "INSERT OR IGNORE INTO items VALUES (?,?,?,?,?,?,?,?,?,?,NULL,"
                "(SELECT COUNT(*) FROM items WHERE session=?))",
                (
                    session,
                    it["id"],
                    it["turn"],
                    it["kind"],
                    it["role"],
                    it.get("tool_name"),
                    it.get("call_id"),
                    json.dumps(it["parents"]),
                    it["hash"],
                    it["text"],
                    session,
                ),
            )

    def mark_removed(self, session: str, ids: list[str], turn: int) -> None:
        for i in ids:
            self._run(
                "UPDATE items SET removed_turn=? WHERE session=? AND id=? AND removed_turn IS NULL",
                (turn, session, i),
            )

    def items(self, session: str) -> list[dict[str, Any]]:
        rows = self._run("SELECT * FROM items WHERE session=? ORDER BY seq", (session,)).fetchall()
        return [{**dict(r), "parents": json.loads(r["parents"])} for r in rows]

    def item(self, session: str, item_id: str) -> dict[str, Any] | None:
        r = self._run("SELECT * FROM items WHERE session=? AND id=?", (session, item_id)).fetchone()
        return {**dict(r), "parents": json.loads(r["parents"])} if r else None

    def save_snapshot(self, session: str, turn: int, ids: list[str]) -> None:
        self._run(
            "INSERT OR REPLACE INTO snapshots VALUES (?,?,?)", (session, turn, json.dumps(ids))
        )

    def snapshot_at(self, session: str, turn: int) -> list[str]:
        """Context ids at the latest snapshot at or before `turn`."""
        r = self._run(
            "SELECT ids FROM snapshots WHERE session=? AND turn<=? ORDER BY turn DESC LIMIT 1",
            (session, turn),
        ).fetchone()
        return json.loads(r["ids"]) if r else []

    def upsert_entry(self, session: str, entry: dict[str, Any]) -> None:
        self._run(
            "INSERT OR REPLACE INTO entries VALUES (?,?,?)",
            (session, entry["id"], json.dumps(entry)),
        )

    def entries(self, session: str) -> list[dict[str, Any]]:
        rows = self._run(
            "SELECT data FROM entries WHERE session=? ORDER BY rowid", (session,)
        ).fetchall()
        return [json.loads(r["data"]) for r in rows]

    def add_event(self, session: str, kind: str, data: dict[str, Any]) -> None:
        self._run(
            "INSERT INTO events VALUES (?,?,?,?)", (session, time.time(), kind, json.dumps(data))
        )

    def events(self, session: str, kind: str | None = None) -> list[dict[str, Any]]:
        sql, args = "SELECT ts, kind, data FROM events WHERE session=?", [session]
        if kind:
            sql, args = sql + " AND kind=?", [session, kind]
        return [
            {"ts": r["ts"], "kind": r["kind"], **json.loads(r["data"])}
            for r in self._run(sql + " ORDER BY ts", tuple(args)).fetchall()
        ]

    def sessions(self) -> list[str]:
        """Every session id in the store: those with items first, then event-only ones."""
        out: list[str] = []
        for sql in (
            "SELECT session FROM items GROUP BY session ORDER BY MIN(seq)",
            "SELECT session FROM events GROUP BY session ORDER BY MIN(ts)",
        ):
            out += [r["session"] for r in self._run(sql).fetchall() if r["session"] not in out]
        return out

    def close(self) -> None:
        with self._lock:
            self._db.close()
