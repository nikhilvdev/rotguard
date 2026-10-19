"""MCP server: lets an MCP client such as Claude Code use a rotguard store.

    rotguard mcp trace.db --session my-session

Tools: `ledger_record` (keep a goal, decision or constraint alive), `recall` (fetch an item that
was compacted away, by id or by text), `blame` (where text came from and which summaries carry it)
and `report` (the session's recall, audit and compaction data).

The server reads and writes the SQLite file. An agent process that wrote the file loads the ledger
from it only when it starts, so entries recorded here reach it on its next run, not mid-run.
The `mcp` package (2.x, `pip install rotguard[mcp]`) is imported only by `make_server`.
"""

from __future__ import annotations

from typing import Any, Literal

from rotguard.ledger import Entry, Ledger
from rotguard.report import blame as blame_text
from rotguard.report.html import build
from rotguard.store import SQLiteStore

MAX_CHARS = 2000
MAX_HITS = 3


class RotguardTools:
    """The tool logic as plain methods, so it works and is tested without the MCP package."""

    def __init__(self, store: SQLiteStore, session: str):
        self.store, self.session = store, session
        self.ledger = Ledger(on_change=lambda e: store.upsert_entry(session, e.model_dump()))
        self.ledger.load([Entry(**e) for e in store.entries(session)])

    def ledger_record(
        self,
        kind: Literal["goal", "decision", "constraint", "question"],
        text: str,
        key: str | None = None,
        supersedes: str | None = None,
    ) -> str:
        """Record a goal, decision, constraint or open question so it survives context
        compaction. Give the same `key` as an earlier entry (or its id in `supersedes`) when
        this overrides it."""
        try:
            return f"recorded {self.ledger.record(kind, text, key=key, supersedes=supersedes).id}"
        except ValueError as exc:
            return f"not recorded: {exc}"

    def recall(self, query: str) -> str:
        """Get back an item that may have been compacted out of the context: pass its id, or a
        piece of its text."""
        items = self.store.items(self.session)
        hits = [i for i in items if i["id"] == query]
        if not hits:
            q = " ".join(query.lower().split())
            hits = [i for i in items if q and q in " ".join(i["text"].lower().split())]
        if not hits:
            return "nothing found"
        out = []
        for i in hits[:MAX_HITS]:
            gone = f", removed at turn {i['removed_turn']}" if i["removed_turn"] is not None else ""
            text = i["text"] if len(i["text"]) <= MAX_CHARS else i["text"][:MAX_CHARS] + "…"
            out.append(f"{i['id']} ({i['kind']}, turn {i['turn']}{gone}):\n{text}")
        more = f"\n(+{len(hits) - MAX_HITS} more)" if len(hits) > MAX_HITS else ""
        return "\n\n".join(out) + more

    def blame(self, text: str) -> dict[str, Any]:
        """Where a piece of text first appeared and which summaries still carry it."""
        b = blame_text(self.store, self.session, text)
        return {
            "origins": [o["id"] for o in b.origins],
            "carried_by": [s["id"] for s in b.carried_by],
            "source_items": b.ancestors,
        }

    def report(self) -> dict[str, Any]:
        """Recall probes, audits, compactions and the ledger for this session."""
        return build(self.store, self.session)


def make_server(tools: RotguardTools) -> Any:
    from mcp.server.mcpserver import MCPServer

    server = MCPServer(
        "rotguard", instructions="Keep goals and constraints alive across context compaction."
    )
    for fn in (tools.ledger_record, tools.recall, tools.blame, tools.report):
        server.tool(name=fn.__name__)(fn)
    return server


def serve(store: SQLiteStore, session: str) -> None:
    make_server(RotguardTools(store, session)).run("stdio")
