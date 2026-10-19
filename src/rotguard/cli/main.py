"""`rotguard` command line: list sessions, render a report, blame a string, replay a trace.

rotguard sessions trace.db
rotguard report trace.db -o report.html
rotguard blame trace.db "only write to staging_orders"
rotguard mcp trace.db --session my-session
rotguard replay trace.db --model ollama:qwen2.5:3b --audit every --recall-every 2 -o replay.html
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rotguard.probes import AuditPolicy, RecallPolicy
from rotguard.replay import replay
from rotguard.report import blame
from rotguard.report.html import build, render_html
from rotguard.store import SQLiteStore


def _store(path: str) -> SQLiteStore:
    if path != ":memory:" and not Path(path).exists():
        raise SystemExit(f"rotguard: no such database: {path}")
    return SQLiteStore(path)


def _session(store: SQLiteStore, wanted: str | None) -> str:
    ids = store.sessions()
    if wanted:
        if wanted not in ids:
            raise SystemExit(f"rotguard: no session {wanted!r}; found: {', '.join(ids) or 'none'}")
        return wanted
    if len(ids) != 1:
        raise SystemExit(f"rotguard: pass --session; found: {', '.join(ids) or 'none'}")
    return ids[0]


def _write(html: str, out: str | None) -> None:
    if out:
        Path(out).write_text(html, encoding="utf-8")
        print(f"wrote {out}")
    else:
        print(html)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="rotguard")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def add(name: str, help: str) -> argparse.ArgumentParser:
        p = sub.add_parser(name, help=help)
        p.add_argument("db", help="SQLite file written by a rotguard adapter")
        p.add_argument("--session", help="session id (needed when the file holds several)")
        return p

    add("sessions", "list the sessions in a database")
    r = add("report", "write an HTML report for one session")
    r.add_argument("-o", "--out", help="output file (default: print to stdout)")
    b = add("blame", "find where a piece of text came from and which summaries carry it")
    b.add_argument("text")
    add("mcp", "serve this session's ledger, recall, blame and report to an MCP client over stdio")
    p = add("replay", "re-run a session's audits and recall probes with other settings")
    p.add_argument("--model", required=True, help="provider:name (anthropic, openai, ollama)")
    p.add_argument("--audit", default="every", choices=["off", "sampled", "every", "ci"])
    p.add_argument("--recall-every", type=int, default=0, help="probe every N turns (0 = off)")
    p.add_argument("--pass-at", type=float, default=0.5)
    p.add_argument(
        "--inject-ledger", action="store_true", help="put the ledger in front of the quiz"
    )
    p.add_argument("--db-out", help="keep the replay's events in this SQLite file")
    p.add_argument("-o", "--out", help="output HTML (default: print to stdout)")
    a = ap.parse_args(argv)

    store = _store(a.db)
    if a.cmd == "sessions":
        print("\n".join(store.sessions()) or "no sessions")
        return 0
    sid = _session(store, a.session)
    if a.cmd == "report":
        _write(render_html(build(store, sid)), a.out)
    elif a.cmd == "mcp":
        try:
            from rotguard.adapters.mcp import serve
        except ImportError as exc:
            raise SystemExit("rotguard: the mcp command needs `pip install rotguard[mcp]`") from exc
        serve(store, sid)
    elif a.cmd == "blame":
        res = blame(store, sid, a.text)
        print(f"origins: {[o['id'] for o in res.origins]}")
        print(f"carried by summaries: {[s['id'] for s in res.carried_by]}")
        print(f"source items behind those summaries: {res.ancestors}")
    else:
        from rotguard.cli.callers import make_caller

        try:
            caller = make_caller(a.model)
        except (ValueError, ImportError) as exc:
            raise SystemExit(f"rotguard: {exc}") from exc
        recall = (
            RecallPolicy(mode="every", every=a.recall_every, pass_at=a.pass_at)
            if a.recall_every
            else None
        )
        dst, new_id = replay(
            store,
            sid,
            caller,
            audit=AuditPolicy(mode=a.audit),
            recall=recall,
            inject_ledger=a.inject_ledger,
            dst=SQLiteStore(a.db_out) if a.db_out else None,
        )
        _write(render_html(build(dst, new_id)), a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
