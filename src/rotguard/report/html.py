"""One session from a store as a data dict and as a self-contained HTML page."""

from __future__ import annotations

from html import escape
from typing import Any

from rotguard.store import SQLiteStore


def build(store: SQLiteStore, session: str) -> dict[str, Any]:
    """Everything the report shows, as plain data (also what the MCP `report` tool returns)."""
    ev = store.events(session)

    def of(kind: str) -> list[dict[str, Any]]:
        return [e for e in ev if e["kind"] == kind]

    audits = of("audit")
    return {
        "session": session,
        "items": len(store.items(session)),
        "recall": [
            {
                "turn": e["turn"],
                "score": e["score"],
                "alert": e.get("alert", False),
                "missed": e["missed"],
                "rolling": e.get("rolling"),
            }
            for e in of("recall")
        ],
        "compactions": [
            {
                "turn": e["turn"],
                "channel": e["channel"],
                "removed": len(e["removed_ids"]),
                "rewritten": len(e["rewritten_ids"]),
                "chars_before": e["chars_before"],
                "chars_after": e["chars_after"],
            }
            for e in of("compaction")
        ],
        "audits": [
            {
                "turn": e["turn"],
                "loss": e["loss"],
                "loss_all": e.get("loss_all"),
                "facts_total": e["facts_total"],
                "lost": [r["answer"] for r in e["lost"]],
                "suggestions": e["suggestions"],
            }
            for e in audits
        ],
        "signals": [{"turn": e["turn"], "reasons": e["reasons"]} for e in of("signals")],
        "marker_misses": [e["turn"] for e in of("marker_miss")],
        "errors": len(of("audit_error")) + len(of("recall_error")),
        "ledger": store.entries(session),
    }


def _curve(data: dict[str, Any]) -> str:
    pts = data["recall"]
    if not pts:
        return "<p class=muted>No recall probes were run in this session.</p>"
    last = max([p["turn"] for p in pts] + [c["turn"] for c in data["compactions"]] + [1])
    w, h, pad = 640, 180, 28

    def x(t: int) -> float:
        return float(pad + (w - 2 * pad) * t / last)

    def y(s: float) -> float:
        return float(h - pad - (h - 2 * pad) * s)

    scored = [p for p in pts if p["score"] is not None]
    line = " ".join(f"{x(p['turn']):.1f},{y(p['score']):.1f}" for p in scored)
    marks = "".join(
        f"<line x1={x(c['turn']):.1f} x2={x(c['turn']):.1f} y1={pad} y2={h - pad} class=cmp>"
        f"<title>compaction at turn {c['turn']}</title></line>"
        for c in data["compactions"]
    )
    dots = "".join(
        f"<circle cx={x(p['turn']):.1f} cy={y(p['score']):.1f} r=4 "
        f"class='{'bad' if p['alert'] else 'dot'}'>"
        f"<title>turn {p['turn']}: score {p['score']:.2f}"
        f"{', ALERT' if p['alert'] else ''}</title></circle>"
        for p in scored
    )
    return (
        f"<svg viewBox='0 0 {w} {h}' role=img aria-label='recall score by turn'>"
        f"<line x1={pad} x2={w - pad} y1={y(0)} y2={y(0)} class=axis />"
        f"<line x1={pad} x2={w - pad} y1={y(1)} y2={y(1)} class=axis />"
        f"{marks}<polyline points='{line}' class=ln />{dots}"
        f"<text x={pad} y=14 class=lbl>1.0</text>"
        f"<text x={pad} y={h - 8} class=lbl>0.0 · turn 0…{last}</text></svg>"
        "<p class=muted>Dashed lines are compactions; red dots are alerts.</p>"
    )


def _table(head: list[str], rows: list[list[Any]]) -> str:
    if not rows:
        return "<p class=muted>None.</p>"
    th = "".join(f"<th>{escape(h)}</th>" for h in head)
    body = "".join(
        "<tr>" + "".join(f"<td>{escape(str(c))}</td>" for c in r) + "</tr>" for r in rows
    )
    return f"<table><tr>{th}</tr>{body}</table>"


CSS = """
:root{--bg:#fff;--fg:#1c1c1e;--mut:#6b6b70;--line:#d8d8dc;--acc:#2563eb;--bad:#dc2626}
@media(prefers-color-scheme:dark){:root{--bg:#141416;--fg:#ececf0;--mut:#9a9aa2;--line:#33333a;--acc:#6ea0ff;--bad:#ff6b6b}}
body{background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,sans-serif;
margin:0 auto;max-width:760px;padding:16px}
h1{font-size:1.4rem}h2{font-size:1.1rem;margin-top:2rem}.muted{color:var(--mut)}
table{border-collapse:collapse;width:100%;display:block;overflow-x:auto}
td,th{border-bottom:1px solid var(--line);padding:4px 8px;text-align:left;vertical-align:top}
svg{width:100%;height:auto}.axis{stroke:var(--line)}.cmp{stroke:var(--mut);stroke-dasharray:4}
.ln{fill:none;stroke:var(--acc);stroke-width:2}.dot{fill:var(--acc)}.bad{fill:var(--bad)}
.lbl{fill:var(--mut);font-size:11px}
"""


def render_html(data: dict[str, Any]) -> str:
    probes = [p for p in data["recall"] if p["score"] is not None]
    alerts = sum(p["alert"] for p in data["recall"])
    ledger = [
        [e["turn"], e["kind"], e["status"], e["text"], e.get("superseded_by") or ""]
        for e in data["ledger"]
    ]
    comp = [
        [c["turn"], c["channel"], c["removed"], c["rewritten"], c["chars_before"], c["chars_after"]]
        for c in data["compactions"]
    ]
    aud = [
        [
            a["turn"],
            "n/a" if a["loss"] is None else f"{a['loss']:.2f}",
            a["facts_total"],
            "; ".join(a["lost"]),
            " ".join(a["suggestions"]),
        ]
        for a in data["audits"]
    ]
    summary = (
        f"{data['items']} items · {len(data['compactions'])} compactions · "
        f"{len(probes)} recall probes · {alerts} alerts · "
        f"{len(data['marker_misses'])} marker misses · {data['errors']} side-call errors"
    )
    return (
        "<!doctype html><html lang=en><head><meta charset=utf-8>"
        "<meta name=viewport content='width=device-width,initial-scale=1'>"
        f"<title>rotguard report</title><style>{CSS}</style></head><body>"
        f"<h1>rotguard report</h1><p class=muted>session {escape(data['session'])}<br>"
        f"{escape(summary)}</p>"
        f"<h2>Recall curve</h2>{_curve(data)}"
        "<h2>Compactions</h2>"
        + _table(["turn", "channel", "removed", "rewritten", "chars before", "chars after"], comp)
        + "<h2>Audits</h2>"
        + _table(["turn", "loss", "facts", "lost", "suggestions"], aud)
        + "<h2>Ledger timeline</h2>"
        + _table(["turn", "kind", "status", "text", "superseded by"], ledger)
        + "</body></html>"
    )
