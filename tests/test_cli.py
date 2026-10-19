import pytest
from test_audit import FakeCaller

from rotguard.cli import main as cli
from rotguard.cli.callers import chat_messages, make_caller
from rotguard.core import Message, Provenance, TextBlock
from rotguard.ledger import Ledger
from rotguard.probes import AuditPolicy, RecallPolicy
from rotguard.replay import replay
from rotguard.report.html import build, render_html
from rotguard.session import Session
from rotguard.store import SQLiteStore


def msg(i, text, kind=None, role="user"):
    return Message(
        id=i,
        role=role,
        blocks=[TextBlock(text=text)],
        provenance=Provenance(kind=kind or role),
    )


def make_trace(path):
    """A session whose third turn replaces the history with a lossy summary."""
    store = SQLiteStore(path)
    ledger = Ledger(on_change=lambda e: store.upsert_entry("orig", e.model_dump()))
    ledger.record("constraint", "Only write to staging_orders. <b>x</b>")
    s = Session(store, "orig", ledger)
    fact = msg("f", "Constraint: only write to staging_orders.")
    a, b, c = msg("a", "noise 1"), msg("b", "noise 2"), msg("c", "noise 3")
    for turn, ctx in ((1, [fact, a, b]), (2, [fact, a, b, c])):
        s.turn = turn
        s.observe_history(ctx)
    s.turn = 3
    s.observe_history([msg("s1", "SUMMARY: auditing the order service.", "summary"), c])
    return store


def fake():
    return FakeCaller(
        [{"question": "Which table may be written to?", "answer": "staging_orders"}],
        {"Which table may be written to?": "staging_orders"},
        ["staging_orders"],
    )


def test_store_lists_sessions_in_order():
    store = SQLiteStore()
    row = {"id": "x", "turn": 1, "kind": "user", "role": "user", "parents": [], "hash": "h"}
    store.add_items("first", [{**row, "text": "t"}])
    store.add_event("events-only", "recall", {"turn": 1})
    assert store.sessions() == ["first", "events-only"]


def test_report_has_the_data_and_escapes_text():
    store = make_trace(":memory:")
    data = build(store, "orig")
    assert data["items"] == 5 and len(data["compactions"]) == 1
    assert data["compactions"][0]["removed"] == 3 and data["ledger"][0]["kind"] == "constraint"
    html = render_html(data)
    assert "<svg" not in html  # no probes ran: says so instead of drawing an empty chart
    assert "No recall probes" in html and "&lt;b&gt;x&lt;/b&gt;" in html and "<b>x</b>" not in html


def test_replay_scores_the_trace_under_new_settings():
    src = make_trace(":memory:")
    alerts = []
    dst, sid = replay(
        src,
        "orig",
        fake(),
        audit=AuditPolicy(mode="every"),
        recall=RecallPolicy(mode="every", every=1, alert_after=1, signals=False),
        on_alert=alerts.append,
    )
    assert sid.startswith("replay-orig-") and sid != "orig"
    scores = [e["score"] for e in dst.events(sid, "recall")]
    assert scores == [1.0, 1.0, 0.0] and len(alerts) == 1  # the summary drops the constraint
    audit = dst.events(sid, "audit")
    assert audit and "staging_orders" in audit[0]["lost"][0]["answer"]
    assert len(src.events("orig", "recall")) == 0  # the source trace is untouched
    html = render_html(build(dst, sid))
    assert "<svg" in html and "ALERT" in html


def test_cli_sessions_report_blame_and_replay(tmp_path, capsys, monkeypatch):
    db = str(tmp_path / "t.db")
    make_trace(db).close()
    assert cli.main(["sessions", db]) == 0 and capsys.readouterr().out.strip() == "orig"
    out = tmp_path / "r.html"
    assert cli.main(["report", db, "-o", str(out)]) == 0 and "rotguard report" in out.read_text()
    cli.main(["blame", db, "staging_orders"])
    assert "['f']" in capsys.readouterr().out

    monkeypatch.setattr("rotguard.cli.callers.make_caller", lambda spec: fake())
    out2, keep = tmp_path / "rp.html", str(tmp_path / "replay.db")
    args = ["replay", db, "--model", "x:y", "--recall-every", "1", "--db-out", keep]
    cli.main([*args, "-o", str(out2)])
    page = out2.read_text()
    assert "<svg" in page and "staging_orders" in page  # curve drawn, ledger carried over
    assert any(s.startswith("replay-orig-") for s in SQLiteStore(keep).sessions())


def test_cli_fails_clearly(tmp_path):
    with pytest.raises(SystemExit, match="no such database"):
        cli.main(["sessions", str(tmp_path / "nope.db")])
    db = str(tmp_path / "t.db")
    make_trace(db).close()
    with pytest.raises(SystemExit, match="no session 'zzz'"):
        cli.main(["report", db, "--session", "zzz"])
    with pytest.raises(ValueError, match="provider:name"):
        make_caller("gpt")
    with pytest.raises(ValueError, match="unknown provider"):
        make_caller("acme:m")


def test_chat_messages_alternate_start_and_end_with_user():
    ctx = [msg("1", "hi", role="assistant"), msg("2", "a"), msg("3", "b")]
    out = chat_messages(ctx, "q?")
    assert [m["role"] for m in out] == ["user", "assistant", "user"]
    assert out[-1]["content"] == "a\n\nb\n\nq?"
