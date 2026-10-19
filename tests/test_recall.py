from test_audit import FakeCaller

from rotguard.core import Message, TextBlock
from rotguard.ledger import Entry, Ledger
from rotguard.probes import RecallMonitor, RecallPolicy, RecallResult, run_recall
from rotguard.session import Session
from rotguard.store import SQLiteStore

ENTRIES = [
    Entry(id="a", kind="constraint", text="Never write to prod_orders."),
    Entry(id="b", kind="constraint", text="The deploy window closes at 17:40."),
]
ANSWERS = ["prod_orders", "17:40"]


def ctx(*texts):
    return [
        Message(id=f"m{i}", role="user", blocks=[TextBlock(text=t)]) for i, t in enumerate(texts)
    ]


def test_run_recall_scores_the_share_of_entries_the_context_still_holds():
    caller = FakeCaller([], {}, ANSWERS)
    full = run_recall(context=ctx("prod_orders", "17:40"), entries=ENTRIES, caller=caller)
    assert full.score == 1.0 and full.missed == []
    half = run_recall(context=ctx("only prod_orders here"), entries=ENTRIES, caller=caller)
    assert half.score == 0.5 and half.missed == ["ledger:b"]


def test_run_recall_skips_superseded_entries_and_an_empty_ledger():
    old = ENTRIES[1].model_copy(update={"status": "superseded"})
    r = run_recall(context=ctx("x"), entries=[ENTRIES[0], old], caller=FakeCaller([], {}, ANSWERS))
    assert r.asked == ["ledger:a"]
    empty = run_recall(context=ctx("x"), entries=[], caller=FakeCaller([], {}))
    assert empty.score is None and empty.side_calls == 0


def test_run_recall_samples_at_most_n():
    r = run_recall(
        context=ctx("prod_orders 17:40"),
        entries=ENTRIES,
        caller=FakeCaller([], {}, ANSWERS),
        sample=1,
    )
    assert len(r.asked) == 1


def test_monitor_spaces_probes_and_alerts_only_on_consecutive_failures():
    m = RecallMonitor(RecallPolicy(mode="every", every=3, alert_after=2, max_probes=3))
    assert [m.due(t) for t in range(7)] == [True, False, False, True, False, False, True]
    assert not m.due(99)  # budget spent
    miss, hit = RecallResult(1, 0.0), RecallResult(2, 1.0)
    assert [m.record(r) for r in (miss, hit, miss, miss, miss)] == [False] * 3 + [True, False]
    assert m.rolling is not None and m.record(RecallResult(9, None)) is False
    assert not RecallMonitor(RecallPolicy()).due(0)  # off by default


def test_session_probes_and_fires_the_alert_hook_once():
    alerts, store = [], SQLiteStore()
    ledger = Ledger()
    ledger.record("constraint", "Never write to prod_orders.")
    policy = RecallPolicy(mode="every", every=1, alert_after=2, background=False)
    s = Session(store, "s", ledger, recall=policy, on_alert=alerts.append)
    caller = FakeCaller([], {}, ["prod_orders"])
    for turn in range(4):
        s.turn = turn
        s.run_recall_check(ctx("nothing relevant"), caller)  # the constraint is gone
    events = store.events("s", "recall")
    assert len(events) == 4 and [e["alert"] for e in events] == [False, True, False, False]
    assert len(alerts) == 1
    s.turn = 9
    s.run_recall_check(ctx("prod_orders"), caller)
    assert store.events("s", "recall")[-1]["score"] == 1.0


def test_a_failing_caller_is_logged_not_raised():
    class Boom:
        def ask(self, context, question):
            raise RuntimeError("down")

    store = SQLiteStore()
    ledger = Ledger()
    ledger.record("goal", "Ship release 4.2 today.")
    s = Session(store, "s", ledger, recall=RecallPolicy(mode="every", background=False))
    s.run_recall_check(ctx("x"), Boom())
    assert store.events("s", "recall_error")


def test_passive_signals_flag_drift_repetition_and_tool_errors():
    from rotguard.core import Provenance
    from rotguard.probes import passive_signals

    def m(i, role, text, kind=None):
        return Message(
            id=str(i),
            role=role,
            blocks=[TextBlock(text=text)],
            provenance=Provenance(kind=kind) if kind else None,
        )

    goal = [Entry(id="g", kind="goal", text="Migrate the orders database to staging.")]
    on_topic = [m(i, "user", "migrate orders database staging") for i in range(6)]
    assert not passive_signals(on_topic, goal).flag
    off_topic = [m(i, "user" if i % 2 else "assistant", "what about lunch today") for i in range(6)]
    sig = passive_signals(off_topic[:1] + off_topic[1:], goal)
    assert "goal_drift" in sig.reasons and sig.goal_overlap == 0.0
    assert "repetition" in sig.reasons  # three identical assistant replies
    errs = [m(i, "tool", "Traceback: boom", kind="tool") for i in range(3)]
    assert passive_signals(errs, []).reasons == ["tool_errors"]
    assert not passive_signals(on_topic[:2], goal).flag  # too little context to judge


def test_a_signal_flag_runs_a_probe_before_the_schedule():
    store, ledger = SQLiteStore(), Ledger()
    ledger.record("goal", "Migrate the orders database to staging.")
    s = Session(
        store,
        "s",
        ledger,
        recall=RecallPolicy(mode="every", every=100, background=False),
    )
    caller = FakeCaller([], {}, ["staging"])
    calm = ctx("migrate orders database staging", "ok")
    s.turn = 1
    s.run_recall_check(calm, caller)  # first probe is always due
    drifted = [
        Message(id=f"d{i}", role="user", blocks=[TextBlock(text="lunch plans today")])
        for i in range(6)
    ]
    s.turn = 2
    s.run_recall_check(drifted, caller)
    assert len(store.events("s", "recall")) == 2 and store.events("s", "signals")


def test_calibrate_sits_just_under_clean_performance():
    from rotguard.probes import calibrate

    assert calibrate([]) == 0.5
    assert calibrate([1.0] * 8) < 1.0 <= calibrate([1.0] * 8) + 1e-6
    assert calibrate([0.0, 0.0, 0.0, 1.0]) == 0.3  # floor
    assert 0.6 < calibrate([0.7, 0.7, 0.7, 1.0, 1.0, 1.0, 1.0, 1.0]) <= 0.7


def test_a_passing_probe_cancels_a_pending_handoff():
    ledger = Ledger()
    ledger.record("constraint", "Never write to prod_orders.")
    policy = RecallPolicy(mode="every", every=1, background=False, signals=False, action="handoff")
    s = Session(SQLiteStore(), "s", ledger, recall=policy)
    s.handoff_pending = True
    s.turn = 1
    s.run_recall_check(ctx("never write to prod_orders"), FakeCaller([], {}, ["prod_orders"]))
    assert s.handoff_pending is False
