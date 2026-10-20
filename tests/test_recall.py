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
SOURCES = [e.text for e in ENTRIES]


def ctx(*texts):
    return [
        Message(id=f"m{i}", role="user", blocks=[TextBlock(text=t)]) for i, t in enumerate(texts)
    ]


def test_run_recall_scores_the_share_of_entries_the_context_still_holds():
    caller = FakeCaller([], {}, ANSWERS, SOURCES)
    full = run_recall(context=ctx("prod_orders", "17:40"), entries=ENTRIES, caller=caller)
    assert full.score == 1.0 and full.missed == []
    half = run_recall(context=ctx("only prod_orders here"), entries=ENTRIES, caller=caller)
    assert half.score == 0.5 and half.missed == ["cued:b"]


def test_run_recall_skips_superseded_entries_and_an_empty_ledger():
    old = ENTRIES[1].model_copy(update={"status": "superseded"})
    r = run_recall(
        context=ctx("x"), entries=[ENTRIES[0], old], caller=FakeCaller([], {}, ANSWERS, SOURCES)
    )
    assert r.asked == ["cued:a"]
    empty = run_recall(context=ctx("x"), entries=[], caller=FakeCaller([], {}))
    assert empty.score is None and empty.side_calls == 0


def test_run_recall_samples_at_most_n():
    r = run_recall(
        context=ctx("prod_orders 17:40"),
        entries=ENTRIES,
        caller=FakeCaller([], {}, ANSWERS, SOURCES),
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
    caller = FakeCaller([], {}, ["prod_orders"], ["Never write to prod_orders."])
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
    caller = FakeCaller([], {}, ["staging"], ["Migrate the orders database to staging."])
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
    s.run_recall_check(
        ctx("never write to prod_orders"),
        FakeCaller([], {}, ["prod_orders"], ["Never write to prod_orders."]),
    )
    assert s.handoff_pending is False


# --- R1: cued, per-entry questions --------------------------------------------------------


class Scripted:
    """A caller that returns the same numbered reply whatever it is asked, and records context."""

    def __init__(self, *answers):
        self.reply = "\n".join(f"{i}. {a}" for i, a in enumerate(answers, 1))
        self.contexts = []

    def ask(self, context, question):
        self.contexts.append(context)
        return self.reply


def led(*pairs):
    ledger = Ledger()
    for kind, text, *key in pairs:
        ledger.record(kind, text, key=key[0] if key else None)
    return ledger


def test_cued_probe_scores_one_for_correct_answers_and_zero_for_unknown():
    caller = FakeCaller([], {}, ANSWERS, SOURCES)
    ok = run_recall(context=ctx("prod_orders 17:40"), entries=ENTRIES, caller=caller)
    assert ok.score == 1.0
    none = run_recall(context=ctx("nothing here"), entries=ENTRIES, caller=caller)
    assert none.score == 0.0 and sorted(none.missed) == ["cued:a", "cued:b"]


def test_a_question_that_contains_another_entrys_answer_is_dropped():
    entries = led(("constraint", "Always use beta_2."), ("constraint", "Use alpha_1 and beta_2."))
    r = run_recall(context=ctx("x"), entries=entries.all(), caller=Scripted("unknown", "unknown"))
    assert r.asked == ["cued:L1"]  # L2's question shows beta_2, which is L1's answer


def test_an_entry_with_no_identifier_still_gets_a_gradable_question():
    from rotguard.probes import cued_ledger_facts

    facts = cued_ledger_facts(
        led(
            ("constraint", "Always answer in a formal English tone."),
            ("goal", "Summarize the quarterly onboarding survey results."),
        ).all()
    )
    assert "____" in facts[0].question and facts[0].answer == "English"
    assert facts[1].question == "What is the current goal of this task?" and facts[1].tokens
    r = run_recall(
        context=ctx("x"),
        entries=led(("constraint", "Always answer in a formal English tone.")).all(),
        caller=Scripted("english"),
    )
    assert r.score == 1.0


def test_style_list_keeps_the_single_list_question():
    caller = FakeCaller([], {}, ANSWERS, SOURCES)
    r = run_recall(context=ctx("prod_orders only"), entries=ENTRIES, caller=caller, style="list")
    assert r.asked == ["ledger:a", "ledger:b"] and r.missed == ["ledger:b"]
    assert "List every goal" in caller.calls[-1][1]


# --- R2: what the probe sees and what it asks about -------------------------------------------


def make_session(ledger, **policy):
    base = {"mode": "every", "every": 1, "signals": False, "background": False}
    return Session(SQLiteStore(), "s", ledger, recall=RecallPolicy(**{**base, **policy}))


def msg(i, text, role="user"):
    return Message(id=i, role=role, blocks=[TextBlock(text=text)])


def test_injected_ledger_means_the_probe_sees_the_note_and_asks_about_history():
    ledger = led(("constraint", "Never write to prod_orders."))
    s, caller = make_session(ledger), Scripted("unknown")
    s.turn = 1
    s.observe_history([msg("m1", "Please deploy to the eu_west_1 region today.")])
    s.run_pending(caller, ledger.render(), ledger.render())  # the ledger is injected
    ev = s.store.events("s", "recall")[0]
    assert ev["target"] == "history"
    assert (
        caller.contexts[0][-1].id == "rotguard-note"
        and "prod_orders" in caller.contexts[0][-1].text
    )
    assert ev["asked"] == ["pool:1"]  # a fact from the conversation, not the ledger entry


def test_without_injection_the_probe_asks_about_the_ledger():
    ledger = led(("constraint", "Never write to prod_orders."))
    s, caller = make_session(ledger), Scripted("unknown")
    s.turn = 1
    s.observe_history([msg("m1", "Please deploy to the eu_west_1 region today.")])
    s.run_pending(caller)
    ev = s.store.events("s", "recall")[0]
    assert ev["target"] == "ledger" and ev["asked"] == ["cued:L1"]


def test_pool_keeps_facts_after_the_source_is_compacted_and_has_no_duplicates():
    ledger = led(("constraint", "Never write to prod_orders."))
    s, caller = make_session(ledger, target="history"), Scripted("unknown")
    s.turn = 1
    s.observe_history([msg("m1", "Please deploy to the eu_west_1 region today.")])
    s.turn = 2
    s.observe_history([msg("m2", "Again: deploy to the eu_west_1 region today.")])  # compaction
    assert len(s.pool.facts) == 1  # same token twice, one fact
    s.run_pending(caller)
    assert s.store.events("s", "recall")[0]["asked"] == ["pool:1"]
    assert "m1" not in [m.id for m in caller.contexts[0]]  # the source is gone, the fact is not


def test_facts_the_ledger_covers_are_left_out_of_the_pool():
    ledger = led(("constraint", "Never write to prod_orders."))
    s = make_session(ledger, target="history")
    s.turn = 1
    s.observe_history([msg("m1", "We must not touch prod_orders at all.")])
    assert s.pool.facts == []


# --- R3: stale beliefs ------------------------------------------------------------------------


def test_an_answer_with_the_replaced_value_is_stale_and_counts_as_a_miss():
    ledger = led(
        ("constraint", "Only write to staging_orders.", "table"),
        ("constraint", "Only write to archive_orders.", "table"),
    )
    r = run_recall(context=ctx("x"), entries=ledger.all(), caller=Scripted("staging_orders"))
    assert r.asked == ["cued:L2"] and r.missed == ["cued:L2"] and r.stale == ["cued:L2"]
    assert r.score == 0.0
    ok = run_recall(context=ctx("x"), entries=ledger.all(), caller=Scripted("archive_orders"))
    assert ok.score == 1.0 and ok.stale == []
    # mentioning the old value after the right one ("not staging_orders") is not stale
    after = run_recall(
        context=ctx("x"),
        entries=ledger.all(),
        caller=Scripted("archive_orders, not staging_orders"),
    )
    assert after.score == 1.0 and after.stale == []
    # naming the old value first is
    first = run_recall(
        context=ctx("x"),
        entries=ledger.all(),
        caller=Scripted("staging_orders, archive_orders"),
    )
    assert first.stale == ["cued:L2"]
