import pytest

pytest.importorskip("opentelemetry.sdk")
from opentelemetry import trace  # noqa: E402
from opentelemetry.sdk.trace import TracerProvider  # noqa: E402
from opentelemetry.sdk.trace.export import SimpleSpanProcessor  # noqa: E402
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter  # noqa: E402

from rotguard import telemetry  # noqa: E402
from rotguard.ledger import Ledger  # noqa: E402
from rotguard.probes import AuditPolicy, RecallPolicy  # noqa: E402
from rotguard.session import Session  # noqa: E402
from rotguard.store import SQLiteStore  # noqa: E402

EXPORTER = InMemorySpanExporter()
_provider = TracerProvider()
_provider.add_span_processor(SimpleSpanProcessor(EXPORTER))
trace.set_tracer_provider(_provider)  # allowed once per process


@pytest.fixture(autouse=True)
def clean():
    EXPORTER.clear()
    telemetry.configure(include_text=False)
    yield
    telemetry.configure(include_text=False)


def events():
    return [(s.name, dict(s.events[0].attributes)) for s in EXPORTER.get_finished_spans()]


def test_each_event_kind_becomes_one_span_event_with_the_expected_attributes():
    s = Session(SQLiteStore(), "sess", Ledger())
    s.event(
        "compaction",
        {
            "channel": "history",
            "turn": 3,
            "removed_ids": ["a", "b"],
            "rewritten_ids": [],
            "summary_ids": ["s"],
            "chars_before": 900,
            "chars_after": 120,
        },
    )
    s.event(
        "audit",
        {
            "channel": "history",
            "turn": 3,
            "loss": 0.5,
            "loss_all": 0.4,
            "facts_total": 4,
            "answerable_before": 2,
            "side_calls": 3,
            "lost": [{"answer": "secret_value"}],
        },
    )
    s.event(
        "recall",
        {
            "turn": 4,
            "score": 0.0,
            "alert": True,
            "target": "ledger",
            "asked": ["a"],
            "missed": ["a"],
            "stale": [],
            "rolling": 0.2,
            "side_calls": 1,
        },
    )
    s.event(
        "signals",
        {
            "turn": 4,
            "reasons": ["goal_drift"],
            "goal_overlap": 0.05,
            "repetition": 0.0,
            "tool_error_rate": 0.0,
        },
    )
    s.event("marker_miss", {"turn": 5})
    s.event("handoff", {"turn": 6, "before": 12, "after": 5})
    s.event(
        "staged",
        {
            "turn": 7,
            "stages": ["clear"],
            "cleared": 4,
            "summarized": 0,
            "tokens_before": 9000,
            "tokens_after": 3000,
            "tokens_estimated": True,
            "forced": False,
        },
    )
    got = dict(events())
    assert list(got) == [
        f"rotguard.{k}"
        for k in ("compaction", "audit", "recall", "signals", "marker_miss", "handoff", "staged")
    ]
    c = got["rotguard.compaction"]
    assert (
        c["rotguard.session_id"] == "sess"
        and c["rotguard.removed_ids_count"] == 2
        and c["rotguard.chars_after"] == 120
    )
    assert (
        got["rotguard.audit"]["rotguard.loss"] == 0.5
        and got["rotguard.audit"]["rotguard.lost_count"] == 1
    )
    r = got["rotguard.recall"]
    assert (
        r["rotguard.alert"] is True
        and r["rotguard.missed_count"] == 1
        and r["rotguard.target"] == "ledger"
    )
    assert tuple(got["rotguard.signals"]["rotguard.reasons"]) == ("goal_drift",)
    assert got["rotguard.handoff"]["rotguard.after"] == 5
    assert tuple(got["rotguard.staged"]["rotguard.stages"]) == ("clear",)


def test_text_is_left_out_unless_asked_for():
    data = {
        "channel": "history",
        "turn": 1,
        "loss": 1.0,
        "lost": [{"answer": "secret_value"}],
        "suggestions": ["do x"],
    }
    telemetry.emit("audit", "s", data)
    attrs = events()[0][1]
    assert "secret_value" not in str(attrs) and "do x" not in str(attrs)
    EXPORTER.clear()
    telemetry.configure(include_text=True)
    telemetry.emit("audit", "s", data)
    assert "secret_value" in str(events()[0][1])
    EXPORTER.clear()
    telemetry.emit("recall_error", "s", {"error": "RuntimeError('boom')"})
    assert "boom" in str(events()[0][1])


def test_an_event_inside_a_traced_call_attaches_to_that_span_and_unknown_kinds_are_skipped():
    tracer = trace.get_tracer("agent")
    with tracer.start_as_current_span("agent-run"):
        telemetry.emit("handoff", "s", {"turn": 2, "before": 4, "after": 2})
    [span] = EXPORTER.get_finished_spans()
    assert span.name == "agent-run" and span.events[0].name == "rotguard.handoff"
    EXPORTER.clear()
    telemetry.emit("pin", "s", {"id": "m1"})
    assert EXPORTER.get_finished_spans() == ()


def test_real_session_events_are_emitted_and_stored_together():
    s = Session(SQLiteStore(), "s", Ledger(), audit=AuditPolicy(mode="off"), recall=RecallPolicy())
    s.event("marker_miss", {"turn": 9})
    assert s.store.events("s", "marker_miss") and events()[0][0] == "rotguard.marker_miss"


def test_without_the_package_nothing_breaks(monkeypatch):
    import builtins

    real = builtins.__import__

    def no_otel(name, *a, **k):
        if name.startswith("opentelemetry"):
            raise ImportError(name)
        return real(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", no_otel)
    telemetry.emit("handoff", "s", {"turn": 1})  # does not raise
    assert EXPORTER.get_finished_spans() == ()
