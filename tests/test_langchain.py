from typing import Any

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from langchain.agents import create_agent
from langchain.agents.middleware import ContextEditingMiddleware, SummarizationMiddleware
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from test_audit import cued_answer

from rotguard.adapters.langchain import (
    RotguardMiddleware,
    from_neutral,
    to_neutral,
    to_neutral_list,
)
from rotguard.report import blame


class FakeChat(BaseChatModel):
    replies: list[Any] = []
    seen: list[Any] = []

    @property
    def _llm_type(self) -> str:
        return "fake"

    def _generate(self, messages, stop=None, run_manager=None, **kw):
        self.seen.append(list(messages))
        reply = self.replies.pop(0) if self.replies else AIMessage(content="ok")
        return ChatResult(generations=[ChatGeneration(message=reply)])

    def bind_tools(self, tools, **kw):
        return self


# --- conversion round-trips --------------------------------------------------------------

texts = st.text(min_size=0, max_size=40)
ids = st.text(alphabet="abcdef0123456789", min_size=3, max_size=8)


@st.composite
def lc_messages(draw):
    kind = draw(st.sampled_from(["human", "ai", "ai_tools", "tool", "system"]))
    i, t = draw(ids), draw(texts)
    if kind == "human":
        return HumanMessage(content=t, id=i)
    if kind == "system":
        return SystemMessage(content=t, id=i)
    if kind == "tool":
        return ToolMessage(
            content=t, tool_call_id=draw(ids), name=draw(st.sampled_from(["grep", "ls"])), id=i
        )
    if kind == "ai":
        return AIMessage(content=t, id=i)
    call = {"id": draw(ids), "name": "grep", "args": {"q": draw(texts)}, "type": "tool_call"}
    return AIMessage(content=t, id=i, tool_calls=[call])


@settings(max_examples=100, deadline=None)
@given(lc_messages())
def test_roundtrip(m):
    back = from_neutral(to_neutral(m, "x"))
    assert type(back) is type(m)
    assert (back.id, back.content, back.tool_calls if isinstance(m, AIMessage) else None) == (
        m.id,
        m.content,
        m.tool_calls if isinstance(m, AIMessage) else None,
    )
    if isinstance(m, ToolMessage):
        assert back.tool_call_id == m.tool_call_id


def test_roundtrip_block_content_and_missing_ids():
    m = HumanMessage(
        content=[{"type": "text", "text": "hi"}, {"type": "image_url", "image_url": {"url": "u"}}]
    )
    assert from_neutral(to_neutral(m, "x")).content == m.content
    ids_ = [x.id for x in to_neutral_list([HumanMessage(content="a"), HumanMessage(content="a")])]
    assert len(set(ids_)) == 2


# --- end to end through create_agent -----------------------------------------------------


def summarizer():
    return FakeChat(
        replies=[AIMessage(content="SUMMARY: user must use staging_orders only") for _ in range(10)]
    )


def build(rg, extra=(), model=None):
    model = model or FakeChat()
    return create_agent(model=model, tools=[], middleware=[*extra, rg]), model


def chat(agent, n, first="Constraint: only use staging_orders."):
    state = {"messages": [HumanMessage(content=first)]}
    for i in range(n):
        state = agent.invoke(state)
        state["messages"].append(HumanMessage(content=f"follow-up {i}"))
    return state


def test_ledger_injected_per_call_but_not_stored():
    rg = RotguardMiddleware()
    rg.ledger.record("constraint", "use staging_orders only", key="table")
    agent, model = build(rg)
    state = agent.invoke({"messages": [HumanMessage(content="go")]})
    sent = model.seen[-1]
    assert sent[-1].name == "rotguard_ledger" and "staging_orders" in sent[-1].content
    assert not any(getattr(m, "name", None) == "rotguard_ledger" for m in state["messages"])
    assert rg.store.entries(rg.session_id)[0]["text"] == "use staging_orders only"


def test_no_ledger_message_when_empty():
    agent, model = build(RotguardMiddleware())
    agent.invoke({"messages": [HumanMessage(content="go")]})
    assert all(getattr(m, "name", None) != "rotguard_ledger" for m in model.seen[-1])


def test_ledger_tool_records_and_supersedes():
    calls = [
        AIMessage(
            content="",
            tool_calls=[
                {
                    "id": "1",
                    "name": "ledger_record",
                    "type": "tool_call",
                    "args": {"kind": "decision", "text": "use Postgres", "key": "db"},
                }
            ],
        ),
        AIMessage(
            content="",
            tool_calls=[
                {
                    "id": "2",
                    "name": "ledger_record",
                    "type": "tool_call",
                    "args": {"kind": "decision", "text": "use Redis", "key": "db"},
                }
            ],
        ),
        AIMessage(content="done"),
    ]
    rg = RotguardMiddleware()
    agent, _ = build(rg, model=FakeChat(replies=calls))
    agent.invoke({"messages": [HumanMessage(content="pick a db")]})
    assert [e.text for e in rg.ledger.active()] == ["use Redis"]
    assert len(rg.ledger.all()) == 2 and rg.ledger.all()[0].source_message_id


def test_summarization_detected_blame_finds_original():
    rg = RotguardMiddleware()
    sm = SummarizationMiddleware(model=summarizer(), trigger=("messages", 5), keep=("messages", 2))
    agent, model = build(rg, extra=[sm])
    chat(agent, 8)
    events = rg.store.events(rg.session_id, "compaction")
    assert any(e["channel"] == "history" and e["summary_ids"] for e in events)
    b = blame(rg.store, rg.session_id, "staging_orders")
    assert b.origins and b.origins[0]["kind"] == "user" and b.origins[0]["turn"] == 1
    assert b.carried_by  # a summary still carries the fact


def test_fails_open(monkeypatch):
    rg = RotguardMiddleware()
    monkeypatch.setattr(rg.store, "add_items", lambda *a, **k: 1 / 0)
    agent, _ = build(rg)
    state = agent.invoke({"messages": [HumanMessage(content="go")]})
    assert state["messages"][-1].content == "ok"


@pytest.mark.asyncio
async def test_async_path():
    rg = RotguardMiddleware()
    rg.ledger.record("goal", "ship it")
    agent, model = build(rg)
    await agent.ainvoke({"messages": [HumanMessage(content="go")]})
    assert model.seen[-1][-1].name == "rotguard_ledger"


# --- middleware ordering (verified against langchain 1.4.4) --------------------------------


def _ordering_run(order):
    from langchain.agents.middleware import ClearToolUsesEdit
    from langchain.tools import tool

    @tool
    def grep(q: str) -> str:
        """search"""
        return "result " * 200

    rg = RotguardMiddleware()
    parts = {
        "sum": SummarizationMiddleware(
            model=summarizer(), trigger=("messages", 9), keep=("messages", 3)
        ),
        "edit": ContextEditingMiddleware(edits=[ClearToolUsesEdit(trigger=100, keep=1)]),
        "rg": rg,
    }
    calls = [
        AIMessage(
            content="",
            tool_calls=[{"id": f"c{i}", "name": "grep", "args": {"q": "x"}, "type": "tool_call"}],
        )
        for i in range(8)
    ]
    agent = create_agent(
        model=FakeChat(replies=[*calls, AIMessage(content="done")]),
        tools=[grep],
        middleware=[parts[k] for k in order],
    )
    agent.invoke(
        {"messages": [HumanMessage(content="Constraint: staging_orders only")]},
        {"recursion_limit": 60},
    )
    events = rg.store.events(rg.session_id, "compaction")
    hist = [e["turn"] for e in events if e["channel"] == "history" and e["summary_ids"]]
    cleared = sum(len(e["rewritten_ids"]) for e in events if e["channel"] == "request")
    return hist, cleared


@pytest.mark.parametrize(
    "order",
    [
        ("sum", "edit", "rg"),
        ("sum", "rg", "edit"),
        ("edit", "sum", "rg"),
        ("edit", "rg", "sum"),
        ("rg", "sum", "edit"),
        ("rg", "edit", "sum"),
    ],
)
def test_ordering_rules(order):
    hist, cleared = _ordering_run(order)
    assert len(hist) == 3  # summaries are caught in every order...
    late = order.index("rg") < order.index("sum")
    assert hist == ([6, 8, 9] if late else [5, 7, 9])  # ...one step late if rotguard is first
    # clearing is applied to a copy of the request, so rotguard sees it only from inside
    assert (cleared > 0) == (order.index("edit") < order.index("rg"))


# --- compaction audit through the middleware ---------------------------------------------------


class QuizChat(FakeChat):
    """Chat model that also answers audit prompts, knowing only what is in the context it gets."""

    extract: list[Any] = []
    qa: dict[str, str] = {}
    ledger_answers: list[str] = []
    sources: list[str] = []

    def _generate(self, messages, stop=None, run_manager=None, **kw):
        last = messages[-1].content if isinstance(messages[-1].content, str) else ""
        if "about to be removed" in last:
            import json

            return ChatResult(
                generations=[ChatGeneration(message=AIMessage(content=json.dumps(self.extract)))]
            )
        if "Answer from the conversation above only" in last:
            import re

            text = "\n".join(str(m.content) for m in messages[:-1]) + last.split("Answer from")[0]
            lines = []
            for line in last.splitlines():
                m = re.match(r"(\d+)\. (.*)", line)
                if not m:
                    continue
                if m.group(2).startswith("List every goal"):
                    ans = "; ".join(a for a in self.ledger_answers if a in text) or "unknown"
                elif (cued := cued_answer(m.group(2), text, self.sources)) is not None:
                    ans = cued
                else:
                    a = self.qa[m.group(2)]
                    ans = a if a in text else "unknown"
                lines.append(f"{m.group(1)}. {ans}")
            return ChatResult(
                generations=[ChatGeneration(message=AIMessage(content="\n".join(lines)))]
            )
        return super()._generate(messages, stop, run_manager, **kw)


def lossy_summarizer():
    return FakeChat(
        replies=[
            AIMessage(content="SUMMARY: the user is auditing the order service.") for _ in range(10)
        ]
    )


def audited_agent(policy, ledger_entry=False):
    from rotguard.probes import AuditPolicy  # noqa: F401

    model = QuizChat(
        extract=[{"question": "Which table may be written to?", "answer": "staging_orders"}],
        qa={"Which table may be written to?": "staging_orders"},
        ledger_answers=["staging_orders"],
    )
    rg = RotguardMiddleware(audit=policy)
    if ledger_entry:
        rg.ledger.record("constraint", "Only write to staging_orders.", key="table")
    sm = SummarizationMiddleware(
        model=lossy_summarizer(), trigger=("messages", 5), keep=("messages", 2)
    )
    return create_agent(model=model, tools=[], middleware=[sm, rg]), rg, model


def test_audit_reports_loss_and_ci_helper_fails():
    from rotguard.probes import AuditPolicy
    from rotguard.testing import assert_audit_loss

    agent, rg, model = audited_agent(AuditPolicy(mode="ci"))
    chat(agent, 6)
    reports = rg.store.events(rg.session_id, "audit")
    assert reports and reports[0]["loss"] == 1.0
    assert (
        reports[0]["lost"][0]["answer"] == "staging_orders"
        and reports[0]["lost"][0]["origin_kind"] == "user"
    )
    assert any("User-stated" in s for s in reports[0]["suggestions"])
    # side calls never entered the agent's history or the stored items
    assert not any(
        "Answer from the conversation" in i["text"] for i in rg.store.items(rg.session_id)
    )
    with pytest.raises(AssertionError, match="staging_orders"):
        assert_audit_loss(rg.store, rg.session_id, max_loss=0.3)
    assert_audit_loss(rg.store, rg.session_id, max_loss=1.0)


def test_ledger_removes_residual_loss():
    from rotguard.probes import AuditPolicy

    agent, rg, _ = audited_agent(AuditPolicy(mode="ci"), ledger_entry=True)
    chat(agent, 6)
    reports = rg.store.events(rg.session_id, "audit")
    assert reports[0]["loss"] == 0.0
    # later compactions only remove an already-lossy summary: nothing checkable, never a loss
    assert all(r["loss"] in (None, 0.0) for r in reports)


def test_audit_is_off_by_default_and_background_audits_finish_on_flush():
    from rotguard.probes import AuditPolicy

    agent, rg, model = audited_agent(None)
    chat(agent, 6)
    assert not rg.store.events(rg.session_id, "audit")
    assert not any(
        "Answer from the conversation" in str(m.content) for call in model.seen for m in call
    )

    agent, rg, _ = audited_agent(AuditPolicy(mode="every"))  # background thread
    chat(agent, 6)
    rg.session.flush()
    assert rg.store.events(rg.session_id, "audit")


def test_audit_error_is_recorded_not_raised():
    from rotguard.probes import AuditPolicy
    from rotguard.testing import assert_audit_loss

    agent, rg, model = audited_agent(AuditPolicy(mode="ci"))
    model.qa = {}  # the quiz prompt will now raise KeyError inside the model
    chat(agent, 6)
    assert rg.store.events(rg.session_id, "audit_error")
    with pytest.raises(AssertionError, match="audit failed"):
        assert_audit_loss(rg.store, rg.session_id)


def recall_agent(recall, on_alert=None):
    model = QuizChat(ledger_answers=["staging_orders"], sources=["Only write to staging_orders."])
    # the ledger is not injected: with it in front of the model a probe could always answer
    rg = RotguardMiddleware(recall=recall, on_alert=on_alert, inject_ledger=False)
    rg.ledger.record("constraint", "Only write to staging_orders.", key="table")
    sm = SummarizationMiddleware(
        model=lossy_summarizer(), trigger=("messages", 5), keep=("messages", 2)
    )
    return create_agent(model=model, tools=[], middleware=[sm, rg]), rg, model


def test_recall_probe_alerts_once_a_summary_drops_the_constraint():
    from rotguard.probes import RecallPolicy

    alerts = []
    policy = RecallPolicy(mode="every", every=1, alert_after=2, background=False, signals=False)
    agent, rg, _ = recall_agent(policy, alerts.append)
    chat(agent, 8)
    scores = [e["score"] for e in rg.store.events(rg.session_id, "recall")]
    assert scores[0] == 1.0 and scores[-1] == 0.0  # remembered at first, lost after compaction
    assert len(alerts) == 1
    # the probe is a side call: it never reached the stored history
    assert not any(
        "Answer from the conversation" in i["text"] for i in rg.store.items(rg.session_id)
    )


def test_marker_miss_forces_a_probe_ahead_of_schedule():
    from rotguard.probes import RecallPolicy

    policy = RecallPolicy(mode="every", every=100, marker="RG-OK", background=False, signals=False)
    agent, rg, model = recall_agent(policy)
    model.replies = [AIMessage(content="RG-OK sure")]  # later replies are plain "ok": no marker
    chat(agent, 3)
    assert len(rg.store.events(rg.session_id, "marker_miss")) >= 1
    assert len(rg.store.events(rg.session_id, "recall")) == 2  # turn 1, then forced by the miss
    assert any("begin every reply with RG-OK" in str(m.content) for m in model.seen[0])


def unrecallable_agent(policy):
    """No summarizer, and a ledger entry the chat never mentions: every probe fails, history grows."""
    model = QuizChat(ledger_answers=["region_zz9"], sources=["Deploy region is region_zz9."])
    rg = RotguardMiddleware(recall=policy, inject_ledger=False)
    rg.ledger.record("constraint", "Deploy region is region_zz9.")
    return create_agent(model=model, tools=[], middleware=[rg]), rg, model


def test_alert_with_handoff_action_restarts_the_context_from_the_ledger():
    from rotguard.probes import RecallPolicy

    policy = RecallPolicy(
        mode="every",
        every=1,
        alert_after=1,
        background=False,
        signals=False,
        action="handoff",
        handoff_keep=2,
    )
    agent, rg, model = unrecallable_agent(policy)
    chat(agent, 8)
    events = rg.store.events(rg.session_id, "handoff")
    assert events and all(e["after"] < e["before"] for e in events)
    seen = [str(m.content) for call in model.seen for m in call]
    assert any("rotguard handoff" in c and "region_zz9" in c for c in seen)
    # the replacement is a real history change, so the observer records it as a compaction
    assert len(rg.store.events(rg.session_id, "compaction")) >= 1


def test_a_handoff_that_would_not_shrink_the_context_waits_until_it_would():
    from rotguard.probes import RecallPolicy

    policy = RecallPolicy(
        mode="every", every=1, alert_after=1, background=False, signals=False, action="handoff"
    )
    agent, rg, _ = recall_agent(policy)  # a summarizer keeps the context at about 3 messages
    chat(agent, 8)
    assert rg.store.events(rg.session_id, "recall") and not rg.store.events(
        rg.session_id, "handoff"
    )


def test_handoff_is_off_by_default():
    from rotguard.probes import RecallPolicy

    policy = RecallPolicy(mode="every", every=1, alert_after=1, background=False, signals=False)
    agent, rg, model = recall_agent(policy)
    chat(agent, 8)
    assert rg.store.events(rg.session_id, "recall") and not rg.store.events(
        rg.session_id, "handoff"
    )
