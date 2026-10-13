from typing import Any

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from langchain.agents import create_agent
from langchain.agents.middleware import ContextEditingMiddleware, SummarizationMiddleware
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult

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
