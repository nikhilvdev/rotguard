"""F5 policies, F4a and pinning through the LangChain middleware."""

from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware
from langchain.tools import tool
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from test_langchain import FakeChat, QuizChat, chat, lossy_summarizer

from rotguard.adapters.langchain import (
    RotguardMiddleware,
    model_summarizer,
    to_neutral_list,
)
from rotguard.policies import DedupPolicy, OffloadPolicy, StagedCleanup
from rotguard.probes import RecallPolicy
from rotguard.report import blame
from rotguard.testing import assert_history_integrity

BIG = "\n".join(f"line {n} of a long listing" for n in range(400))  # about 2,000 tokens


@tool
def listing(q: str) -> str:
    """list things"""
    return BIG


@tool
def small(q: str) -> str:
    """short output"""
    return "ok"


def calls(name, n, start=0):
    return [
        AIMessage(
            content="",
            tool_calls=[
                {"id": f"c{start + i}", "name": name, "args": {"q": "x"}, "type": "tool_call"}
            ],
        )
        for i in range(n)
    ]


def agent_with(rg, replies, tools=(listing, small), extra=()):
    model = FakeChat(replies=[*replies, AIMessage(content="done")])
    return create_agent(model=model, tools=list(tools), middleware=[*extra, rg]), model


def run(agent, text="go"):
    return agent.invoke({"messages": [HumanMessage(content=text)]}, {"recursion_limit": 80})


# --- F5a offload --------------------------------------------------------------------------------


def test_big_tool_output_is_stored_and_replaced_by_a_stub_with_the_same_call_id():
    rg = RotguardMiddleware(offload=OffloadPolicy(max_tokens=300, preview_lines=2))
    agent, _ = agent_with(rg, calls("listing", 1))
    state = run(agent)
    msg = next(m for m in state["messages"] if isinstance(m, ToolMessage))
    assert msg.tool_call_id == "c0" and "offload:c0" in msg.content and len(msg.content) < 400
    assert msg.content.splitlines()[:2] == BIG.splitlines()[:2]
    row = rg.store.item(rg.session_id, "offload:c0")
    assert row["text"] == BIG and row["kind"] == "offloaded"
    assert_history_integrity(to_neutral_list(state["messages"]))
    # nothing is lost: blame still finds the text, and the agent can page it back
    assert blame(rg.store, rg.session_id, "line 399 of a long listing").origins
    reader = rg.tools[-1]
    assert reader.name == "recall_output"
    page = reader.invoke({"id": "offload:c0", "start_line": 0, "max_lines": 3})
    assert page.startswith("line 0 of a long listing") and "next: start_line=3" in page
    assert "no offloaded output" in reader.invoke({"id": "nope"})


def test_small_outputs_pass_through_and_offload_is_off_by_default():
    rg = RotguardMiddleware(offload=OffloadPolicy(max_tokens=300))
    state = run(agent_with(rg, calls("small", 1))[0])
    assert next(m for m in state["messages"] if isinstance(m, ToolMessage)).content == "ok"
    plain = RotguardMiddleware()
    state = run(agent_with(plain, calls("listing", 1))[0])
    assert next(m for m in state["messages"] if isinstance(m, ToolMessage)).content == BIG
    assert all(t.name != "recall_output" for t in plain.tools)


def test_offload_fails_open(monkeypatch):
    import rotguard.adapters.langchain as lc

    monkeypatch.setattr(lc, "store_output", lambda *a, **k: 1 / 0)
    rg = RotguardMiddleware(offload=OffloadPolicy(max_tokens=300))
    state = run(agent_with(rg, calls("listing", 1))[0])
    assert next(m for m in state["messages"] if isinstance(m, ToolMessage)).content == BIG


# --- F5b dedup ----------------------------------------------------------------------------------


def test_repeated_identical_outputs_are_collapsed_for_the_call_only():
    rg = RotguardMiddleware(dedup=DedupPolicy(min_chars=100))
    agent, model = agent_with(rg, calls("listing", 2))
    state = run(agent)
    sent = [m for m in model.seen[-1] if isinstance(m, ToolMessage)]
    assert sent[0].content == BIG and "same output as" in sent[1].content
    assert sent[1].tool_call_id == "c1"  # pairing kept
    stored = [m for m in state["messages"] if isinstance(m, ToolMessage)]
    assert [m.content for m in stored] == [BIG, BIG]  # history is unchanged
    assert_history_integrity(to_neutral_list(state["messages"]))


# --- F5c pinning --------------------------------------------------------------------------------


def test_a_pinned_message_dropped_by_a_summarizer_is_reattached_for_the_call():
    rg = RotguardMiddleware(inject_ledger=False)
    sm = SummarizationMiddleware(
        model=lossy_summarizer(), trigger=("messages", 5), keep=("messages", 2)
    )
    agent, model = build_chat(rg, sm)
    state = {"messages": [HumanMessage(content="Constraint: only use staging_orders.", id="m1")]}
    rg.pin("m1")
    for i in range(6):
        state = agent.invoke(state)
        state["messages"].append(HumanMessage(content=f"follow-up {i}"))
    last = " ".join(str(m.content) for m in model.seen[-1])
    assert "rotguard pinned message m1" in last and "only use staging_orders" in last


def build_chat(rg, sm):
    model = FakeChat()
    return create_agent(model=model, tools=[], middleware=[sm, rg]), model


def test_pins_persist_and_ledger_record_can_pin_its_source_message():
    rg = RotguardMiddleware()
    rg._last_assistant_id = "a7"
    rg.tools[0].invoke({"kind": "decision", "text": "use Redis", "pin": True})
    assert rg.session.pinned == {"a7"}
    again = RotguardMiddleware(rg.store, rg.session_id)
    assert again.session.pinned == {"a7"}  # reloaded from the store


# --- F5d staged cleanup, F4a compact -----------------------------------------------------------


def test_staged_cleanup_clears_old_outputs_first_and_never_calls_the_summarizer_if_enough():
    rg = RotguardMiddleware(staged=StagedCleanup(budget_tokens=4500, clear_keep=1))
    agent, _ = agent_with(rg, calls("listing", 5))
    state = run(agent)
    ev = rg.store.events(rg.session_id, "staged")
    assert ev and all(e["stages"] == ["clear"] for e in ev)
    tools = [m for m in state["messages"] if isinstance(m, ToolMessage)]
    assert BIG not in [m.content for m in tools[:-1]] and tools[-1].content == BIG
    assert rg.store.item(rg.session_id, "offload:c0")["text"] == BIG
    assert_history_integrity(to_neutral_list(state["messages"]))
    assert ev[0]["tokens_after"] < ev[0]["tokens_before"] and ev[0]["tokens_estimated"] is True


def test_staged_cleanup_summarizes_only_when_clearing_is_not_enough():
    summarizer = FakeChat(replies=[AIMessage(content="SUMMARY") for _ in range(10)])
    cfg = StagedCleanup(
        budget_tokens=800, clear_keep=1, summarizer=model_summarizer(summarizer), keep_messages=3
    )
    rg = RotguardMiddleware(staged=cfg)
    agent, _ = agent_with(rg, calls("listing", 5))
    state = run(agent)
    stages = [s for e in rg.store.events(rg.session_id, "staged") for s in e["stages"]]
    assert "clear" in stages and "summarize" in stages
    assert any(m.id.startswith("staged-summary-") for m in state["messages"])
    assert_history_integrity(to_neutral_list(state["messages"]))
    comp = rg.store.events(rg.session_id, "compaction")
    assert any(e["summary_ids"] for e in comp)  # the summary is a recorded compaction


def test_stage_two_is_disabled_when_another_summarizer_is_active():
    cfg = StagedCleanup(
        budget_tokens=50, clear_keep=1, summarizer=model_summarizer(lossy_summarizer())
    )
    rg = RotguardMiddleware(staged=cfg)
    sm = SummarizationMiddleware(
        model=lossy_summarizer(), trigger=("messages", 5), keep=("messages", 2)
    )
    agent, _ = agent_with(rg, calls("small", 6), extra=[sm])
    run(agent)
    stages = [s for e in rg.store.events(rg.session_id, "staged") for s in e["stages"]]
    assert "summarize" not in stages


def compact_agent(staged):
    model = QuizChat(ledger_answers=["region_zz9"], sources=["Deploy region is region_zz9."])
    policy = RecallPolicy(
        mode="every", every=1, alert_after=1, background=False, signals=False, action="compact"
    )
    rg = RotguardMiddleware(recall=policy, inject_ledger=False, staged=staged)
    rg.ledger.record("constraint", "Deploy region is region_zz9.")
    return create_agent(model=model, tools=[], middleware=[rg]), rg


def test_compact_action_runs_staged_cleanup_once_on_alert_even_under_budget():
    summarizer = FakeChat(replies=[AIMessage(content="SUMMARY") for _ in range(10)])
    cfg = StagedCleanup(
        budget_tokens=10**6, summarizer=model_summarizer(summarizer), keep_messages=2
    )
    agent, rg = compact_agent(cfg)
    chat(agent, 6)
    ev = rg.store.events(rg.session_id, "staged")
    assert ev and ev[0]["forced"] is True and "summarize" in ev[0]["stages"]
    assert len(ev) == 1  # one-shot: the alert fires once


def test_compact_action_without_staged_cleanup_only_logs():
    agent, rg = compact_agent(None)
    chat(agent, 4)
    assert rg.store.events(rg.session_id, "recall")
    assert not rg.store.events(rg.session_id, "staged")
