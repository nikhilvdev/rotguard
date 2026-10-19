from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from test_langchain import FakeChat

from rotguard.adapters.langgraph import RotguardGraph, RotguardState


class State(MessagesState, RotguardState):
    pass


def build(rg, model, saver):
    def call_llm(state):
        return {"messages": [model.invoke(rg.with_ledger(state["messages"]))]}

    g = StateGraph(State)
    g.add_node("rotguard", rg.node)
    g.add_node("llm", call_llm)
    g.add_edge(START, "rotguard")
    g.add_edge("rotguard", "llm")
    g.add_edge("llm", END)
    return g.compile(checkpointer=saver)


def test_ledger_is_injected_checkpointed_and_restored_in_a_new_process():
    saver, cfg = MemorySaver(), {"configurable": {"thread_id": "t1"}}
    model = FakeChat()
    rg = RotguardGraph()
    rg.ledger.record("constraint", "only write to staging_orders", key="table")
    graph = build(rg, model, saver)
    graph.invoke({"messages": [HumanMessage(content="start")]}, cfg)
    assert model.seen[-1][-1].name == "rotguard_ledger"
    assert (
        graph.get_state(cfg).values["rotguard_ledger"][0]["text"] == "only write to staging_orders"
    )

    # a fresh process: new RotguardGraph, empty ledger, same checkpointer
    model2, rg2 = FakeChat(), RotguardGraph()
    graph2 = build(rg2, model2, saver)
    graph2.invoke({"messages": [HumanMessage(content="and again")]}, cfg)
    assert [e.text for e in rg2.ledger.active()] == ["only write to staging_orders"]
    assert "staging_orders" in model2.seen[-1][-1].content
    # the ledger message was never written into the graph's messages
    assert all(
        getattr(m, "name", None) != "rotguard_ledger"
        for m in graph2.get_state(cfg).values["messages"]
    )


def test_node_sees_compaction_and_audit_needs_a_model():
    import pytest

    from rotguard.probes import AuditPolicy

    with pytest.raises(ValueError):
        RotguardGraph(audit=AuditPolicy(mode="every"))
    rg = RotguardGraph()
    msgs = [HumanMessage(content=f"m{i}", id=f"id{i}") for i in range(6)]
    rg.node({"messages": msgs})
    summary = HumanMessage(content="s", id="sum", additional_kwargs={"lc_source": "summarization"})
    rg.node({"messages": [summary, *msgs[4:]]})
    events = rg.store.events(rg.session_id, "compaction")
    assert (
        len(events) == 1
        and events[0]["summary_ids"] == ["sum"]
        and len(events[0]["removed_ids"]) == 4
    )


def test_recall_needs_a_model_and_probes_each_pass_through_the_node():
    import pytest

    from rotguard.probes import RecallPolicy

    policy = RecallPolicy(mode="every", every=1, signals=False, background=False)
    with pytest.raises(ValueError, match="recall probes"):
        RotguardGraph(recall=policy)
    model = FakeChat()
    rg = RotguardGraph(model=model, recall=policy)
    rg.ledger.record("constraint", "only write to staging_orders", key="table")
    graph = build(rg, model, MemorySaver())
    graph.invoke(
        {"messages": [HumanMessage(content="only write to staging_orders")]},
        {"configurable": {"thread_id": "r"}},
    )
    ev = rg.store.events(rg.session_id, "recall")
    assert len(ev) == 1 and ev[0]["asked"]


def test_node_returns_a_replacement_history_when_a_handoff_is_pending():
    from langchain_core.messages import RemoveMessage

    from rotguard.probes import RecallPolicy

    rg = RotguardGraph(model=FakeChat(), recall=RecallPolicy(action="handoff", handoff_keep=1))
    rg.ledger.record("constraint", "only write to staging_orders", key="table")
    msgs = [
        HumanMessage(content="first", id="m1"),
        AIMessage(content="ok", id="m2"),
        HumanMessage(content="latest", id="m3"),
    ]
    assert "messages" not in rg.node({"messages": msgs})  # nothing pending
    rg.session.handoff_pending = True
    update = rg.node({"messages": msgs})["messages"]
    assert isinstance(update[0], RemoveMessage)
    assert [m.id for m in update[1:]] == ["handoff-ledger", "m3"]
    assert "staging_orders" in update[1].content
    assert "messages" not in rg.node({"messages": msgs})  # one-shot
