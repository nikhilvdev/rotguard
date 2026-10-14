from langchain_core.messages import HumanMessage
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
