"""LangGraph helper: a node to put before your LLM node, and a ledger that survives checkpoints.

    rg = RotguardGraph("rotguard.db", model=model, audit=AuditPolicy(mode="every"))

    class State(MessagesState, RotguardState): ...

    def call_llm(state):
        return {"messages": [model.invoke(rg.with_ledger(state["messages"]))]}

    g = StateGraph(State)
    g.add_node("rotguard", rg.node); g.add_node("llm", call_llm)
    g.add_node("tools", ToolNode(rg.tools))      # plus your own tools
    g.add_edge(START, "rotguard"); g.add_edge("rotguard", "llm")

The ledger is written to the `rotguard_ledger` state key on every pass through the node, so a
checkpointer keeps it: a resumed thread in a new process loads it back. Observations, provenance and
audit results go to the store (a SQLite file you keep next to the checkpoint database).
"""

from __future__ import annotations

import uuid
from collections.abc import Callable
from typing import Any

from langchain_core.messages import AnyMessage, HumanMessage, RemoveMessage
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from typing_extensions import TypedDict

from rotguard.adapters.langchain import (
    LEDGER_MESSAGE_NAME,
    LangChainCaller,
    from_neutral,
    make_ledger_tool,
    to_neutral_list,
)
from rotguard.ledger import Entry, Ledger
from rotguard.probes import AuditPolicy, RecallPolicy, RecallResult
from rotguard.session import Session
from rotguard.store import Store, open_store


class RotguardState(TypedDict, total=False):
    """Mix into your graph state so the ledger is checkpointed."""

    rotguard_ledger: list[dict[str, Any]]


class RotguardGraph:
    def __init__(
        self,
        store: Store | str | None = None,
        session_id: str | None = None,
        audit: AuditPolicy | None = None,
        model: Any = None,
        inject_ledger: bool = True,
        recall: RecallPolicy | None = None,
        on_alert: Callable[[RecallResult], None] | None = None,
    ):
        for mode, what in (
            (audit and audit.mode, "audits"),
            (recall and recall.mode, "recall probes"),
        ):
            if mode not in (None, "off") and model is None:
                raise ValueError(f"{what} need the agent's model: pass model=...")
        self.store = open_store(store)
        self.session_id = session_id or uuid.uuid4().hex
        self.model, self.inject_ledger = model, inject_ledger
        self.ledger = Ledger(
            on_change=lambda e: self.store.upsert_entry(self.session_id, e.model_dump())
        )
        self.session = Session(self.store, self.session_id, self.ledger, audit, recall, on_alert)
        self.tools = [make_ledger_tool(self.ledger, lambda: self.session.turn, lambda: None)]

    def _save_entry(self, entry: Entry) -> None:
        self.store.upsert_entry(self.session_id, entry.model_dump())

    def node(self, state: dict[str, Any]) -> dict[str, Any]:
        """Observe the history, run due audits, and sync the ledger into the graph state."""
        saved = state.get("rotguard_ledger") or []
        if not self.ledger.all() and saved:  # resumed from a checkpoint
            self.ledger.load([Entry(**e) for e in saved])
        self.session.turn += 1
        neutral = to_neutral_list(state["messages"])
        self.session.observe_history(neutral)
        if self.model is not None:
            text = self.ledger.render() if self.inject_ledger else ""
            note = "\n".join(t for t in (text, self.session.marker_note()) if t)
            self.session.run_pending(LangChainCaller(self.model), text, note)
        current = [e.model_dump() for e in self.ledger.all()]
        update: dict[str, Any] = {"rotguard_ledger": current} if current != saved else {}
        fresh = self.session.take_handoff(neutral)
        if fresh is not None:  # RecallPolicy(action="handoff") and an alert fired
            update["messages"] = [
                RemoveMessage(id=REMOVE_ALL_MESSAGES),
                *[from_neutral(m) for m in fresh],
            ]
        return update

    def with_ledger(self, messages: list[AnyMessage]) -> list[AnyMessage]:
        """`messages` plus the active ledger at the end, for this model call only."""
        text = "\n".join(
            t
            for t in (
                self.ledger.render() if self.inject_ledger else "",
                self.session.marker_note(),
            )
            if t
        )
        return (
            [*messages, HumanMessage(content=text, name=LEDGER_MESSAGE_NAME)]
            if text
            else list(messages)
        )

    def observe_reply(self, text: str) -> None:
        """Call with each model reply when `recall.marker` is set, so a dropped marker is seen."""
        self.session.observe_reply(text)
