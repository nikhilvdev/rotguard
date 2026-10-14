"""End to end with a real model: create_agent + a real summarizer + the audit in CI mode.

    uv run python -m evals.e2e_langchain --model qwen2.5:3b

Two runs of the same conversation: without a ledger, and with the constraints recorded in it
(recorded by hand here; a model without tool calling cannot call `ledger_record`).
"""

from __future__ import annotations

import argparse
import json

from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware
from langchain_core.messages import HumanMessage

from rotguard.adapters.langchain import RotguardMiddleware
from rotguard.probes import AuditPolicy

from .lc_ollama import OllamaChat

FIRST = (
    "Task: audit the order service. Constraints: only write to the staging_orders table, never "
    "to prod_orders. The customer's budget cap is 12,500 EUR. Acknowledge briefly."
)
PROBE = "Which table may you write to, and what is the budget cap? Answer in one sentence."


def filler(i: int) -> str:
    rows = " ".join(f"ref{j * 7 + i}:{(j * 193 + i * 71) % 9000}" for j in range(70))
    return (
        f"[tool result] grep output for handle_request, batch {i}\n{rows}\nSummarize in one line."
    )


def run(label: str, model: str, with_ledger: bool) -> dict:
    llm = OllamaChat(model=model)
    rg = RotguardMiddleware(audit=AuditPolicy(mode="ci", max_facts=4))
    if with_ledger:
        rg.ledger.record(
            "constraint", "Only write to staging_orders; never prod_orders.", key="table"
        )
        rg.ledger.record("constraint", "The customer's budget cap is 12,500 EUR.", key="budget")
    summarizer = SummarizationMiddleware(model=llm, trigger=("messages", 8), keep=("messages", 2))
    agent = create_agent(model=llm, tools=[], middleware=[summarizer, rg])
    state = {"messages": [HumanMessage(content=FIRST)]}
    for i in range(6):
        state = agent.invoke(state)
        state["messages"].append(HumanMessage(content=filler(i)))
    state["messages"].append(HumanMessage(content=PROBE))
    state = agent.invoke(state)
    answer = state["messages"][-1].content
    ok = "staging_orders" in answer.lower() and "12500" in answer.lower().replace(",", "")
    audits = rg.store.events(rg.session_id, "audit")
    out = {
        "run": label,
        "compactions": len(rg.store.events(rg.session_id, "compaction")),
        "audits": [
            {k: a[k] for k in ("turn", "loss", "facts_total", "answerable_before", "side_calls")}
            | {"lost": [x["answer"] for x in a["lost"]]}
            for a in audits
        ],
        "errors": rg.store.events(rg.session_id, "audit_error"),
        "model_calls": llm.calls,
        "probe_answer": answer,
        "probe_ok": ok,
    }
    print(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen2.5:3b")
    a = ap.parse_args()
    run("no ledger", a.model, False)
    run("with ledger", a.model, True)
