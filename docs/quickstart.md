# Quickstart

## The ledger

The ledger holds the goals, decisions, constraints and open questions that must survive compaction.

- Entries sharing a `key` replace each other. The old entry is **kept** and marked `superseded`;
  nothing is ever deleted.
- The active entries are added to the end of each model request **for that call only**. They are
  never written into your stored history.
- With the LangChain middleware and the LangGraph helper the agent gets a `ledger_record` tool, so
  it can keep its own constraints alive.

```python
rotguard.ledger.record("constraint", "Only write to staging_orders.", key="table")
rotguard.ledger.record("constraint", "Use archive_orders instead.", key="table")  # supersedes it
```

## LangChain v1

```python
from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware
from rotguard.adapters.langchain import RotguardMiddleware

rotguard = RotguardMiddleware("rotguard.db")  # or no path for an in-memory store

agent = create_agent(
    model="anthropic:claude-sonnet-5-5",
    tools=[...],
    middleware=[
        SummarizationMiddleware(model="anthropic:claude-haiku-5-5", trigger=("tokens", 100_000)),
        rotguard,
    ],
)
```

!!! warning "List rotguard last"
    It must come after `SummarizationMiddleware` and `ContextEditingMiddleware` so it sees what they
    did. A test in the repo pins this ordering.

Options: `inject_ledger=True`, `ledger_tool=True`, `audit=AuditPolicy(...)`,
`recall=RecallPolicy(...)`, `on_alert=callable`.

## LangGraph

Add a node before your LLM node and mix `RotguardState` into your state so the ledger is
checkpointed:

```python
from rotguard.adapters.langgraph import RotguardGraph, RotguardState

rg = RotguardGraph("rotguard.db", model=model)  # `model` is needed for audits and recall checks


class State(MessagesState, RotguardState): ...


def call_llm(state):
    return {"messages": [model.invoke(rg.with_ledger(state["messages"]))]}


g = StateGraph(State)
g.add_node("rotguard", rg.node)
g.add_node("llm", call_llm)
g.add_node("tools", ToolNode(rg.tools))  # plus your own tools
g.add_edge(START, "rotguard")
g.add_edge("rotguard", "llm")
```

A resumed thread in a new process loads the ledger back from the checkpoint.

## Anthropic SDK

```python
import anthropic
from rotguard.adapters.anthropic import RotguardAnthropic

rg = RotguardAnthropic(anthropic.Anthropic(), "rotguard.db", audit=AuditPolicy(mode="every"))
resp = rg.beta.messages.create(model=..., messages=..., betas=["compact-2026-01-12"], ...)
```

It recognizes server-side compaction blocks (threshold and on-demand) and audits them. Side calls
resend your `tools` and `betas` and keep the signed compaction block exactly as returned.

!!! note
    This wrapper observes only: it injects no ledger and adds no tool. Record entries yourself with
    `rg.ledger.record(...)`. It is tested against the documented block shapes and the SDK's types,
    not the live API.

## OpenAI Agents SDK

Wrap any session that has the SDK's methods (`get_items`, `add_items`, `pop_item`, `clear_items`),
such as a trimming or summarizing session:

```python
from rotguard.adapters.openai_agents import RotguardSession

session = RotguardSession(
    inner_session, openai.OpenAI(), model="gpt-4.1", audit=AuditPolicy(mode="every")
)
Runner.run(agent, "hi", session=session)
```

A compaction done by the inner session shows up the next time the SDK reads the items. Like the
Anthropic wrapper it observes only. It is tested against a fake session, not the real SDK.

## Ask where something came from

```python
from rotguard.report import blame, diff

blame(rotguard.store, rotguard.session_id, "staging_orders")  # which message introduced it?
diff(rotguard.store, rotguard.session_id, 3, 5)  # what changed between turns 3 and 5?
```

## Logging

Logging uses [logquill](https://pypi.org/project/logquill/). Call
`rotguard.log.configure([...transports])` to see it.
