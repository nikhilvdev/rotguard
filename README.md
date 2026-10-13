# rotguard

Measure and guard against context rot in LLM agents.

rotguard does not compact your context. It watches whatever does, keeps a ledger of the goals,
decisions and constraints that must survive, and records where every item in the context came from.

## Quickstart (LangChain v1)

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
    ],  # list rotguard LAST, after any compacting middleware
)
```

- The agent gets a `ledger_record` tool. Entries sharing a `key` replace each other; the old one is
  kept and marked superseded.
- The active ledger is added to the end of each model request. It is never written to your history.
- Compactions by any middleware are detected and stored with provenance. Query them:

```python
from rotguard.report import blame, diff

blame(rotguard.store, rotguard.session_id, "staging_orders")  # which message introduced it?
diff(rotguard.store, rotguard.session_id, 3, 5)  # what changed between turns?
```

Every hook fails open: if rotguard errors, the agent's call goes ahead unchanged and the error is logged.
Logging uses [logquill](https://pypi.org/project/logquill/); call `rotguard.log.configure([...transports])` to see it.

## Development

```bash
uv sync --extra langchain --extra eval
uv run pytest
uv run python -m evals.run --model ollama:qwen2.5:3b --turns 40 --strategies none,summarize,summarize+ledger
```
