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

## Audit what a compaction lost

Audits are off unless you ask for them, because each one costs about three extra model calls
(one to extract facts from the removed span, one quiz on the old context, one on the new).

```python
from rotguard.probes import AuditPolicy

rotguard = RotguardMiddleware("rotguard.db", audit=AuditPolicy(mode="sampled", sample_rate=0.5))
...
rotguard.store.events(rotguard.session_id, "audit")  # loss, the facts lost, and what to change
```

The model is quizzed on facts taken from the removed text (each one checked to appear in it word for
word) and on your ledger entries, once with the old context and once with the new. A fact counts as
lost if only the old context answered it. `loss` is lost / answerable-before over facts that did
not come from tool output (tool output is mostly ids and listings that no summary keeps); `loss_all`
includes them, and every lost fact is listed either way. `loss` is `None` when the old context
answered none of the facts it scores. Modes: `off`, `sampled`, `every`, and `ci`, which runs synchronously:

```python
from rotguard.testing import assert_audit_loss

assert_audit_loss(rotguard.store, rotguard.session_id, max_loss=0.3)  # fails the test run
```

Limits worth knowing: a report compares one compaction with the context just before it, so a summary
of an already-lossy summary reports little loss; answers are matched as text, not judged for meaning
(a negated constraint can pass); and the questions are only as good as the facts the model extracts.
When extraction finds too few facts, `AuditPolicy(cloze=2)` (the default) adds facts built directly from
the text. In our small-model tests a "kept" verdict on a fact was reliable, and a "lost" one was a
warning rather than proof; the audit also never checked the probed fact in about 40% of failures.
In-place clearing by `ContextEditingMiddleware` repeats on every call, so it is audited once.

## Other frameworks

```python
# Anthropic SDK: recognises server-side compaction blocks (threshold and on-demand) and audits them
from rotguard.adapters.anthropic import RotguardAnthropic
rg = RotguardAnthropic(anthropic.Anthropic(), audit=AuditPolicy(mode="every"))
resp = rg.beta.messages.create(model=..., messages=..., betas=["compact-2026-01-12"], ...)

# LangGraph: a node before your LLM node; the ledger is kept in graph state, so checkpoints keep it
from rotguard.adapters.langgraph import RotguardGraph, RotguardState
```

See [src/rotguard/adapters/langgraph.py](src/rotguard/adapters/langgraph.py) for a full graph. The
Anthropic wrapper is tested against the documented block shapes and the SDK's own types, not against
the live API.

## Recall checks, handoff, CLI and MCP

`RecallPolicy(mode="every")` periodically asks the model to restate ledger entries and alerts after
repeated misses; `action="handoff"` can then restart the context from the ledger. A `rotguard`
command writes HTML reports, replays a saved trace under another model, and serves the ledger to MCP
clients (`rotguard[mcp]`). Recall checks have **not** been shown to warn before a task fails (first
benchmark: qwen2.5 3B, 9 scenarios, inconclusive), so use them to log and experiment. See the
[docs](docs/index.md): `docs/evaluation.md` has the numbers and caveats.

Every hook fails open: if rotguard errors, the agent's call goes ahead unchanged and the error is logged.
Logging uses [logquill](https://pypi.org/project/logquill/); call `rotguard.log.configure([...transports])` to see it.

## Development

```bash
uv sync --extra langchain --extra eval
uv run pytest
uv run python -m evals.run --model ollama:qwen2.5:3b --turns 40 --strategies none,summarize,summarize+ledger
uv sync --group docs && uv run mkdocs serve   # the docs site
```
