# rotguard

Measure and guard against context rot in LLM agents.

rotguard does not compact your context. It watches whatever does, keeps a **ledger** of the goals,
decisions and constraints that must survive, and records where every item in the context came from.

## What it gives you

| Feature | What it does | Cost |
| --- | --- | --- |
| [Ledger](quickstart.md#the-ledger) | Goals, decisions and constraints with supersession, shown to the model on every call | Free |
| Provenance, `diff`, `blame` | Where each item came from, and which summaries carry it | Free |
| [Compaction audit](audit.md) | After a compaction, which facts did the model lose? | About 3 side calls per audit |
| [Recall checks](recall.md) | Periodically ask pointed questions about the ledger or the conversation; alert on repeated failures | 1 side call per probe |
| [Policies](policies.md) | Offload large tool outputs, dedup repeats, pin messages, staged cleanup, ledger extractor | Free (summaries cost a call) |
| [Fresh-context handoff](recall.md#fresh-context-handoff) | Restart the context from the system prompt, the ledger and the last few messages | Free |
| [CLI and MCP server](cli.md) | HTML reports, replay a saved trace under another model, ledger tools for MCP clients | Replay: side calls |

Audits and recall checks are **off by default**. Every hook **fails open**: if rotguard errors, your
agent's call goes ahead unchanged and the error is logged.

## What it does not claim

The audit's per-fact verdict is a useful debugging signal in our small-model tests. Recall checks
have **not** been shown to warn before a task fails. See [Evaluation](evaluation.md) for the numbers
and the caveats before relying on either.

## Install

rotguard is not on PyPI yet (the version in the repo is `0.1.0a1`). Install from a clone:

```bash
git clone https://github.com/nikhilvdev/rotguard && cd rotguard
uv sync --extra langchain            # LangChain v1 middleware and the LangGraph node
# other extras: --extra anthropic, --extra openai, --extra mcp (needs mcp 2.x), --extra eval
```

The core (ledger, store, audit, CLI) has no framework dependency. Python 3.10 or newer. State is
kept in SQLite (standard library); pass a file path to keep it between runs.
