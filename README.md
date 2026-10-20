# rotguard

Measure and guard against context rot in LLM agents.

rotguard does not compact your context. It watches whatever does, keeps a **ledger** of the goals,
decisions and constraints that must survive, records where every item in the context came from, and
can check what a compaction lost.

> **Status: pre-release (`0.1.0a1`), not on PyPI yet.** The ledger, provenance and compaction audit are
> built and tested. Recall checks are built but have **not** been shown to warn before a task fails
> (see [Evaluation](#evaluation)). Nothing has been run on a hosted model yet.

## Contents

[The problem](#the-problem) · [Features](#features) · [Install](#install) · [Quickstart](#quickstart-langchain-v1) ·
[Adapters](#adapters) · [Ledger](#the-ledger) · [Provenance](#provenance-diff-and-blame) · [Audit](#audit-what-a-compaction-lost) ·
[Recall checks](#recall-checks) · [Handoff](#fresh-context-handoff) · [CLI](#command-line) · [MCP](#mcp-server) ·
[Evaluation](#evaluation) · [Design rules](#design-rules) · [Status](#project-status) · [Development](#development)

## The problem

Long-running agents compact their context: they summarize, trim or clear old tool output. Each step
can silently drop a constraint ("only write to `staging_orders`") that a later step depends on. The
agent then fails far from where the information was lost, and nothing records that it happened.

rotguard answers three questions:

1. **What must survive?** A ledger the agent and you maintain, shown to the model on every call.
2. **Where did this come from?** Provenance for every item, through any number of summaries.
3. **What did this compaction lose?** An opt-in audit that quizzes the model before and after.

## Features

| Feature | What it does | Cost | Maturity |
| --- | --- | --- | --- |
| [Ledger](#the-ledger) | Goals, decisions, constraints with supersession; shown per call, never stored in history | Free | Tested |
| [Provenance](#provenance-diff-and-blame) | `diff` and `blame` across compactions | Free | Tested |
| [Compaction audit](#audit-what-a-compaction-lost) | Which facts did a compaction lose? Per-fact verdicts and suggestions | About 3 side calls | Useful as a debugging aid |
| [Recall checks](#recall-checks) | Periodic fill-in-the-blank probes about the ledger or the conversation; alert on repeated misses | 1 side call per probe | **Experimental** |
| [Policies](#policies) | Offload large tool outputs, dedup repeats, pin messages, staged cleanup, optional ledger extractor | Free (summaries cost a call) | **Experimental**, LangChain only |
| [Handoff](#fresh-context-handoff) | Restart the context from the system prompt, the ledger and the last few messages | Free | **Experimental** |
| [CLI](#command-line) | HTML report, `blame`, replay a trace under another model | Replay: side calls | Tested |
| [MCP server](#mcp-server) | `ledger_record`, `recall`, `blame`, `report` for MCP clients | Free | Tested |

Audits and recall checks are **off by default**. Every hook **fails open**: if rotguard errors, your
agent's call goes ahead unchanged and the error is logged.

## Install

Not published yet. Install from a clone:

```bash
git clone https://github.com/nikhilvdev/rotguard && cd rotguard
uv sync --extra langchain        # LangChain v1 middleware and the LangGraph node
```

| Extra | For |
| --- | --- |
| `langchain` | `RotguardMiddleware`, `RotguardGraph` |
| `anthropic` | `RotguardAnthropic`, `--model anthropic:` in the CLI |
| `openai` | `RotguardSession` side calls, `--model openai:` and `ollama:` in the CLI |
| `mcp` | the MCP server (needs `mcp` 2.x) |
| `eval` | the benchmark harness in `evals/` |

The core (ledger, store, audit, CLI) needs no framework. Python 3.10 or newer. State is kept in SQLite
from the standard library.

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

List rotguard **after** `SummarizationMiddleware` and `ContextEditingMiddleware` so it sees what they
did (a test pins this). The agent gets a `ledger_record` tool, the active ledger is added to each
model request, and compactions by any middleware are detected and stored with provenance.

## Adapters

| | LangChain | LangGraph | Anthropic SDK | OpenAI Agents SDK |
| --- | --- | --- | --- | --- |
| Class | `RotguardMiddleware` | `RotguardGraph` | `RotguardAnthropic` | `RotguardSession` |
| Shows the ledger to the model | Yes | Yes (`rg.with_ledger`) | No | No |
| `ledger_record` tool for the agent | Yes | Yes (`rg.tools`) | No | No |
| Compaction audit | Yes | Yes (pass `model=`) | Yes (server-side compaction blocks) | Yes |
| Recall checks | Yes | Yes (pass `model=`) | Yes | Yes |
| Marker check | Yes | Yes (`rg.observe_reply`) | No | No |
| Handoff | Automatic | Automatic (in the node) | Manual (`session.take_handoff`) | Manual |
| Policies (offload, dedup, pin, staged cleanup, extractor) | Yes | No | No | No |

- **LangGraph:** add `rg.node` before your LLM node and mix `RotguardState` into your state, so the
  ledger is checkpointed and a resumed thread in a new process loads it back. See
  [`langgraph.py`](src/rotguard/adapters/langgraph.py).
- **Anthropic:** wraps `client.beta.messages`; side calls resend your `tools` and `betas` and keep the
  signed compaction block exactly as returned. Tested against the documented block shapes and the
  SDK's types, **not the live API**.
- **OpenAI Agents SDK:** wraps any session with `get_items`, `add_items`, `pop_item`, `clear_items`
  (a trimming or summarizing session). Duck-typed and tested against a fake session, **not the real
  SDK**.

The Anthropic and OpenAI wrappers observe only. Record ledger entries yourself with
`rg.ledger.record(...)`.

## The ledger

```python
rotguard.ledger.record("constraint", "Only write to staging_orders.", key="table")
rotguard.ledger.record("constraint", "Use archive_orders instead.", key="table")  # replaces it
```

Kinds: `goal`, `decision`, `constraint`, `question`. Entries sharing a `key` (or named in
`supersedes`) replace each other. **Entries are never deleted:** the old one is kept and marked
`superseded`. The active entries are added to the end of each model request for that call only and
are never written into your stored history.

## Provenance, diff and blame

```python
from rotguard.report import blame, diff

blame(rotguard.store, rotguard.session_id, "staging_orders")  # which message introduced it?
diff(rotguard.store, rotguard.session_id, 3, 5)  # what changed between turns 3 and 5?
```

`blame` finds the earliest original items containing the text, the summaries that still carry it,
and the source items behind those summaries, through any number of rounds of summarizing.

## Audit: what a compaction lost

Off unless you ask, because each audit costs about three extra model calls: one to extract facts from
the removed span, then one quiz on the old context and one on the new.

```python
from rotguard.probes import AuditPolicy

rotguard = RotguardMiddleware("rotguard.db", audit=AuditPolicy(mode="sampled", sample_rate=0.5))
...
rotguard.store.events(rotguard.session_id, "audit")  # loss, the facts lost, and what to change
```

The model is quizzed on facts taken from the removed text (each checked to appear in it word for
word) and on your ledger entries. A fact counts as **lost** if only the old context answered it.
`loss` is lost / answerable-before over facts that did not come from tool output (mostly ids and
listings that no summary keeps); `loss_all` includes them, and every lost fact is listed either way.
Modes: `off`, `sampled`, `every`, and `ci` (synchronous):

```python
from rotguard.testing import assert_audit_loss

assert_audit_loss(rotguard.store, rotguard.session_id, max_loss=0.3)  # fails the test run
```

**Limits:** a report compares one compaction with the context just before it, so a summary of an
already-lossy summary reports little loss. Answers are matched as text, not judged for meaning (a
negated constraint can pass). The questions are only as good as the extracted facts; when extraction
finds too few, `AuditPolicy(cloze=2)` (the default) adds facts built directly from the text. A clean
report is not proof that nothing was lost: in our tests the audit never checked the probed fact in
about 28% of scenarios. Side calls are never written to history. See [docs/audit.md](docs/audit.md).

## Recall checks

Every few turns, one side call asks the model fill-in-the-blank questions about facts it was given
("Complete the constraint currently in effect: Only write to ____."). A miss means it may no longer
hold that fact. When the ledger is shown to the model on every call, the probe asks about facts from
the conversation that the ledger does not cover (the model could always answer from the ledger);
otherwise it asks about the ledger. An answer that gives a replaced value counts as a miss. Off by
default.

```python
from rotguard.probes import RecallPolicy

rotguard = RotguardMiddleware(
    "rotguard.db",
    recall=RecallPolicy(mode="every", every=5),
    on_alert=lambda result: print("recall failing:", result.missed),
)
```

Three tiers, free to paid: **passive signals** (goal words missing, repeated replies, tool errors),
a **marker** instruction the model must keep following, and the **recall probe** itself. A free tier
flagging makes the next probe run now. One miss is noise: an alert fires once, after `alert_after`
failed probes in a row. `rotguard.probes.calibrate` picks a per-model `pass_at` from clean-context
scores. **Not yet shown to warn early**; see [Evaluation](#evaluation) and
[docs/recall.md](docs/recall.md).

## Policies

Opt-in, LangChain middleware only, and each keeps tool calls paired with their results:
`OffloadPolicy` (store large tool outputs, leave a stub, read back with `recall_output`),
`DedupPolicy` (collapse repeated outputs for one call), pinning (`rotguard.pin(id)`, or
`ledger_record(pin=True)`), `StagedCleanup` (clear old tool outputs first, summarize only if still over
budget), and `LedgerExtractor` (propose ledger entries from each reply, validated against the
conversation). Offload and staged cleanup rewrite stored history; nothing is deleted. See
[docs/policies.md](docs/policies.md).

## Fresh-context handoff

`RecallPolicy(action="handoff")` restarts the context on alert: the system messages, a ledger note
and the last `handoff_keep` messages, widened so the tail starts on a user message and every tool
result keeps its call. It **rewrites stored history** (the only place rotguard does, and opt-in),
waits while it would not shrink the context, and a passing probe cancels it. Because recall alerts
are unproven, be careful with this action.

## Command line

```text
rotguard sessions trace.db
rotguard report   trace.db -o report.html
rotguard blame    trace.db "only write to staging_orders"
rotguard replay   trace.db --model ollama:qwen2.5:3b --audit every --recall-every 2 -o replay.html
rotguard mcp      trace.db --session my-session
```

`report` writes a self-contained HTML page: recall curve with compactions marked, compaction and
audit tables, ledger timeline. `replay` re-runs only rotguard's own side calls (audits and probes) on
a saved session under another model or policy; your agent's calls are not re-run, and the source
trace is untouched. Providers: `anthropic:`, `openai:`, `ollama:`. See [docs/cli.md](docs/cli.md).

## MCP server

`rotguard mcp` serves a session over stdio to an MCP client such as Claude Code (`rotguard[mcp]`).

| Tool | What it does |
| --- | --- |
| `ledger_record` | Record a goal, decision, constraint or question |
| `recall` | Fetch an item that was compacted away, by id or by text |
| `blame` | Where text first appeared and which summaries carry it |
| `report` | Probes, audits, compactions and the ledger as data |

A running agent loads the ledger when it starts, so entries added through MCP reach it on its next
run, not mid-run.

## Evaluation

Everything below is small, synthetic and on local models (qwen2.5 3B unless noted), usually one run
per cell, with about nine scenario templates, so the effective sample is nine and the p-values are
optimistic. Details and caveats: [docs/evaluation.md](docs/evaluation.md) and `CHANGELOG.md`.

| Question | Result |
| --- | --- |
| Does the audit's aggregate `loss` predict a later failure? | **No** (AUC 0.61 and 0.44; 3B and 7B, 50 scenarios) |
| Does its verdict on the one probed fact? | **As a flag, yes.** "Kept": 0 of 13 failed; "lost": 10 of 23 failed |
| Held-out check after fixes (3B, 9 new templates) | AUC 0.90 over 18 scored; "kept" 0 of 8 failed |
| Does the audit check the fact that matters? | In 39 of 54 scenarios; the suites that informed this are no longer clean held-out sets |
| Does the recall probe work on a clean context? | **Yes after a redesign** (3B, 9 scenarios, 1 seed): 0.82, up from 0.03 for the old single question; 0 false alarms in 6 control runs |
| **Do recall probes fail before the task does?** | **No.** The alert came at or after the task failure in 7 of 8 failing runs (median lead -1 turn); one run had a 10-turn lead |

Not done: any hosted-model run (Anthropic or OpenAI), the plan's baselines on two models, and evaluating the `history` probe target (the harness's oracle ledger leaves nothing for it to ask).

## Design rules

- **Fail open.** Adapter hooks catch errors, log them and let the agent's call proceed unchanged.
- **Non-destructive.** The ledger is added to the model request for that call only. The one exception is the opt-in handoff.
- **Nothing is deleted.** A newer ledger entry marks the older one `superseded`.
- **Side calls are invisible.** Audits and probes never enter history or the stored items, and run on a background thread (or inline in `ci` mode).
- **No framework in the core.** Each adapter is an optional extra.

## Project status

The project is built in five phases, each ending in a gate.

| Phase | Status |
| --- | --- |
| 0 Validate | Partly done: harness and landscape checks done; name reservation and two-model baselines open |
| 1 Core alpha | Done |
| 2 Audit | Done; Gate 2 took its fallback (the audit is a debugging aid) |
| 3 Recall checks | Built; **Gate 3 not met** |
| 4 Launch | CLI, MCP, handoff and docs built; benchmark report and launch posts not started |

## Repository layout

```text
src/rotguard/
  core/        neutral message model and the observer (no framework imports)
  ledger/      entries and supersession
  store/       SQLite store
  report/      diff, blame, HTML report
  probes/      audit, recall probes, passive signals, quiz and grading
  adapters/    langchain, langgraph, anthropic, openai_agents, mcp
  cli/         the `rotguard` command
  session.py   observation, store writes and probe scheduling shared by every adapter
  handoff.py   fresh-context handoff        replay.py   replay a trace
evals/         benchmark harness (a dev tool, not shipped)
schemas/       generated JSON schemas        docs/   the MkDocs site
```

## Development

```bash
uv sync --extra langchain --extra anthropic --extra openai --extra mcp --extra eval
uv run pytest                    # fake models only, no API keys
uv run ruff check . && uv run ruff format .
uv run mypy                      # strict, src/rotguard only
uv run python -m evals.run --model ollama:qwen2.5:3b --turns 40 --strategies none,summarize,summarize+ledger
uv sync --group docs && uv run mkdocs serve   # the docs site
```

Logging uses [logquill](https://pypi.org/project/logquill/); call `rotguard.log.configure([...transports])`
to see it. Contributions are welcome: see [CONTRIBUTING.md](CONTRIBUTING.md) and the
[Code of Conduct](CODE_OF_CONDUCT.md).

## License

rotguard is dual-licensed:

1. **Open source: [GNU AGPL-3.0](LICENSE)** (`AGPL-3.0-only`). Anyone may use, modify and share it.
   If you modify rotguard, or build it into software you distribute or run as a network service, the
   AGPL requires you to make the corresponding source available under the same license.
2. **Commercial license.** If you can't or don't want to comply with the AGPL, for example in a
   proprietary product or a hosted service whose source you will not release, you can get a
   commercial license from the author: open an issue at
   [github.com/nikhilvdev/rotguard/issues](https://github.com/nikhilvdev/rotguard/issues).

The AGPL's source-sharing duties arise when you distribute rotguard or offer a modified version to
users over a network; running it privately, for example to test or debug your own agents, generally
does not trigger them. Whether a particular product is covered is a legal question, and this
summary is not legal advice; the [LICENSE](LICENSE) file is what governs.

Earlier copies of rotguard that were published under the MIT or the PolyForm Noncommercial license
stay under those terms for the people who received them.
