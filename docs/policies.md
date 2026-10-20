# Policies

Thin policies that keep the context small without losing anything. They are **off by default**, they
work on the LangChain middleware (the only adapter that owns the message history), and every one of
them keeps tool calls and their results paired.

```python
from rotguard.adapters.langchain import RotguardMiddleware, model_summarizer
from rotguard.policies import DedupPolicy, OffloadPolicy, StagedCleanup

rotguard = RotguardMiddleware(
    "rotguard.db",
    offload=OffloadPolicy(max_tokens=2000, preview_lines=10),
    dedup=DedupPolicy(min_chars=200),
    staged=StagedCleanup(
        budget_tokens=60_000,
        clear_keep=3,
        keep_messages=6,
        summarizer=model_summarizer(model),  # a LangChain chat model; omit to turn stage 2 off
    ),
)
```

!!! warning "Some of these rewrite stored history"
    **Offload** and **staged cleanup** replace messages in the stored history (the only rotguard
    features that do, besides the handoff), so they are opt-in. Nothing is deleted: the full text of
    every offloaded output stays in the store. **Dedup** only changes the copy sent to the model.

## Offload large tool outputs

When a tool returns more than `max_tokens` (estimated), the full text is stored as an item of kind
`offloaded` and the context gets a stub: the first `preview_lines` lines (at most `preview_chars`
characters), a note, and the item's id. The stub keeps the **same `tool_call_id`**, so the call and its
result stay paired.

The agent gets a `recall_output(id, start_line=0, max_lines=200)` tool to read the rest in pages.
`blame` and the MCP `recall` tool still find the text. If offloading itself fails, the original output
is returned unchanged.

## Dedup repeated outputs

If a tool returns the same text more than once (at least `min_chars` long), later copies become
`[rotguard: same output as <id>]` **in the request for that call only**. Stored history is unchanged,
the first copy stays, and user and assistant messages are never touched.

## Pin messages

```python
rotguard.pin(message_id)  # keep this message
# or let the agent do it when it records something:  ledger_record(..., pin=True)
```

A pinned message is exempt from offload, dedup, staged cleanup and the handoff (with the tool call or
result it is paired with). If another tool, such as `SummarizationMiddleware`, drops a pinned message
from a request, its text is added after the ledger note for that call, labelled as pinned. Pins are
stored, so they survive a restart.

## Staged cleanup

LangChain's built-in middleware cannot say "clear old tool outputs first, and summarize only if that
was not enough". `StagedCleanup` does, when the context is over `budget_tokens`:

1. **Clear.** Offload every tool result except the newest `clear_keep`. If that gets under budget,
   stop: no model call is made.
2. **Summarize, only if still over.** Summarize everything before the last `keep_messages` messages,
   except system messages and pinned ones, with your `summarizer`. The summary is recorded with the
   ids it covers, so `blame` works through it.

Each run is stored as a `staged` event (stages, counts, tokens before and after) and the history
change is recorded as a compaction, so the [audit](audit.md) checks it like any other.
`RecallPolicy(action="compact")` can trigger it from a [recall alert](recall.md#compact-on-alert).

If another summarizer is active (a summary this policy did not write appears in the history), stage 2
turns itself off with a warning so two summarizers do not fight. Stage 1 keeps working.

## Token counting

Budgets use `rotguard.tokens.TokenCounter`. By default it **estimates** characters divided by 4,
which is rough (code and non-English text use more tokens) and is flagged as `tokens_estimated` in the
events. Pass a `counter=TokenCounter(provider)` to use a provider's counting API; if the provider
errors, it falls back to the estimate instead of failing the agent. `anthropic_counter(client, model)`
is one such provider.

## Ledger extractor

An agent that forgets to call `ledger_record` loses its constraints silently. A `LedgerExtractor`
makes one side call after each reply and proposes entries:

```python
from rotguard.adapters.langchain import LangChainCaller
from rotguard.probes import LedgerExtractor

rotguard = RotguardMiddleware(
    extract=LedgerExtractor(LangChainCaller(model), every=1, confirm=lambda entry: True),
)
```

Everything it proposes is checked: the kind must be allowed (`goal`, `decision`, `constraint`), the
text short, not already in the ledger, and it must **share a distinctive token** (an identifier, a
number or a rare word) with the messages it came from, so a model cannot invent a constraint the
conversation does not support. `confirm` can veto any entry. Entries record who added them in
`origin`: `tool`, `manual` or `extractor`. The excerpt is delimited and labelled as data to make
prompt injection harder, but a model can still be swayed by text in the conversation. What protects
the ledger is the check above, and `confirm` if you need a human or a rule in the loop.
