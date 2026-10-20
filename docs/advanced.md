# Telemetry, storage and schemas

## OpenTelemetry

```bash
pip install "rotguard[otel]"      # from a clone: uv sync --extra otel
```

Every event rotguard stores (compaction, audit, recall, signals, marker miss, handoff, staged
cleanup) is also emitted as an OpenTelemetry **span event** named `rotguard.<kind>`, so it shows up
in LangSmith and other tracing tools. If your agent is being traced, the event attaches to the current
span; otherwise it gets a short span of its own. With no tracer provider set, or without the package,
nothing happens.

Attributes are an allowlist of counts, scores, ids and flags: for example `rotguard.session_id`,
`rotguard.turn`, `rotguard.loss`, `rotguard.score`, `rotguard.alert`, `rotguard.removed_ids_count`,
`rotguard.tokens_before`. **Message text, lost answers and error messages are left out.** To add them:

```python
from rotguard import telemetry

telemetry.configure(include_text=True)
```

## Storage

State lives behind a small interface, `rotguard.store.Store` (a `typing.Protocol`): add and read
items, snapshots, ledger entries and events, list sessions. `SQLiteStore` is the only implementation
today. The adapters accept either a path (opened as SQLite) or any object that implements the
protocol, so another backend such as Postgres or Redis can be added without touching rotguard. The
protocol is tested by running the middleware on a store that exposes only its methods.

## JSON schemas

`schemas/` holds the language-neutral contract, generated with `uv run python -m rotguard.schemas`
and checked in CI:

| File | Describes |
| --- | --- |
| `message.schema.json` | the neutral message model |
| `ledger_entry.schema.json` | a ledger entry, including `origin` |
| `report.schema.json` | what `build()` and the MCP `report` tool return |
| `audit_event.schema.json` | a stored `audit` event |
| `recall_event.schema.json` | a stored `recall` event |

The report and event schemas forbid unknown fields, and a test validates real output against them, so
a changed payload fails a test instead of drifting.
