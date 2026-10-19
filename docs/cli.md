# CLI and MCP server

The `rotguard` command works on the SQLite file an adapter writes (`RotguardMiddleware("trace.db")`).
If the file holds several sessions, add `--session ID`.

```text
rotguard sessions trace.db
rotguard report   trace.db -o report.html
rotguard blame    trace.db "only write to staging_orders"
rotguard replay   trace.db --model ollama:qwen2.5:3b --audit every --recall-every 2 -o replay.html
rotguard mcp      trace.db --session my-session
```

## report

Writes one self-contained HTML page: a recall curve with compactions marked, compaction and audit
tables, and the ledger timeline. Stored text is escaped. Without `-o` it prints to stdout.

## blame

Prints the items where the text first appeared, the summaries that still carry it, and the source
items behind those summaries.

## replay

Re-runs rotguard's side calls (audits and recall probes) on a saved session under another model or
policy. It rebuilds each turn's context from the stored snapshots. **Your agent's own calls are not
re-run**, only rotguard's. The source trace is untouched and the results go to a new session.

| Flag | Default | Meaning |
| --- | --- | --- |
| `--model` | required | `anthropic:<model>`, `openai:<model>` or `ollama:<model>` |
| `--audit` | `every` | `off`, `sampled`, `every`, `ci` |
| `--recall-every N` | `0` (off) | probe every N turns |
| `--pass-at` | `0.5` | recall pass threshold |
| `--inject-ledger` | off | put the ledger in front of the quiz |
| `--db-out FILE` | in memory | keep the replay's events in a SQLite file |
| `-o FILE` | stdout | write the HTML report |

Providers need their package (`anthropic` or `openai`; Ollama uses the `openai` package against
`http://localhost:11434/v1`) and the usual API key in your environment. Replay audits always run
synchronously.

## MCP server

`rotguard mcp` serves the session over stdio to an MCP client such as Claude Code. It needs
`rotguard[mcp]` (the `mcp` 2.x package).

| Tool | What it does |
| --- | --- |
| `ledger_record` | Record a goal, decision, constraint or question; `key` or `supersedes` replace an earlier entry |
| `recall` | Fetch an item that was compacted away, by id or by a piece of its text (at most 3 hits, 2,000 characters each) |
| `blame` | Where text first appeared and which summaries carry it |
| `report` | Recall probes, audits, compactions and the ledger as data |

!!! note "The agent loads the ledger when it starts"
    The server writes entries to the SQLite file. A running agent that wrote the file loads its
    ledger once, at the start, so entries recorded through MCP reach it on its **next run**, not
    mid-run.

For example, in Claude Code:

```bash
claude mcp add rotguard -- rotguard mcp /path/to/trace.db --session my-session
```
