# Changelog

All notable changes to this project are documented in this file.

## Unreleased

- `refix.md` work, M1 to M4 (code and tests; nothing here has run on a hosted model):
  - **Recall probe redesign (R1 to R4).** The default probe (`style="cued"`) asks one
    fill-in-the-blank question per entry instead of one "list everything" question (`style="list"`
    keeps the old one). `target="auto"` asks about facts from the conversation that the ledger does
    not cover when the ledger is shown to the model on every call, and about the ledger otherwise;
    the probed context now includes the injected ledger note and marker, as the model sees them.
    A session-long `FactPool` keeps conversation facts after their source is compacted away.
    Answers that give a replaced value are `stale` and count as a miss. The Gate 3 harness now
    compounds compaction, runs a no-compaction control arm, reports calibrated `pass_at`, per-fact
    lead and overhead, and compares styles and targets on identical simulations.
  - **Result (qwen2.5 3B, `holdout`, 30 turns, 1 seed, 9 scenarios).** Recall score before any
    compaction went from 0.03 (old probe) to 0.82 (cued), meeting the 0.7 bar, and false alarms in
    control runs whose task passed went from 6 of 6 to 0 of 6. Early warning did **not** follow: the
    alert came at or after the task failure in 7 of 8 failing runs (median lead -1 turn), with one
    10-turn lead. The `history` target scored 0 of 510 turns because the oracle ledger covers every
    planted fact. Gate 3 is still not met. Details in `docs/evaluation.md`.
  - **Policies (F5, F4a).** `OffloadPolicy` stores large tool outputs and leaves a stub with the same
    `tool_call_id`, read back with `recall_output`; `DedupPolicy` collapses repeated outputs in the
    request only; `Session.pin` / `RotguardMiddleware.pin` / `ledger_record(pin=True)` keep a message
    through cleanup and re-attach it if another tool compacted it away; `StagedCleanup` clears old
    tool outputs first and summarizes only if still over budget (stage 2 turns itself off when another
    summarizer is active); `RecallPolicy(action="compact")` runs it once on an alert. A tail that
    must start on a user message is no longer required when a summary leads (a pure tool loop could
    never be summarized before). `TokenCounter` uses a provider's counting API when given one and
    otherwise estimates characters / 4 and says so.
  - **Ledger extractor (F1x).** `LedgerExtractor` proposes entries from each reply; an entry must
    share a distinctive token with the conversation, `confirm` can veto it, and entries carry
    `origin` (`tool`, `manual`, `extractor`).
  - **Architecture (A1 to A3).** Every stored event is also emitted as an OpenTelemetry span event
    when `rotguard[otel]` is installed (text excluded unless `telemetry.configure(include_text=True)`);
    a `Store` protocol with `open_store`; JSON schemas for the report and the stored `audit` and
    `recall` events, validated against the real producers.
  - **Evaluation tooling (E1, E2, E7, E8).** Eval clients count calls and tokens by purpose
    (provider-reported where available) and every entry point has `--estimate`; baselines
    `lc-summarize`, `anthropic-compaction` and `rg-staged`; stale-decision rate per result;
    `assert_history_integrity`; a nightly workflow and `evals/nightly.py`. `anthropic-compaction`
    was written from the SDK's type definitions and tested against a fake client, **not the live
    API**. ContextForge is not wired in.
  - Tests: 158 pass (99 before), ruff and strict mypy clean.

- License changed again, to a dual license: GNU AGPL-3.0 (`AGPL-3.0-only`, the full text in
  `LICENSE`) for everyone, plus a commercial license from the author for those who can't or won't
  comply with the AGPL. This replaces the PolyForm Noncommercial license below, which kept company
  teams from even trying the audit in CI. Copies obtained under MIT or PolyForm keep those terms.
  `CONTRIBUTING.md` now says contributions may be offered under both.
- Housekeeping: `ruff format --check` added to CI; `.vscode/` and `skills-lock.json` are no longer
  tracked; references to the private plan file replaced; package metadata (URLs, keywords,
  classifiers) and a tag-triggered release workflow added in preparation for a first PyPI alpha.

- (Superseded by the entry above.) License changed from MIT to the PolyForm Noncommercial License 1.0.0 (`LICENSE`, `pyproject.toml`
  SPDX metadata, README, CONTRIBUTING). Noncommercial use is free; commercial use needs a separate
  license. This is a source-available license, not an OSI open source one. Copies already obtained
  under MIT keep MIT.

- Phase 4 (pre-release, in progress): CLI and MCP server.
  - `rotguard` command (`sessions`, `report`, `blame`, `replay`, `mcp`) and `python -m rotguard.cli`.
  - `report`: one self-contained HTML page per session (recall curve with compactions marked,
    compaction and audit tables, ledger timeline); all stored text is escaped.
  - `replay`: rebuilds each turn's context from a saved session and re-runs only rotguard's side
    calls (audit, recall probes) under another model or policy (`--model anthropic|openai|ollama:`,
    `--audit`, `--recall-every`, `--pass-at`, `--inject-ledger`); the agent's own calls are not
    re-run. The source trace is untouched; results go to a new session.
  - `rotguard.adapters.mcp` (`rotguard[mcp]`, needs `mcp>=2`): tools `ledger_record`, `recall`,
    `blame`, `report` over stdio. The agent loads the ledger at start, so entries added here reach
    it on its next run.
  - `SQLiteStore.sessions()`.
  - Fresh-context handoff (F7): `rotguard.handoff.handoff` builds system messages, a ledger note
    and the last `keep` messages, widened so the tail starts on a user message and every tool
    result keeps its call. `RecallPolicy(action="handoff", handoff_keep=)`: when an alert fires,
    `RotguardMiddleware` and `RotguardGraph` replace the history on the next model call (the only
    place rotguard rewrites stored history, and opt-in). It waits while it would not shrink the
    context, and a passing probe cancels it. The Anthropic and OpenAI wrappers do not own the
    history; call `session.take_handoff(context)` and replace it yourself.
  - Tests: 99 pass (78 before Phase 4), ruff and strict mypy clean. The stdio server was checked
    with a raw `initialize` and `tools/list` exchange. The provider callers (`anthropic`, `openai`,
    `ollama`) are not exercised by tests; only their message shaping is.
  - Docs: a MkDocs Material site in `docs/` (home, quickstart, audit, recall and handoff, CLI and
    MCP, evaluation), built with `uv sync --group docs && uv run mkdocs build --strict`. The
    Evaluation page restates the measured results and says Gate 3 is unmet. Install instructions say
    "from source" because the package is not on PyPI.
  - Not done: benchmark report and launch posts. Gate 3 is
    not met (see Phase 3), so the early-warning claim must not be in launch material.

- Phase 3 (pre-release): recall checks.
  - `rotguard.probes.recall`: tier 3 recall probes. Every `every` turns a side call over the
    context the agent would use asks the model to restate sampled active ledger entries, graded
    per entry; `RecallMonitor` keeps a rolling score and alerts after `alert_after` failed probes
    in a row, never on one miss. `calibrate()` picks a per-model `pass_at` from clean-context
    probes. Off by default; probes never enter history.
  - `rotguard.probes.signals`: tier 1 passive signals (goal-word overlap, repeated replies, tool
    error rate), no model call. A flag runs a probe now instead of on schedule. Overlap is lexical,
    not embedding based.
  - Tier 2 marker: `RecallPolicy(marker=...)` adds a standing instruction to the injected ledger
    note; `Session.observe_reply` forces a probe when a reply lacks it. LangChain and LangGraph
    only (`RotguardGraph.observe_reply`); the Anthropic and OpenAI wrappers inject nothing.
  - `Session(recall=, on_alert=)`; `recall=` and `on_alert=` on `RotguardMiddleware`,
    `RotguardGraph` and `RotguardAnthropic`. Probes run on the audit's background worker.
  - `rotguard.adapters.openai_agents.RotguardSession`: wraps any Agents SDK session (trimming or
    summarizing), sees compactions at `get_items`, audits and probes through `responses.create`.
    Duck-typed; tested against a fake session, not the real SDK.
  - `evals/gate3.py`: replays a scenario turn by turn and reports the lead of the recall alert
    over the first task failure, misses, false alarms and probe calls per turn.
  - Test status (2026-10-19, branch `feat/phase-3-recall-checks`): 78 tests pass (was 60 before
    Phase 3; the new ones cover recall, signals, the marker, all four adapters and the Gate 3
    summary); ruff, ruff format and strict mypy are clean. All tests use fake models. The Agents
    SDK wrapper is tested against a fake session, not the real SDK. Gate 3 on a real model: see
    the Gate 3 entry below and `docs/evaluation.md`.
  - Gate 3, first run (qwen2.5 3B, `holdout`, 30 turns, 1 seed, 9 scenarios): not met. The recall
    score before any compaction averaged 0.11, so the alert fired at turn 1 in 8 of 9 runs; 3 of 9
    task probes failed before any compaction. The probe is not usable on this model as it stands,
    and the "lead over task failure" is not evidence of early warning. No hosted-model run yet.

- Audit hardening after Phase 2 (uncommitted work on top of PR #2):
  - Long extracted answers are graded on their distinctive tokens (identifiers and numbers), not
    as an exact sentence. A cloze fallback (`AuditPolicy.cloze`, default 2) builds facts straight
    from user and agent text when extraction finds fewer than `max_facts`: a sentence with an
    identifier or number is blanked and asked back. It needs no model call and works on models too
    weak to return the extraction JSON.
  - Fix: a quiz question that contained another fact's answer let the model copy it from the
    prompt, so a lost fact read as kept. Such facts are now dropped (`drop_leaky`). Matching is
    whole-token (`45` no longer matches inside `ref4558`). Both have regression tests.
  - `loss` now scores facts that did not come from tool output; `loss_all` keeps the old definition.
  - Evidence (qwen2.5 3B, 30 turns; one run per cell, so treat as indicative): on the tuning suites
    AUC rose from 0.61 to 0.79. A first held-out set then exposed the leakage bug above (AUC 0.55,
    one "kept" verdict followed by a failure). After the fix, a second held-out set of 9 new
    templates, never run before the configuration was fixed, gave AUC 0.90 (18 scored scenarios),
    "kept" 0 failures of 8, "lost" 8 failures of 10. The 1.5B model now runs the audit (24 of 30
    scenarios scored, from 1) but fails 24 of 27 probes, so its AUC (0.63) says little.
  - Not fixed: coverage. The audit never checked the probed fact in 5 of the 3B's 13 failures
    on the second held-out set, and in 11 of the 1.5B's 24. Nine templates, three seeds each: the
    effective sample is nine, so the p-values are optimistic.
  - Coverage: `run_audit` keeps at most one tool-origin fact, so small models no longer spend the
    fact slots on tool listings and the cloze fallback can fill them. qwen2.5 3B, 30 turns, 3 seeds,
    `holdout` + `holdout2` (54 scenarios, one run per cell): probed fact checked in 39 of 54 (was
    33), AUC 0.96 over 35 scored (was 0.90 over 31). These suites informed this change, so they are
    no longer clean held-out sets.
  - `evals`: two held-out suites (`holdout`, `holdout2`, selected by name only), `--cloze`,
    `ROTGUARD_OPENAI_API_KEY` for hosted runs (a placeholder `OPENAI_API_KEY` is rejected).

- Phase 2 (pre-release): compaction audit.
  - `rotguard.probes`: the audit. Facts come from the removed span (extracted by the model, kept
    only if they appear in it literally) and from the ledger; the same quiz is asked of the old and
    the new context and graded as text. Report: loss, facts lost, and suggestions. Policy modes
    `off` / `sampled` / `every` / `ci`, with a per-session budget. Opt-in.
  - `rotguard.session.Session`: observation, store writes and audit scheduling shared by all
    adapters (audits run on a background thread, or synchronously in `ci` mode).
  - `RotguardMiddleware(audit=...)`; in-place context-editing clearing is audited once.
  - `rotguard.adapters.anthropic.RotguardAnthropic`: recognises threshold and on-demand
    compaction blocks, audits them, and sends side calls with the request's tools and betas.
  - `rotguard.adapters.langgraph.RotguardGraph`: a node, a ledger tool, and a ledger kept in graph
    state so checkpointers persist it.
  - `rotguard.testing.assert_audit_loss` for CI.
  - `evals/gate2.py`: does the loss score predict later probe failures? Result on local models
    (qwen2.5 3B, qwen2.5-coder 7B, 30 turns, 50 scenarios, naive summarizer): the aggregate `loss`
    does not predict probe failure (AUC 0.61 and 0.44). Two exploratory follow-ups: excluding
    tool-origin facts (about 70% of what gets checked, mostly random ids) gives AUC 0.71 and 0.75
    (p about 0.09 and 0.10); and the audit's verdict on the probed fact itself was "kept" for 13
    scenarios, none of which failed, against 10 failures in 23 "lost" (pooled Fisher p = 0.0045,
    unadjusted). Treat the per-fact verdict and the list of lost facts as the useful output, not
    the single number. The 1.5B model cannot run the audit at all.
  - Fix: the fact-extraction prompt could be hijacked by an instruction inside the transcript
    (a transcript ending in "Summarize in one line" made a 3B model summarize instead of
    extracting). The transcript is now delimited and the task restated after it.

- Phase 1 core (pre-release, not published):
  - Neutral message model with provenance (`rotguard.core`), and the observer that diffs
    consecutive contexts and recognises compactions: removed items, rewritten items, summaries.
  - Ledger of goals, decisions, constraints and open questions with supersession
    (`rotguard.ledger`). Entries are never deleted.
  - SQLite store, plus `diff` and `blame` over recorded provenance (`rotguard.report`).
  - `RotguardMiddleware` for LangChain v1 (`rotguard.adapters.langchain`): a `ledger_record`
    tool, per-call ledger injection that never touches stored history, and fail-open hooks.
  - Verified middleware ordering against langchain 1.4.4: list rotguard after
    `SummarizationMiddleware` and `ContextEditingMiddleware` (plan section 7.2).
  - JSON schemas for the message and ledger entry (`schemas/`).
- Phase 0 evaluation harness (`evals/`): planted-facts, superseded-decision and goal-drift suites,
  `none` / `truncate` / `summarize` strategies, a `+ledger` variant, and `evals.rescore`.
- Logging through [logquill](https://pypi.org/project/logquill/).
