# Changelog

All notable changes to this project are documented in this file.

## Unreleased

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
