# Changelog

All notable changes to this project are documented in this file.

## Unreleased

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
