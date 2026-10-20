# rotguard: Refix Plan

What is missing or needs fixing in `nikhilvdev/rotguard`, measured against the project plan (`plan.md`, 11 Oct 2026), with the exact change, how to test it and when it counts as done.

| | |
| --- | --- |
| **Repo reviewed** | `main` at `88f78ca` (19 Oct 2026, PR #6) |
| **Review date** | 20 Oct 2026 |
| **State at review** | 99 tests pass, `ruff check` clean, strict `mypy` clean, `ruff format --check` flags `README.md` line 133 |
| **Gates** | Gate 1 not recorded, Gate 2 took its fallback, Gate 3 not met |

## How to use this file

Every item has an ID, a priority and an effort estimate.

- **P0:** blocks the gates or the launch, or can't be undone later. Do first.
- **P1:** needed for v1.0 as the plan defines it.
- **P2:** worth doing; can slip past v1.0.
- **Effort:** S is under half a day, M is 1 to 3 days, L is more than 3 days.

Each item lists **Why**, **Where** (files), **Change**, **Tests** and **Done when**. Work through the milestones in order; items inside a milestone can run in parallel.

---

## Contents

1. [Summary of all items](#1-summary-of-all-items)
2. [Milestones and order](#2-milestones-and-order)
3. [M0: Claim the name and clean up](#3-m0-claim-the-name-and-clean-up)
4. [M1: Redesign the recall probe](#4-m1-redesign-the-recall-probe)
5. [M2: Evaluation on hosted models](#5-m2-evaluation-on-hosted-models)
6. [M3: Thin policies (F5)](#6-m3-thin-policies-f5)
7. [M4: Remaining features and architecture](#7-m4-remaining-features-and-architecture)
8. [M5: Launch](#8-m5-launch)
9. [Decisions to make](#9-decisions-to-make)
10. [Updated gates](#10-updated-gates)
11. [Rules that still apply](#11-rules-that-still-apply)
12. [Sources](#12-sources)

---

## 1. Summary of all items

| ID | Item | Area | Priority | Effort | Depends on |
| --- | --- | --- | --- | --- | --- |
| H1 | Claim `rotguard` on PyPI with a real alpha | Launch | **P0** | S | D1 |
| H2 | Fix README formatting; add `ruff format --check` to CI | Hygiene | P1 | S | |
| H3 | Fix dangling `rotguard.md` references | Hygiene | P1 | S | |
| H4 | Stop committing `.vscode/` and `skills-lock.json` | Hygiene | P2 | S | |
| R1 | Cued, per-entry recall questions instead of one "list everything" question | Recall | **P0** | M | |
| R2 | Probe the context the agent actually sees; target facts the ledger doesn't cover | Recall | **P0** | M | R1 |
| R3 | Detect stale beliefs in probe answers | Recall | P1 | S | R1 |
| R4 | Fix the Gate 3 harness (compounding compaction, clean control arm, more seeds) | Eval | **P0** | M | R1, R2 |
| E1 | Token and cost tracking in eval clients | Eval | **P0** | S | |
| E2 | Real baselines: LangChain summarizer, Anthropic compaction, ContextForge | Eval | **P0** | M | E1 |
| E3 | Gate 1 run: ledger vs no ledger on hosted models | Eval | **P0** | M | E1, E2 |
| E4 | Re-run Gates 2 and 3 on hosted models | Eval | **P0** | M | R1–R4, E1, E2 |
| E5 | Suite 4: replicate part of Chroma's context-rot results | Eval | P1 | M | E1 |
| E6 | Suite 5: real agent traces | Eval | P1 | L | |
| E7 | Missing metrics: stale-decision rate, overhead %, broken histories | Eval | P1 | S | E1 |
| E8 | Nightly CI eval on a small subset | Eval | P1 | S | E1 |
| E9 | Fill in the open targets in the evaluation table | Eval | P1 | S | E3, E4 |
| F5a | Tool-output offload with a `recall_output` tool | Policies | P1 | M | A4 |
| F5b | Dedup repeated tool outputs per call | Policies | P1 | S | |
| F5c | Pinning | Policies | P1 | M | |
| F5d | Staged cleanup: clear first, summarize only if still over budget | Policies | P1 | L | F5a, A4 |
| F1x | Optional ledger extractor | Ledger | P1 | M | |
| F4a | `action="compact"` on recall alert | Recall | P2 | S | F5d |
| T1 | Live tests for the Anthropic and OpenAI adapters | Adapters | P1 | S | |
| A1 | OpenTelemetry events | Architecture | P1 | M | |
| A2 | Store protocol | Architecture | P2 | S | |
| A3 | JSON schema for reports and events | Architecture | P2 | S | |
| A4 | Token counting helper | Architecture | P1 | S | |
| L1 | Benchmark report "What does compaction cost?" | Launch | P1 | L | E2–E4 |
| L2 | LangChain middleware-ordering guide | Launch | P1 | S | |
| L3 | Sample HTML report from a real run | Launch | P1 | S | E4 |
| L4 | README tagline, PyPI metadata, GitHub topics | Launch | P1 | S | |
| L5 | Launch posts | Launch | P1 | M | L1 |
| D1 | License decision | Decision | **P0** | S | |
| D2 | AnchorLoop integration: now or after v1 | Decision | P2 | S | |

---

## 2. Milestones and order

| Milestone | Items | Ends with |
| --- | --- | --- |
| **M0: Claim and clean** | D1, H1–H4 | Name claimed on PyPI; CI checks formatting |
| **M1: Probe redesign** | R1, R2, R3, R4 | Recall probes pass on clean contexts in fake-model tests and on the 3B model |
| **M2: Hosted evaluation** | E1–E4, E7, E9 | Gates 1, 2 and 3 have results on one Anthropic and one OpenAI model |
| **M3: Policies** | A4, F5a–F5d, F4a | Staged cleanup works and its compactions are audited |
| **M4: Remaining features** | F1x, T1, A1–A3, E5, E6, E8 | Everything in plan section 6 is built or explicitly deferred |
| **M5: Launch** | L1–L5 | v0.1.0 on PyPI, benchmark report public |

M1 comes before M2 so the paid hosted runs test the fixed probe, not the broken one. One batch of hosted runs then answers Gates 1, 2 and 3 together.

---

## 3. M0: Claim the name and clean up

### D1. License decision (P0, S)

**Why.** `LICENSE` is PolyForm Noncommercial 1.0.0. The plan's target users are developers building LangChain agents, most of them at companies, and its strongest adoption signal is "teams running the audit in CI". Under this license those teams need a separate commercial license before they can try it. This shapes the README, the PyPI listing and every launch post, so settle it before H1 and L5.

**Options** (summary only, not legal advice):

| Option | Who can use it freely | Fits if you want |
| --- | --- | --- |
| Keep PolyForm Noncommercial | Individuals, research, nonprofits | To sell every commercial license yourself |
| AGPL-3.0 plus a commercial license | Everyone, if they share changes when they offer it as a service | Wide use, with paid licenses for companies that won't accept AGPL |
| Business Source License, converting to Apache-2.0 after a set date | Everyone except the uses you exclude, until the date | Time-limited commercial protection |
| Apache-2.0 or MIT | Everyone | Maximum adoption and contributions (the original plan) |

**Done when:** the license is chosen and `LICENSE`, `pyproject.toml` (`license`), `README.md` and `CONTRIBUTING.md` all match it.

### H1. Claim `rotguard` on PyPI with a real alpha (P0, S)

**Why.** `rotguard` was still unclaimed on PyPI and npm on 20 Oct 2026. Losing the name is the only risk here that can't be undone.

**Change.**

1. Set `version = "0.1.0a1"` in `pyproject.toml`.
2. Publish the current code as a real pre-release, not an empty placeholder: PyPI discourages name squatting, and a working alpha is more useful anyway.
3. Use PyPI trusted publishing from a GitHub Actions workflow (`.github/workflows/release.yml`, triggered by a version tag), so no API token is stored.
4. Add `[project.urls]` (Homepage, Documentation, Issues, Changelog), `keywords` (`context rot`, `langchain`, `llm agents`, `compaction`, `context engineering`) and classifiers (`Development Status :: 3 - Alpha`, Python versions, license).
5. npm: claim the name only if a JavaScript port is genuinely planned, and then with a minimal real package that points to the Python project. npm also discourages squatting.

**Done when:** `pip install rotguard==0.1.0a1` works, and the PyPI page shows the README, links and keywords.

### H2. README formatting and a CI format check (P1, S)

**Where:** `README.md` line 133, `.github/workflows/ci.yml`.

**Change.** Run `uv run ruff format .`, then add `- run: uv run ruff format --check .` after `ruff check` in `ci.yml`.

**Done when:** CI fails on an unformatted file.

### H3. Dangling `rotguard.md` references (P1, S)

**Where:** `CHANGELOG.md` ("see `rotguard.md` 10.2") and the README's Project status section ("The project plan (kept locally, not in this repository)").

**Change.** Either commit the plan as `docs/plan.md` and link it, or replace each reference with the result it pointed to. Committing it is better: contributors can then see the gates the README refers to.

**Done when:** no file refers to a document outside the repo.

### H4. Editor and tool files (P2, S)

**Change.** Add `.vscode/` and `skills-lock.json` to `.gitignore` and `git rm --cached` them, unless `skills-lock.json` is meant for contributors. If it is, say what it's for in `CONTRIBUTING.md`.

---

## 4. M1: Redesign the recall probe

Gate 3 failed because the probe failed on clean contexts: on qwen2.5 3B the recall score averaged 0.11 *before* any compaction, so the alert fired at turn 1 in 8 of 9 runs (`docs/evaluation.md`). The code shows two design causes, which R1 and R2 fix. Fix both before spending money on hosted runs.

### R1. Cued, per-entry questions (P0, M)

**Why.** `run_recall` (`src/rotguard/probes/recall.py`) asks a single open question, `LEDGER_QUESTION` in `src/rotguard/probes/quiz.py`: "List every goal, decision and constraint currently in effect for this task, with exact names, numbers and identifiers." It then grades every sampled entry against that one answer. That is free recall: asked to list everything with no cue, a model leaves things out even when they are still in its context. A pointed question per item is far more reliable. The audit already uses per-fact questions, and the cloze builder (`cloze_facts`) already shows how to make them without a model call.

**Change.**

1. Add `cued_ledger_facts(entries) -> list[Fact]` to `probes/quiz.py`. For each active entry:
    - If `fact_tokens(entry.text)` finds an identifier or number, blank its first occurrence and ask: `Complete the {kind} currently in effect: <text with ____>`. The answer is the blanked token.
    - If not, blank the longest content word not in `_STOP`.
    - For `goal` entries with no distinctive token, ask `What is the current goal of this task?` and grade on content words, as `fact_tokens` already does.
2. Run `drop_leaky` over the sampled questions, so no question contains another one's answer.
3. Put all sampled questions in one numbered prompt with `build_prompt`, and grade each answer on its own line with `parse_numbered` and `grade`. This is one side call, as now.
4. Add `style: Literal["cued", "list"] = "cued"` to `RecallPolicy`. Keep `"list"` so the old behaviour can still be compared in evals.

**Tests** (`tests/test_recall.py`):

- A fake model that answers each cloze correctly scores 1.0; one that answers `unknown` scores 0.
- A question containing another fact's answer is dropped.
- An entry with no identifier still produces a gradable question.
- `style="list"` reproduces the current behaviour.

**Done when:** on qwen2.5 3B, the mean recall score on turns before any compaction is at least 0.7. Record the number in `docs/evaluation.md` either way.

### R2. Probe what the agent sees; target what the ledger misses (P0, M)

**Why.** `Session.run_pending` calls `run_recall_check(self._prev_req or self._prev_hist, ...)`. `_prev_req` is recorded in `observe_request` *before* `RotguardMiddleware._prepare` appends the ledger note (`src/rotguard/adapters/langchain.py`). So the probe tests a context without the ledger, and `docs/recall.md` says this is deliberate. But `inject_ledger=True` is the default, so the agent always has the ledger in front of it. A probe miss on a ledger entry then measures a risk the agent doesn't face, and can't predict a task failure. The probe's real value is catching what the ledger does **not** hold.

**Change.**

1. **Probe the real request.** Give `run_recall_check` the request exactly as the model receives it: `_prev_req` plus the ledger note and marker when they are injected. Add `Session.last_request_seen(ledger_text, marker_note)`, or pass the final message list in from `_prepare`.
2. **Choose targets by mode.** Add `target: Literal["auto", "ledger", "history"] = "auto"` to `RecallPolicy`.
    - `"auto"` resolves to `"history"` when the ledger is injected, and to `"ledger"` when it isn't.
    - `"ledger"` asks about active ledger entries, using R1's questions.
    - `"history"` asks about facts from the conversation that the ledger does **not** cover.
3. **Build a stable pool of history facts.** Add `FactPool` (in `probes/recall.py`, or a new `probes/pool.py`):
    - The first time a user or assistant message is seen, run `cloze_facts` on it and add the facts to the pool, with the message id and turn.
    - Skip any token already in an active ledger entry: pass the ledger's tokens as `skip`.
    - Keep the pool between turns, so the same fact is asked again over time. That gives a recall curve per fact, which is what early warning needs.
    - Facts whose source message was compacted away stay in the pool. Whether the summary kept them is exactly the question.
4. **Update `docs/recall.md`.** Rewrite "What a probe sees" to match the new behaviour.

**Tests:**

- With `inject_ledger=True`, the context passed to the probe caller includes the ledger note, and no sampled fact token appears in an active ledger entry.
- With `inject_ledger=False`, ledger entries are probed.
- A fact whose source message was removed by a compaction is still in the pool and still asked.
- The pool doesn't grow with duplicate tokens.

**Done when:** all of the above pass, and the Gate 3 harness (R4) can run with both `target="ledger"` and `target="history"`.

### R3. Detect stale beliefs in probe answers (P1, S)

**Why.** The plan's fourth failure mode is a decision that was overridden but keeps steering the agent. Today a probe only marks an entry recalled or missed. When an answer gives a *superseded* value (`staging_orders` after it was replaced by `archive_orders`), that is a different and more serious signal.

**Change.** When grading a cued question for an entry that has superseded predecessors (same `key`, or `superseded_by` pointing at it), also check the answer for the old entry's tokens. Add `stale: list[str]` to `RecallResult`, and count a stale answer as a miss. Add `stale` to the stored `recall` event and to the HTML report.

**Tests:** a fake model that answers with the superseded value is graded missed, and the entry appears in `stale`.

**Done when:** the superseded-decision suite reports a stale count per run.

### R4. Fix the Gate 3 harness (P0, M)

**Where:** `evals/gate3.py`.

**Why.** Three problems make the first Gate 3 run hard to read:

- **Compaction doesn't compound.** The docstring says each turn re-compacts the original prefix instead of the previous summary, and that this should change "if Gate 3 is close".
- **There is no clean control.** Every task failed (9 of 9), so false alarms couldn't be measured.
- **There was only one seed.**

**Change.**

1. **Compound the compaction.** Summarize the previous summary plus the new messages, as a real agent does.
2. **Add a control arm.** Run the same scenario with no compaction. Count a false alarm as an alert in a run where the task never fails.
3. **Calibrate per run.** Set `pass_at` from the probes taken before any compaction (`calibrate`), and report the calibrated value next to the default 0.5.
4. **Add arguments.** `--style cued|list`, `--target ledger|history`, and `--seeds 3` as the default.
5. **Report more.** For each fact, record the turn of its first miss alongside the turn the task first failed: the lead time, per fact.
6. **Report overhead.** Probe tokens as a percentage of agent tokens, using E1.

**Done when:** one command produces, per model, the lead time, misses, false alarms against the control arm, calibrated `pass_at` and overhead percentage, for both styles and both targets.

---

## 5. M2: Evaluation on hosted models

Every result so far is from qwen2.5 1.5B, 3B and 7B running locally. At that size it's hard to tell context rot apart from a model that can't do the task at all: 3 of 9 Gate 3 tasks failed before any compaction. The plan's Phase 0 asked for one Anthropic and one OpenAI model. `evals/models.py` already has clients for both, defaulting to `gpt-4.1-mini` and `claude-haiku-5-5`.

### E1. Token and cost tracking (P0, S)

**Where:** `evals/models.py` (`Client.complete`, which returns only text), and the side callers used by `gate2.py` and `gate3.py`.

**Change.**

- Record input and output tokens from each response's usage field, for both providers, in a per-run counter split by purpose: agent call, summarizer, audit, recall probe.
- Add `--estimate` to `run.py`, `gate2.py` and `gate3.py`. It does a dry run with the mock model, counts the calls each run would make, and prints the count and an approximate token total before anything is spent.
- Write totals into each results JSON.

**Done when:** every results file has token totals by purpose, and `--estimate` prints a call count before any paid run.

### E2. Real baselines (P0, M)

**Why.** The benchmark's strategies are `none`, `truncate` and a naive `summarize` (`evals/core.py`). The plan's baselines also include LangChain's `SummarizationMiddleware`, Anthropic server-side compaction and ContextForge. LangChain's summarizer appears only in the smoke test `evals/e2e_langchain.py`.

**Change.** Add these strategies to `STRATEGIES`:

| Strategy | How |
| --- | --- |
| `lc-summarize` | Run the history through LangChain's `SummarizationMiddleware`, reusing the setup in `e2e_langchain.py` |
| `anthropic-compaction` | Anthropic server-side compaction, threshold mode. Per Anthropic's docs it uses the beta header `compact-2026-09-04`. Keep the returned compaction block exactly as returned, as `RotguardAnthropic` already does. Anthropic models only. |
| `contextforge` | Optional. Install ContextForge per its README, run its compile step on the history, and skip with a clear message if it isn't installed. |

Each strategy also runs with `+ledger`, as the current ones do.

**Done when:** `evals.run --strategies none,summarize,lc-summarize,anthropic-compaction` runs end to end on `claude-haiku-5-5`, and the same minus `anthropic-compaction` on `gpt-4.1-mini`.

### E3. Gate 1: does the ledger help? (P0, M)

**Why.** The plan's Gate 1 is "the ledger beats baselines on goal retention". The harness has a `+ledger` variant, but no result is recorded in `CHANGELOG.md` or `docs/evaluation.md`.

**Change.** On both hosted models, with 3 seeds, run suites `planted_facts`, `superseded` and `goal_drift` with each E2 strategy, with and without `+ledger`. Report:

- **Goal retention:** share of forced-compaction runs where the agent keeps its goal.
- **Stale-decision rate:** share of runs that act on a superseded decision (E7).
- **Planted-fact recall.**

Present each with and without the ledger, per strategy and model, with run counts. Then write the result in `docs/evaluation.md` and `CHANGELOG.md` in the same style as Gates 2 and 3, including the null result if the ledger doesn't help.

**Done when:** Gate 1 has a recorded verdict on two hosted models.

### E4. Re-run Gates 2 and 3 on hosted models (P0, M)

**Change.**

- **Gate 2:** `evals.gate2 --suites holdout,holdout2 --seeds 3 --cloze 2` on both models.
- **Gate 3:** the fixed harness (R4) with `--style cued` and both targets, on both models.

Keep the 3B runs as a small-model reference. State plainly that `holdout` and `holdout2` informed earlier changes and are no longer clean held-out sets. Write a new held-out suite (`holdout3`, about 9 new templates) and run it only after the configuration is frozen.

**Done when:** `docs/evaluation.md` has Gate 2 and Gate 3 tables for both hosted models, with a verdict on each gate.

### E5. Suite 4: replicate part of Chroma's results (P1, M)

**Why.** The plan's suite 4 confirms the harness sees the same effect published research does.

**Change.** Take one experiment from Chroma's open-source toolkit (`chroma-core/context-rot`; the NIAH extension is the closest fit) and run a small grid of input lengths on the two hosted models. Add `evals/suites/replication.py`, or a script under `evals/replication/` that drives the toolkit.

**Done when:** a chart or table shows accuracy against input length for both models, next to Chroma's direction of effect.

### E6. Suite 5: real agent traces (P1, L)

**Why.** All scenarios are synthetic. The plan asks for long runs of real agents, such as research and coding agents.

**Change.**

1. Build two small real agents with `RotguardMiddleware` writing to SQLite: a research agent with web search, and a coding agent that works on a sample repo.
2. Run each for 3 to 5 long sessions with `SummarizationMiddleware` on.
3. By hand, label the constraints that mattered and whether the agent broke one.
4. Replay with `rotguard replay --audit every --recall-every 2` and check whether audits and probes flagged the lost constraints.

**Done when:** at least 6 labelled real traces exist under `evals/traces/`, with a short write-up of what the audit and probes caught and missed.

### E7. Missing metrics (P1, S)

| Metric | How to measure |
| --- | --- |
| Stale-decision rate | In the `superseded` suite, the share of runs whose final answer uses the superseded value. Reuse R3's token check. |
| Overhead | Tokens spent on audits and probes as a percentage of agent tokens, from E1. Cost percentage follows from the provider's per-token prices on the day of the run, which should be recorded. |
| Broken histories | Count of requests where a tool result lacks its call or the reverse. Add `rotguard.testing.assert_history_integrity(messages)` and call it on every request in the harness. |

**Done when:** all three appear in every Gate run's summary.

### E8. Nightly CI eval (P1, S)

**Change.** Add `.github/workflows/nightly-eval.yml`:

- Runs on a `schedule` (cron) and `workflow_dispatch`.
- Uses the cheapest hosted model, with the key stored as a repository secret.
- Runs a fixed small subset: one scenario per suite, 1 seed.
- Uploads the results JSON as an artifact.
- Fails if goal retention or audit validity drops below the last recorded value by more than a set margin.

**Done when:** the workflow has run green once and stored an artifact.

### E9. Fill in the open targets (P1, S)

**Why.** The plan's evaluation table left "false alarms" and the "audit validity bar" open until baselines existed.

**Change.** After E3 and E4, set both numbers in `docs/evaluation.md` and in `docs/plan.md` (H3), with one line saying which runs they came from.

---

## 6. M3: Thin policies (F5)

Feature F5 is missing entirely: there is no `src/rotguard/policies/` module. Only the MCP `recall` tool looks up compacted items. The audit's own suggestion text (`probes/audit.py`, `suggest`) tells users to "offload large tool outputs", which rotguard can't yet do. Plan section 7.2 also found that LangChain's built-ins can't express "clear tool outputs first, summarize only if still over budget". F5d is rotguard's chance to fill that gap.

**Rules for every policy:** off by default; recorded as compactions with provenance, so the audit (F3) checks them automatically; tool calls and their results always stay paired; fail open.

### A4. Token counting helper (P1, S)

**Why.** Budgets (F5a, F5d) and overhead (E7) need token counts. Nothing in `src/rotguard` counts tokens today.

**Change.** Add `src/rotguard/tokens.py` with `count(messages, model=None) -> int`:

- Use the provider's token-counting API when one is configured and reachable.
- Otherwise use a characters-divided-by-4 estimate, and say so in the docstring and in any event that relies on it.
- Cache counts by message id.

**Tests:** the fallback is deterministic, and a provider error falls back instead of raising.

### F5a. Tool-output offload (P1, M)

**Where:** new `src/rotguard/policies/offload.py`; `RotguardMiddleware.wrap_tool_call` in `adapters/langchain.py`.

**Change.**

1. Add `OffloadPolicy(max_tokens=2000, preview_lines=10)`.
2. In `wrap_tool_call`, if a result exceeds `max_tokens`, store the full text in the store as an item with provenance kind `offloaded`. Return a `ToolMessage` with the **same `tool_call_id`** whose content is a stub: the first `preview_lines` lines, the item id, and the instruction "call `recall_output(id)` for the full output".
3. Add a `recall_output(id, start_line=0, max_lines=200)` tool to `self.tools`, so large outputs can be read in pages.
4. This edits stored history (the tool result is stored as the stub), so it is opt-in, like handoff.

**Tests:** large output becomes a stub with the same `tool_call_id`; `recall_output` returns the stored text in pages; small outputs pass through untouched; an offload is recorded with provenance.

### F5b. Dedup repeated tool outputs (P1, S)

**Change.** In `_prepare`, per call only (the request override, never state): when two tool results have identical content, measured by hash, replace the later copies with `[rotguard: same output as <id>]`. Never touch user or assistant messages.

**Tests:** duplicates collapse in the request, stored history is unchanged, and pairing is kept.

### F5c. Pinning (P1, M)

**Change.**

1. Add `rotguard.pin(message_id)`, and `pin=True` on `ledger_record` for anything that must survive verbatim.
2. Pinned items are exempt from F5a, F5b and F5d, and from handoff trimming (`handoff.py`).
3. If a pinned message is missing from a request because another tool compacted it away, re-attach its text after the ledger note for that call only, labelled as pinned.

**Tests:** pinned items survive handoff and offload, and a pinned message removed by a summarizer reappears in the next request.

### F5d. Staged cleanup (P1, L)

**Where:** new `src/rotguard/policies/staged.py`; a hook in `RotguardMiddleware.before_model`.

**Change.** Add `StagedCleanup(budget_tokens, clear_keep=3, summarizer=None, keep_messages=6)`. In `before_model`:

1. Count tokens (A4). If under budget, stop.
2. **Stage 1, clear.** Offload every tool result except the newest `clear_keep`, using F5a's stubs so nothing is lost for good. Recount. If now under budget, stop.
3. **Stage 2, summarize, only if still over budget.** Summarize everything except system messages, pinned items and the last `keep_messages` messages, using the given summarizer: a LangChain chat model or a callable. Create a summary message with provenance kind `summary` and the ids it covers, so `blame` works through it.
4. Return the new message list as a state update. Use `RemoveMessage(id=REMOVE_ALL_MESSAGES)`, as handoff already does.
5. Record each stage as a compaction event, so F3 audits it.
6. If `SummarizationMiddleware` is also in the middleware list, or Anthropic compaction is in use, log a warning and disable stage 2, so two summarizers don't fight.

**Tests:**

- Under budget: nothing happens.
- Clearing is enough: no summarizer call is made.
- Still over budget: exactly one summarizer call, and pinned items are kept.
- Pairing is kept, and both stages appear as compaction events.

**Done when:** the eval harness has an `rg-staged` strategy, and E3 includes it.

### F4a. `action="compact"` on recall alert (P2, S)

**Change.** Add `"compact"` to `RecallPolicy.action`. On alert, run F5d once on the next `before_model`, even if under budget, using the configured `StagedCleanup`. Without one, fall back to `"log"` and warn.

---

## 7. M4: Remaining features and architecture

### F1x. Optional ledger extractor (P1, M)

**Why.** Plan F1 has two ways to add entries: the agent's `ledger_record` tool (built) and an optional extractor that reads each reply (missing). An agent that forgets to call the tool loses its constraints silently.

**Change.**

1. Add `src/rotguard/ledger/extract.py` with `LedgerExtractor(caller, every=1, kinds=("goal", "decision", "constraint"))`.
2. After the model call, it makes one side call over the last user message and the reply. The prompt is delimited like the audit's extraction prompt, so instructions inside the transcript can't take it over. It returns a JSON array of `{kind, text, key}`.
3. Accept an entry only if at least one of its distinctive tokens (`fact_tokens`) appears in the source messages, which keeps out invented entries.
4. Add `origin: Literal["tool", "manual", "extractor"] = "manual"` to `Entry` and set it, then regenerate `schemas/ledger_entry.schema.json`.
5. Add `confirm: Callable[[Entry], bool] | None`: when set, extracted entries are recorded only if it returns True.
6. Expose it as `RotguardMiddleware(extract=LedgerExtractor(...))`. Off by default.

**Tests:** a fake extractor reply creates entries with `origin="extractor"`; invented entries are rejected; an injected instruction in the transcript doesn't change the output; `confirm` blocks entries.

### T1. Live adapter tests (P1, S)

**Why.** The README says the Anthropic wrapper is tested "against the documented block shapes and the SDK's types, not the live API", and the OpenAI Agents wrapper "against a fake session, not the real SDK".

**Change.** Add `tests/live/` with a `live` pytest marker, skipped when no key is set:

- **Anthropic:** one on-demand compaction round trip through `RotguardAnthropic`. Check that the compaction block is preserved exactly and that an audit runs.
- **OpenAI Agents SDK:** wrap a real SDK session with `RotguardSession`, run a short agent, and check that compactions are seen at `get_items`.

Run them in the nightly workflow (E8), not on every push.

### A1. OpenTelemetry events (P1, M)

**Why.** The plan chose OpenTelemetry so rotguard's events appear in LangSmith and other tracing tools. Today rotguard logs through LogQuill only.

**Change.**

1. Add an `otel` extra (`opentelemetry-api`).
2. Add `src/rotguard/telemetry.py`. When the extra is installed and a tracer provider is set, it emits a span event for each stored event: compaction, audit, recall, signals, marker_miss, handoff. Attributes include the session id, turn, loss or score, and counts. Never include message text unless `include_text=True`.
3. Call it from `Session` wherever `store.add_event` is called. Keep LogQuill for logs.

**Tests:** with an in-memory span exporter, each event kind produces one span event with the expected attributes; with the extra not installed, nothing breaks.

### A2. Store protocol (P2, S)

**Change.** Add `rotguard.store.Store`, a `typing.Protocol` with the public methods `SQLiteStore` already has (`add_items`, `mark_removed`, `items`, `save_snapshot`, `snapshot_at`, `upsert_entry`, `entries`, `add_event`, `events`, `sessions`, `close`). Type `Session`, the adapters, `report.query` and the CLI against `Store` instead of `SQLiteStore`. No new backend is needed yet. This only makes one possible later.

### A3. JSON schema for reports and events (P2, S)

**Change.** Generate `schemas/report.schema.json`, covering the MCP `report` tool's output and the stored `audit` and `recall` event payloads (`AuditReport.to_dict`, `RecallResult.to_dict`). Do it the same way `message.schema.json` is generated, and add a test like the existing ones in `tests/test_schemas.py`.

---

## 8. M5: Launch

Launch only after M2. Rules from the CHANGELOG still hold: the early-warning claim stays out of launch material unless Gate 3 passes, and the single `loss` number isn't presented as predictive.

### L1. Benchmark report "What does compaction cost?" (P1, L)

**Content.**

- What each baseline (E2) lost on the same scenarios, measured by goal retention, planted-fact recall and stale-decision rate.
- What the ledger changed (E3).
- What the audit caught (E4).
- Overhead (E7).
- Every number with its run count, model and date. Methods and scripts are reproducible from the repo.
- A limitations section in the README's current tone.

**Where:** `docs/benchmark.md`, plus a blog-length version for posting.

### L2. LangChain middleware-ordering guide (P1, S)

**Change.** Add `docs/langchain-ordering.md` to answer the open forum question:

- Summarization runs in `before_model` and rewrites stored history.
- Context editing runs in `wrap_model_call` on a copy, and repeats every call.
- So summarization always runs first, whatever the list order.
- List rotguard last.
- Use F5d when you want "clear first, then summarize".

Include the pinned ordering test from Phase 1 as evidence and a small runnable example.

### L3. Sample HTML report (P1, S)

**Change.** Generate `rotguard report` from one real hosted run (E4 or E6). Commit it as `docs/assets/sample-report.html` and link it from the README, plus a screenshot.

### L4. Discoverability (P1, S)

- **README header:** add the tagline *"Your agent's context is rotting. rotguard tells you how much."*
- **PyPI:** keywords and classifiers (H1).
- **GitHub topics:** `context-rot`, `langchain`, `llm-agents`, `context-engineering`, `llm`, `agents`.
- **Repo description:** the README's first line.

### L5. Launch posts (P1, M)

**Channels from the plan:** the LangChain forum (answer the ordering thread with L2), Hacker News, r/LangChain and r/LocalLLaMA, X and LinkedIn.

Lead with L1's findings, not the feature list. Say plainly what isn't proven.

---

## 9. Decisions to make

| ID | Decision | Recommendation | When |
| --- | --- | --- | --- |
| D1 | License | See section 3. Choose deliberately before publishing to PyPI. | M0 |
| D2 | AnchorLoop integration | After v1.0. `RotguardGraph` already saves the ledger through LangGraph checkpoints, which is the groundwork. | After M5 |
| D3 | Probe target default | `target="auto"` (R2): history facts when the ledger is injected, ledger entries otherwise | M1 |
| D4 | Hosted models for the gates | `claude-haiku-5-5` and `gpt-4.1-mini` (the defaults in `evals/models.py`), plus one larger model for a spot check if the budget allows | M2 |
| D5 | Keep `style="list"` probes | Keep as an option for comparison; default to `"cued"` | M1 |

---

## 10. Updated gates

| Gate | Criterion | Status at review | After this plan |
| --- | --- | --- | --- |
| 0 | Rot reproduces on our suites; open targets set | Partly: suites built on local models; targets not set | E3, E5, E9 |
| 1 | Ledger beats baselines on goal retention; no broken histories | **Not recorded** | E3, E7 |
| 2 | Audit loss predicts later failures; if not, the audit becomes a debugging tool | Fallback taken: the per-fact verdict works as a flag | E4 re-tests on hosted models and a fresh held-out suite |
| 3 | Alerts at least 1 turn before failure, at no more than 10% extra cost | **Not met**: probe failed on clean contexts | R1–R4, then E4 |
| Release | v1.0 on PyPI, benchmark report public | 0.0.0, not published | H1 (alpha now), L1–L5 |

If Gate 3 still fails after R1–R4 on hosted models, keep recall checks as an experimental logging feature and launch on the ledger, provenance and audit alone. Those don't depend on it.

---

## 11. Rules that still apply

These come from the plan and the current README, and every item above keeps them:

- **Fail open.** A rotguard error never blocks the agent's call.
- **Non-destructive by default.** Only opt-in features rewrite stored history: handoff, F5a and F5d.
- **Nothing is deleted.** Superseded ledger entries and offloaded outputs stay in the store.
- **Side calls never enter history.**
- **No framework in the core.** New policies take neutral messages; framework code stays in `adapters/`.
- **Tool calls and results stay paired** in every request.
- **Claims match evidence.** Each README claim cites a measured result with its run count, or says it is unproven.

---

## 12. Sources

Repository files read on 20 Oct 2026 at commit `88f78ca`:

- [README.md](https://github.com/nikhilvdev/rotguard/blob/main/README.md), [CHANGELOG.md](https://github.com/nikhilvdev/rotguard/blob/main/CHANGELOG.md), [LICENSE](https://github.com/nikhilvdev/rotguard/blob/main/LICENSE), [pyproject.toml](https://github.com/nikhilvdev/rotguard/blob/main/pyproject.toml), [.github/workflows/ci.yml](https://github.com/nikhilvdev/rotguard/blob/main/.github/workflows/ci.yml)
- [src/rotguard/probes/recall.py](https://github.com/nikhilvdev/rotguard/blob/main/src/rotguard/probes/recall.py), [src/rotguard/probes/quiz.py](https://github.com/nikhilvdev/rotguard/blob/main/src/rotguard/probes/quiz.py), [src/rotguard/probes/signals.py](https://github.com/nikhilvdev/rotguard/blob/main/src/rotguard/probes/signals.py), [src/rotguard/session.py](https://github.com/nikhilvdev/rotguard/blob/main/src/rotguard/session.py), [src/rotguard/adapters/langchain.py](https://github.com/nikhilvdev/rotguard/blob/main/src/rotguard/adapters/langchain.py), [src/rotguard/ledger/ledger.py](https://github.com/nikhilvdev/rotguard/blob/main/src/rotguard/ledger/ledger.py), [src/rotguard/store/sqlite.py](https://github.com/nikhilvdev/rotguard/blob/main/src/rotguard/store/sqlite.py)
- [docs/evaluation.md](https://github.com/nikhilvdev/rotguard/blob/main/docs/evaluation.md), [docs/recall.md](https://github.com/nikhilvdev/rotguard/blob/main/docs/recall.md)
- [evals/gate3.py](https://github.com/nikhilvdev/rotguard/blob/main/evals/gate3.py), [evals/core.py](https://github.com/nikhilvdev/rotguard/blob/main/evals/core.py), [evals/models.py](https://github.com/nikhilvdev/rotguard/blob/main/evals/models.py)

External references (opened on 11 Oct 2026 for the plan):

- [Compaction](https://platform.claude.com/docs/en/build-with-claude/compaction), Claude API docs (beta header `compact-2026-09-04`)
- [Custom middleware](https://docs.langchain.com/oss/python/langchain/middleware/custom), LangChain docs (`wrap_tool_call` returns a `ToolMessage` or `Command`)
- [LangChain forum thread on middleware ordering](https://forum.langchain.com/t/how-do-contexteditingmiddleware-and-summarizationmiddleware-interact-when-used-together-combining-contexteditingmiddleware-summarizationmiddleware-execution-order-and-behavior-when-both-trigger/3463)
- [chroma-core/context-rot](https://github.com/chroma-core/context-rot), replication toolkit
- [ContextForge](https://github.com/eatakishiyev/context-forge)

Checks run for this review: `uv run pytest` (99 passed), `uv run ruff check .` (clean), `uv run ruff format --check .` (1 file), `uv run mypy` (clean), and PyPI and npm registry lookups for `rotguard` on 20 Oct 2026 (both unclaimed).