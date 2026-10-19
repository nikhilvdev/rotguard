# Evaluation

What has been measured, on what, and what it does and does not show. Everything here is small, on
local models and on a synthetic benchmark. **Nothing has been run on a hosted model yet.** Treat all
of it as a first read, and re-run on your own model before depending on it. The raw numbers and
caveats are in `CHANGELOG.md`.

## Setup

- Harness: `evals/` in the repo (`python -m evals.gate2`, `python -m evals.gate3`).
- Models: qwen2.5 3B (most results), qwen2.5-coder 7B and qwen2.5 1.5B, run locally with Ollama.
- Histories: 30 turns with a fact planted early, compacted by a naive LLM summarizer; then a probe
  asks about the fact. A probe **fails** if the answer is wrong.
- Samples are small: usually one run per cell, and about nine scenario templates, so the effective
  sample is nine and the p-values are optimistic.

## Compaction audit (Gate 2)

| Question | Result |
| --- | --- |
| Does the aggregate `loss` predict a later probe failure? | **No.** AUC 0.61 and 0.44 (3B and 7B, 50 scenarios) |
| After dropping tool-origin facts? | Suggestive, not significant: AUC 0.71 and 0.75 (p about 0.09 and 0.10). Exploratory |
| Does the audit's verdict on the one probed fact predict failure? | **As a flag, yes.** "Kept": 0 of 13 failed. "Lost": 10 of 23 failed (pooled Fisher p = 0.0045, unadjusted) |

After fixes to answer grading, a model-free fact fallback and a leakage bug, a second held-out set
of 9 new templates (3B) gave AUC 0.90 over 18 scored scenarios: "kept" 0 failures of 8, "lost" 8 of
10. A later change (cap tool-origin facts) raised the share of scenarios where the audit checked
the probed fact from 33 to 39 of 54 and AUC to 0.96 over 35 scored. Those suites informed that
change, so they are no longer clean held-out sets.

**What this supports:** the list of lost facts and the per-fact verdict are useful debugging output.
**What it does not:** the audit still never checked the probed fact in 15 of 54 scenarios, so a
clean report is not proof that nothing was lost. The 1.5B model could run the audit but failed 24
of 27 probes, so its result (AUC 0.63) says little.

## Recall checks (Gate 3)

The question: do recall probes fail at least one turn **before** the task does, without too many
false alarms? **The first run does not answer it.**

qwen2.5 3B, `holdout` suite, 30 turns, 1 seed, 9 scenarios, `pass_at` 0.5, alert after 2 failed
probes:

| Measure | Result |
| --- | --- |
| Task failed | 9 of 9 (3 of them before any compaction) |
| Alert fired at turn 1 | 8 of 9 |
| Recall score before any compaction | 0.11 mean |
| Missed / false alarms | 0 / 0 (every task failed) |
| Probe cost | 1 side call per turn (probing every turn) |

The 3B model fails the recall probe even when the constraint is still in its context, so the alert
is effectively always on and the measured "lead" is not evidence of early warning. Calibration only
lowers `pass_at` to its floor of 0.3, which does not fix a model that cannot answer the probe.

**Open:** why the clean-context score is so low (probe wording, grading or the model), and a run on
a hosted Anthropic and OpenAI model. Until then, use recall checks to log and experiment, not to
trigger actions you cannot undo. In particular, be careful with `action="handoff"`, which rewrites
history.

## Run it yourself

```bash
uv sync --extra langchain --extra eval
uv run python -m evals.gate2 --model ollama:qwen2.5:3b --turns 30 --seeds 3 --suites holdout,holdout2 --cloze 2
uv run python -m evals.gate3 --model ollama:qwen2.5:3b --turns 30 --seeds 3 --suites holdout
```

Hosted runs need `ROTGUARD_OPENAI_API_KEY` or an Anthropic key and cost money.
