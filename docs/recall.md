# Recall checks and handoff

Recall checks ask the model, every few turns, pointed questions about facts it was given. A miss
means the model may no longer be holding that fact. They are **off by default**.

!!! warning "Not yet shown to warn early"
    The probe works on clean contexts (recall 0.82 on qwen2.5 3B after the redesign, up from 0.03)
    and no longer raises false alarms. But in the same benchmark its alert arrived at or after the
    task failure in 7 of 8 failing runs. Read [Evaluation](evaluation.md) before relying on an alert.

```python
from rotguard.probes import RecallPolicy

rotguard = RotguardMiddleware(
    "rotguard.db",
    recall=RecallPolicy(mode="every", every=5),
    on_alert=lambda result: print("recall failing", result.missed),
)
```

Works on the LangChain middleware, `RotguardGraph`, `RotguardAnthropic` and `RotguardSession`
(OpenAI Agents).

## Three tiers, free to paid

| Tier | What | Cost |
| --- | --- | --- |
| 1. Passive signals | Goal words missing from recent messages, repeated assistant replies, a high share of tool errors | Free |
| 2. Marker | A standing instruction to begin every reply with a marker; a reply without it means the instruction was dropped | Free |
| 3. Recall probe | One side call over the context the model will see, with up to `sample` fill-in-the-blank questions, graded per question | 1 side call |

A tier 1 or 2 flag makes the next probe run **now** instead of on schedule (the probe budget still
applies). The signals are lexical, not embedding based, and need a few messages before they flag.

The marker (`RecallPolicy(marker="RG-OK")`) works on LangChain and LangGraph only, because it is
added to the injected ledger note. For LangGraph call `rg.observe_reply(text)` with each reply. The
Anthropic and OpenAI wrappers inject nothing, so they have no marker check.

## What a probe sees and asks

The probe runs over the request **exactly as the model receives it**, including the ledger note and
the marker when they are injected. So that it still measures something, what it asks about depends on
`target`:

- `"ledger"`: the active ledger entries. Used when the ledger is not shown to the model.
- `"history"`: facts from the conversation that the ledger does **not** cover. Used when the ledger
  is shown on every call, because the model can always answer ledger questions from it.
- `"auto"` (default) picks between them.

**Cued questions** (`style="cued"`, the default). Each fact becomes a fill-in-the-blank question
("Complete the constraint currently in effect: Only write to ____."). Every occurrence of the answer
is blanked, so a question cannot give away the answer, and a question that contains another's answer
is dropped. An entry with no identifier or number gets its longest content word blanked; one open
goal gets "What is the current goal of this task?". `style="list"` is the old single "list everything
in effect" question; it fails far more often on clean contexts and is kept only for comparison.

**History facts** come from a pool collected as messages arrive, one or two per message from user and
assistant text (never tool output or summaries). A fact stays in the pool after its source message is
compacted away, because whether the summary kept it is exactly what the probe asks.

**Stale answers.** When an entry replaced an older one, an answer that gives the *old* value instead
of the new one (or names it first) is `stale` and counts as a miss. Mentioning the old value after the
right one ("archive_orders, not staging_orders") does not.

Probes are side calls: they never enter your history.

## Alerts

One miss is noise. An alert fires when `alert_after` probes **in a row** score below `pass_at`, and
it fires **once** when the streak reaches that length, not on every later failure. Each probe is
stored as a `recall` event with its score, the entries missed and a rolling score.

## Policy

| Field | Default | Meaning |
| --- | --- | --- |
| `mode` | `"off"` | `off` or `every` |
| `every` | `5` | turns between scheduled probes |
| `sample` | `3` | facts asked per probe |
| `style` | `"cued"` | `cued` (one question per fact) or `list` (one "list everything") |
| `target` | `"auto"` | `ledger`, `history` or `auto` |
| `max_probes` | `20` | per session |
| `pass_at` | `0.5` | share of sampled entries that must come back for a pass |
| `alert_after` | `2` | consecutive failed probes before an alert |
| `window` | `5` | probes in the rolling score |
| `signals` | `True` | tier 1 flags force a probe |
| `marker` | `None` | tier 2 marker text |
| `action` | `"log"` | on alert: `log`, `handoff` or `compact` |
| `handoff_keep` | `4` | messages kept verbatim by a handoff |
| `background` | `True` | probes run on a worker thread |

## Per-model calibration

A model that restates only 70% of entries even with a fresh context would fail every probe at a high
`pass_at`. `rotguard.probes.calibrate(clean_scores)` returns a `pass_at` just under the model's
performance on contexts known to be rot-free (the lower quartile, clamped to 0.3 to 1.0). Collect a
handful of clean probe scores first. It cannot rescue a model that cannot answer the probe at all.

## Compact on alert

`RecallPolicy(action="compact")` runs [staged cleanup](policies.md#staged-cleanup) once on the next
model call after an alert, even if the context is under budget. It needs a `StagedCleanup` on the
middleware; without one it only logs, and says so once.

## Fresh-context handoff

With `RecallPolicy(action="handoff")`, an alert restarts the context: the system messages, a ledger
note and the last `handoff_keep` messages. The tail is widened so it starts on a user message and
every tool result keeps its tool call.

- `RotguardMiddleware` and `RotguardGraph` replace the history on the next model call.
- It **rewrites stored history**. It is the only place rotguard does, and it is opt-in. The change is
  recorded as a compaction and as a `handoff` event.
- It waits while it would not shrink the context, and a passing probe cancels it.
- `RotguardAnthropic` and `RotguardSession` do not own the history. Call
  `session.take_handoff(context)` and replace the history yourself.
- Pinned messages and the tool call or result each is paired with are kept ahead of the tail.
- After a handoff the ledger can appear twice: in the note, and from per-call injection.

You can also build one directly:

```python
from rotguard.handoff import handoff

fresh = handoff(messages, rotguard.ledger, keep=4)
```
