# Recall checks and handoff

Recall checks ask the model, every few turns, to restate some of your active ledger entries. A
miss means the model may no longer be holding that constraint. They are **off by default**.

!!! warning "Not yet shown to warn early"
    The mechanism is tested with fake models. On a real model (qwen2.5 3B) the first benchmark did
    **not** show that probe failures arrive before task failures. Read
    [Evaluation](evaluation.md) before relying on an alert.

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
| 3. Recall probe | A side call over the current context asking for up to `sample` ledger entries, graded per entry | 1 side call |

A tier 1 or 2 flag makes the next probe run **now** instead of on schedule (the probe budget still
applies). The signals are lexical, not embedding based, and need a few messages before they flag.

The marker (`RecallPolicy(marker="RG-OK")`) works on LangChain and LangGraph only, because it is
added to the injected ledger note. For LangGraph call `rg.observe_reply(text)` with each reply. The
Anthropic and OpenAI wrappers inject nothing, so they have no marker check.

## What a probe sees

The probe runs over the context **without** the injected ledger. The point is to find out whether
the underlying context still holds the constraint; with the ledger in front, the model could always
answer. Probes are side calls: they never enter your history.

## Alerts

One miss is noise. An alert fires when `alert_after` probes **in a row** score below `pass_at`, and
it fires **once** when the streak reaches that length, not on every later failure. Each probe is
stored as a `recall` event with its score, the entries missed and a rolling score.

## Policy

| Field | Default | Meaning |
| --- | --- | --- |
| `mode` | `"off"` | `off` or `every` |
| `every` | `5` | turns between scheduled probes |
| `sample` | `3` | ledger entries asked per probe |
| `max_probes` | `20` | per session |
| `pass_at` | `0.5` | share of sampled entries that must come back for a pass |
| `alert_after` | `2` | consecutive failed probes before an alert |
| `window` | `5` | probes in the rolling score |
| `signals` | `True` | tier 1 flags force a probe |
| `marker` | `None` | tier 2 marker text |
| `action` | `"log"` | on alert: `log` only, or `handoff` |
| `handoff_keep` | `4` | messages kept verbatim by a handoff |
| `background` | `True` | probes run on a worker thread |

## Per-model calibration

A model that restates only 70% of entries even with a fresh context would fail every probe at a high
`pass_at`. `rotguard.probes.calibrate(clean_scores)` returns a `pass_at` just under the model's
performance on contexts known to be rot-free (the lower quartile, clamped to 0.3 to 1.0). Collect a
handful of clean probe scores first. It cannot rescue a model that cannot answer the probe at all.

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
- After a handoff the ledger can appear twice: in the note, and from per-call injection.

You can also build one directly:

```python
from rotguard.handoff import handoff

fresh = handoff(messages, rotguard.ledger, keep=4)
```
