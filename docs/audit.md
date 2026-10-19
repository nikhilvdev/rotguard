# Audit a compaction

After a summarization or trimming step, the audit asks: **what did the model lose?**

It is **off unless you ask for it**, because each audit costs about three extra model calls: one to
extract facts from the span that was removed, then one quiz on the old context and one on the new.

```python
from rotguard.probes import AuditPolicy

rotguard = RotguardMiddleware("rotguard.db", audit=AuditPolicy(mode="sampled", sample_rate=0.5))
...
rotguard.store.events(rotguard.session_id, "audit")  # loss, the facts lost, and what to change
```

## How it works

1. Facts come from the removed text (each one is checked to appear in it word for word) and from your
   ledger entries.
2. The same quiz is asked of the model twice: once with the context before the compaction, once with
   the context after.
3. A fact counts as **lost** if only the old context answered it.

Side calls go through the same model and are **never** written to your history or to the store's
items. When the ledger is injected into requests, it is put in front of the quiz for both contexts
and ledger entries are not checked (they would be guaranteed and dilute the loss).

## The report

| Field | Meaning |
| --- | --- |
| `loss` | lost / answerable-before, over facts that did **not** come from tool output. `None` if the old context answered none of them |
| `loss_all` | the same ratio over every fact, tool output included |
| `lost` | every lost fact, with its origin message and kind |
| `checked` | every fact, with its before and after result |
| `suggestions` | what to change (record it in the ledger, raise `keep`, change the summary prompt) |
| `side_calls` | how many calls the audit made |

Tool output is mostly ids and listings that no summary keeps, which is why `loss` leaves it out; the
`lost` list still shows those facts.

## Policy

| Field | Default | Meaning |
| --- | --- | --- |
| `mode` | `"sampled"` | `off`, `sampled`, `every`, or `ci` (runs synchronously) |
| `sample_rate` | `0.5` | share of compactions audited in `sampled` mode |
| `max_facts` | `5` | facts per audit |
| `cloze` | `2` | facts built straight from the text when extraction finds fewer than `max_facts`; needs no model call |
| `max_audits` | `20` | per session |
| `ci_max_loss` | `0.3` | used by `assert_audit_loss` |
| `background` | `True` | audits run on a worker thread; `ci` mode is always synchronous |

## Fail a test when too much is lost

```python
from rotguard.testing import assert_audit_loss

assert_audit_loss(rotguard.store, rotguard.session_id, max_loss=0.3)
```

## Limits

- A report compares one compaction with the context just before it. A summary of an already-lossy
  summary reports little loss, even though the cumulative loss is high.
- Answers are matched as text, not judged for meaning, so a negated constraint can pass.
- The questions are only as good as the facts the model extracts. The audit can miss the one fact
  that matters: see [Evaluation](evaluation.md) for how often.
- In-place clearing by `ContextEditingMiddleware` repeats on every call, so it is audited once.

## How much to trust it

On small local models the per-fact verdict was more useful than the single `loss` number. A "kept"
verdict was reliable and a "lost" verdict was a warning, not proof. The aggregate score did not
predict later failures, so treat it as a debugging aid. Numbers and sample sizes are on the
[Evaluation](evaluation.md) page.
