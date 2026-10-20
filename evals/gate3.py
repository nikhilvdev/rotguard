"""Gate 3: do recall probes fail before the task fails, at a cost that is worth it?

Each scenario's history is replayed one agent turn at a time, in two arms:

- **compact**: once the context is longer than `--threshold` messages it is summarized, and the
  next summary covers the previous summary plus the new messages, as a real agent's would (the
  compaction compounds).
- **control**: the same history, never compacted. An alert here is a false alarm, and a task
  failure here is the model failing the task for reasons other than rot.

At every turn the task probe (does the agent still answer the scenario's question?) and a recall
probe for each requested `--style` / `--target` combination are measured on the same context, so
the combinations are compared on identical simulations.

    uv run python -m evals.gate3 --model ollama:qwen2.5:3b --turns 30 --suites holdout
    uv run python -m evals.gate3 --model mock --estimate     # count calls first, spend nothing

Reported per model and combination: lead time (turns between the first alert and the first task
failure), the lead per fact (first miss of that fact to the task failure), misses, false alarms
against the control arm, the calibrated `pass_at` next to the default, and probe tokens as a
percentage of agent tokens.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from rotguard.core import Message, Provenance, TextBlock
from rotguard.ledger import Ledger
from rotguard.probes import (
    FactPool,
    RecallMonitor,
    RecallPolicy,
    RecallResult,
    calibrate,
    ledger_tokens,
    run_recall,
)
from rotguard.probes.quiz import norm
from rotguard.testing import broken_pairs

from .core import STRATEGIES, SYSTEM, score
from .gate2 import KEEP, HarnessCaller, neutral
from .models import Client, make_client, print_estimate, purpose
from .suites import ALL_SUITES, SUITES

SUMMARY_PREFIX = "[summary of earlier work]"


@dataclass
class Turn:
    messages: int  # messages in the context at this turn
    compacted: bool  # a compaction has happened by this turn
    recall: float | None
    task_failed: bool
    missed: list[str] = field(default_factory=list)
    stale: list[str] = field(default_factory=list)


@dataclass
class Series:
    suite: str
    scenario: str
    model: str
    arm: str  # "compact" or "control"
    style: str
    target: str
    turns: list[Turn] = field(default_factory=list)
    task_fail_turn: int | None = None  # index into `turns`
    alert_turn: int | None = None  # at the default pass_at
    first_miss: dict[str, int] = field(default_factory=dict)  # fact id -> first turn it was missed
    probe_calls: int = 0
    broken_histories: int = 0


def _context(hist: list[dict]) -> list[Message]:
    ctx = neutral(hist, "c")
    if hist and hist[0]["content"].startswith(SUMMARY_PREFIX):
        ctx[0] = Message(
            id="summary",
            role="user",
            blocks=[TextBlock(text=hist[0]["content"])],
            provenance=Provenance(kind="summary"),
        )
    return ctx


def replay(
    sc: Any,
    client: Client,
    combos: list[tuple[str, str]],
    policy: RecallPolicy,
    *,
    arm: str,
    step: int,
    threshold: int,
    seed: int,
) -> list[Series]:
    """One simulation of `sc`; one Series per (style, target) combination."""
    ledger = Ledger()
    for e in sc.ledger:
        ledger.record(**e)
    entries = ledger.all()
    caller = HarnessCaller(client, "probe")
    everything = neutral(sc.history, "h")  # stable ids, so the pool sees each message once
    series = {c: Series(sc.suite, sc.id, client.name, arm, *c) for c in combos}
    mons = {c: RecallMonitor(policy) for c in combos}
    rngs = {c: random.Random(seed) for c in combos}
    pool = FactPool()
    hist: list[dict] = []
    prev, compacted = 0, False
    for k in range(KEEP + step, len(sc.history) + 1, step):
        pool.add(everything[prev:k], ledger_tokens(entries))
        hist += sc.history[prev:k]
        prev = k
        if arm == "compact" and len(hist) > threshold:
            hist = STRATEGIES["summarize"](hist, client)  # summarizes the last summary too
            compacted = True
        ctx = _context(hist)
        reply = client.complete(SYSTEM, [*hist, {"role": "user", "content": sc.probe.question}])
        task_failed = not score(sc.probe, reply)
        covered = ledger_tokens(entries)
        for c in combos:
            style, target = c
            facts = None
            if target == "history":
                facts = [f for f in pool.facts if norm(f.answer) not in covered]
            with purpose(client, "probe"):
                r = run_recall(
                    context=ctx,
                    entries=entries,
                    caller=caller,
                    sample=policy.sample,
                    rng=rngs[c],
                    style=style,  # type: ignore[arg-type]
                    facts=facts,
                )
            s = series[c]
            s.probe_calls += r.side_calls
            s.broken_histories += bool(broken_pairs(ctx))
            i = len(s.turns)
            s.turns.append(Turn(len(ctx), compacted, r.score, task_failed, r.missed, r.stale))
            for f in r.missed:
                s.first_miss.setdefault(f, i)
            if s.alert_turn is None and mons[c].record(RecallResult(i, r.score)):
                s.alert_turn = i
            if s.task_fail_turn is None and task_failed:
                s.task_fail_turn = i
    return list(series.values())


def alert_turn(s: Series, pass_at: float, alert_after: int) -> int | None:
    """When a run of `alert_after` probes below `pass_at` first completes."""
    streak = 0
    for i, t in enumerate(s.turns):
        if t.recall is None:
            continue
        streak = streak + 1 if t.recall < pass_at else 0
        if streak == alert_after:
            return i
    return None


def _median(xs: list[int]) -> int | None:
    return sorted(xs)[len(xs) // 2] if xs else None


def summarize_arm(
    compact: list[Series], control: list[Series], pass_at: float, alert_after: int
) -> dict[str, Any]:
    alerts = {id(s): alert_turn(s, pass_at, alert_after) for s in [*compact, *control]}
    failed = [s for s in compact if s.task_fail_turn is not None]
    leads = [
        s.task_fail_turn - a
        for s in failed
        if (a := alerts[id(s)]) is not None and s.task_fail_turn is not None
    ]
    fact_leads = [
        s.task_fail_turn - m
        for s in failed
        if s.task_fail_turn is not None
        for m in s.first_miss.values()
    ]
    return {
        "pass_at": pass_at,
        "runs": len(compact),
        "task_failed": len(failed),
        "alerted_at_least_one_turn_before": sum(1 for x in leads if x >= 1),
        "alerted_at_or_before": sum(1 for x in leads if x >= 0),
        "median_lead_turns": _median(leads),
        "median_lead_turns_per_fact": _median(fact_leads),
        "missed": sum(1 for s in failed if alerts[id(s)] is None),
        "control_runs": len(control),
        "control_task_failed": sum(1 for s in control if s.task_fail_turn is not None),
        # an alert where nothing went wrong: the control run's task passed
        "false_alarms": sum(
            1 for s in control if alerts[id(s)] is not None and s.task_fail_turn is None
        ),
        # an alert in a control run whose task failed anyway: the model cannot do it unaided
        "control_alerts_task_also_failed": sum(
            1 for s in control if alerts[id(s)] is not None and s.task_fail_turn is not None
        ),
        # an alert in a compacted run whose task never failed
        "alerts_without_failure": sum(
            1 for s in compact if alerts[id(s)] is not None and s.task_fail_turn is None
        ),
    }


def summarize_series(
    runs: list[Series], policy: RecallPolicy, usage: dict[str, Any] | None = None
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for style, target in sorted({(s.style, s.target) for s in runs}):
        mine = [s for s in runs if (s.style, s.target) == (style, target)]
        compact = [s for s in mine if s.arm == "compact"]
        control = [s for s in mine if s.arm == "control"]
        clean = [
            t.recall for s in mine for t in s.turns if not t.compacted and t.recall is not None
        ]
        cal = calibrate(clean) if clean else None
        n_turns = sum(len(s.turns) for s in mine) or 1
        out[f"{style}/{target}"] = {
            "clean_recall_mean": (sum(clean) / len(clean)) if clean else None,
            "clean_probes": len(clean),
            "default": summarize_arm(compact, control, policy.pass_at, policy.alert_after),
            "calibrated": summarize_arm(compact, control, cal, policy.alert_after)
            if cal is not None
            else None,
            "stale_answers": sum(len(t.stale) for s in mine for t in s.turns),
            "probe_calls_per_turn": sum(s.probe_calls for s in mine) / n_turns,
            "broken_histories": sum(s.broken_histories for s in mine),
        }
    if usage is not None:
        out["usage"] = usage
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--turns", type=int, default=30)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--suites", default="all")
    ap.add_argument("--step", type=int, default=2, help="history messages per turn")
    ap.add_argument("--threshold", type=int, default=14, help="compact above this many messages")
    ap.add_argument("--style", default="cued,list", help="comma list of cued, list")
    ap.add_argument("--target", default="ledger,history", help="comma list of ledger, history")
    ap.add_argument("--pass-at", type=float, default=0.5)
    ap.add_argument("--alert-after", type=int, default=2)
    ap.add_argument("--out", default="evals/results")
    ap.add_argument(
        "--estimate",
        action="store_true",
        help="dry run on the mock model: count calls, spend nothing",
    )
    a = ap.parse_args(argv)
    client = make_client("mock" if a.estimate else a.model)
    # the list style only exists for ledger entries; history facts are always cued
    combos = [
        (st, tg)
        for st in a.style.split(",")
        for tg in a.target.split(",")
        if not (st == "list" and tg == "history")
    ]
    policy = RecallPolicy(
        mode="every", every=1, pass_at=a.pass_at, alert_after=a.alert_after, max_probes=10**6
    )
    runs: list[Series] = []
    for name in list(SUITES) if a.suites == "all" else a.suites.split(","):
        for seed in range(a.seeds):
            for sc in ALL_SUITES[name](seed=seed, turns=a.turns):
                for arm in ("compact", "control"):
                    runs += replay(
                        sc,
                        client,
                        combos,
                        policy,
                        arm=arm,
                        step=a.step,
                        threshold=a.threshold,
                        seed=seed,
                    )
                lead = ", ".join(
                    f"{s.style}/{s.target} fail@{s.task_fail_turn} alert@{s.alert_turn}"
                    for s in runs[-2 * len(combos) : -len(combos)]
                )
                print(f"  {sc.id}/s{seed}: {lead}", flush=True)
    if a.estimate:
        what = f"gate3 {a.model}, suites={a.suites}, seeds={a.seeds}, combos={combos}"
        print_estimate(client, what)
        return
    summary = summarize_series(runs, policy, client.totals())
    print(json.dumps(summary, indent=2))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    stamp = f"gate3-{int(time.time())}-{client.name.replace(':', '_')}-t{a.turns}"
    (out / f"{stamp}.jsonl").write_text("\n".join(json.dumps(asdict(s)) for s in runs))
    (out / f"{stamp}.summary.json").write_text(json.dumps(summary, indent=2))
    print("wrote", out / f"{stamp}.jsonl", "and the summary")


if __name__ == "__main__":
    main()
