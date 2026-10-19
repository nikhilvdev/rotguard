"""Gate 3: do recall probes fail before the task fails, at a cost that is worth it?

For each scenario the history is replayed one agent turn at a time. Once it is longer than
`--threshold` messages it is compacted with the naive summarizer (as an agent would), and at every
turn two things are measured on the context the agent would use: the task probe (does the agent
still answer the scenario's question?) and a recall probe over the scenario's ledger entries.

    uv run python -m evals.gate3 --model ollama:qwen2.5:3b --turns 30 --seeds 3 --suites holdout

Reported per run: the turn the task first failed, the turn a recall alert (`alert_after` failed
probes in a row) first fired, the lead between them, false alarms (alert, task never failed),
misses (task failed, no alert), probe calls per turn, and a `pass_at` calibrated on the turns
before any compaction.

shortcut: each turn compacts the original prefix again instead of the previous summary, so
compaction does not compound; fine for a first read, change it if Gate 3 is close.
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
from rotguard.probes import RecallMonitor, RecallPolicy, RecallResult, calibrate, run_recall

from .core import STRATEGIES, SYSTEM, score
from .gate2 import KEEP, HarnessCaller, neutral
from .models import Client, make_client
from .suites import ALL_SUITES, SUITES


@dataclass
class Turn:
    messages: int  # history length at this turn
    compacted: bool
    recall: float | None
    task_failed: bool


@dataclass
class Series:
    suite: str
    scenario: str
    model: str
    turns: list[Turn] = field(default_factory=list)
    task_fail_turn: int | None = None  # index into `turns`
    alert_turn: int | None = None
    probe_calls: int = 0


def replay(
    sc: Any, client: Client, policy: RecallPolicy, step: int, threshold: int, seed: int
) -> Series:
    ledger = Ledger()
    for e in sc.ledger:
        ledger.record(**e)
    entries = ledger.active()
    mon, rng = RecallMonitor(policy), random.Random(seed)
    out, caller = Series(sc.suite, sc.id, client.name), HarnessCaller(client)
    for k in range(KEEP + step, len(sc.history) + 1, step):
        prefix, compacted = sc.history[:k], k > threshold
        hist = STRATEGIES["summarize"](prefix, client) if compacted else prefix
        reply = client.complete(SYSTEM, [*hist, {"role": "user", "content": sc.probe.question}])
        ctx = neutral(hist, "h")
        if compacted:
            ctx[0] = Message(
                id="summary",
                role="user",
                blocks=[TextBlock(text=hist[0]["content"])],
                provenance=Provenance(kind="summary"),
            )
        r = run_recall(context=ctx, entries=entries, caller=caller, sample=policy.sample, rng=rng)
        out.probe_calls += r.side_calls
        i = len(out.turns)
        out.turns.append(Turn(k, compacted, r.score, not score(sc.probe, reply)))
        if out.alert_turn is None and mon.record(RecallResult(i, r.score)):
            out.alert_turn = i
        if out.task_fail_turn is None and out.turns[-1].task_failed:
            out.task_fail_turn = i
    return out


def summarize_series(runs: list[Series], policy: RecallPolicy) -> dict[str, Any]:
    failed = [s for s in runs if s.task_fail_turn is not None]
    leads = [
        s.task_fail_turn - s.alert_turn
        for s in failed
        if s.alert_turn is not None and s.task_fail_turn is not None
    ]
    clean = [t.recall for s in runs for t in s.turns if not t.compacted and t.recall is not None]
    n_turns = sum(len(s.turns) for s in runs)
    return {
        "n": len(runs),
        "task_failed": len(failed),
        "alerted_before_or_at_failure": sum(1 for x in leads if x >= 0),
        "alerted_at_least_one_turn_before": sum(1 for x in leads if x >= 1),
        "median_lead_turns": sorted(leads)[len(leads) // 2] if leads else None,
        "missed": sum(1 for s in failed if s.alert_turn is None),
        "false_alarms": sum(
            1 for s in runs if s.alert_turn is not None and s.task_fail_turn is None
        ),
        "no_failure_runs": sum(1 for s in runs if s.task_fail_turn is None),
        "probe_calls_per_turn": (sum(s.probe_calls for s in runs) / n_turns) if n_turns else None,
        "clean_recall_mean": (sum(clean) / len(clean)) if clean else None,
        "calibrated_pass_at": calibrate(clean) if clean else None,
        "policy_pass_at": policy.pass_at,
    }


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--turns", type=int, default=30)
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--suites", default="all")
    ap.add_argument("--step", type=int, default=2, help="history messages per turn")
    ap.add_argument("--threshold", type=int, default=14, help="compact above this many messages")
    ap.add_argument("--pass-at", type=float, default=0.5)
    ap.add_argument("--alert-after", type=int, default=2)
    ap.add_argument("--out", default="evals/results")
    a = ap.parse_args(argv)
    client = make_client(a.model)
    policy = RecallPolicy(
        mode="every", every=1, pass_at=a.pass_at, alert_after=a.alert_after, max_probes=10**6
    )
    runs = []
    for name in list(SUITES) if a.suites == "all" else a.suites.split(","):
        for seed in range(a.seeds):
            for sc in ALL_SUITES[name](seed=seed, turns=a.turns):
                runs.append(replay(sc, client, policy, a.step, a.threshold, seed))
                r = runs[-1]
                print(
                    f"  {r.scenario}/s{seed} fail@{r.task_fail_turn} alert@{r.alert_turn}",
                    flush=True,
                )
    summary = summarize_series(runs, policy)
    print(json.dumps(summary, indent=2))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"gate3-{int(time.time())}-{client.name.replace(':', '_')}-t{a.turns}.jsonl"
    path.write_text("\n".join(json.dumps(asdict(s)) for s in runs))
    print("wrote", path)


if __name__ == "__main__":
    main()
