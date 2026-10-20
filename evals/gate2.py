"""Gate 2: does a compaction audit's loss score predict that a later probe fails?

For each scenario: compact the history with the naive summarizer, audit that compaction (no ledger),
then run the scenario's probe on the compacted context. Compare loss with probe failure.

    uv run python -m evals.gate2 --model ollama:qwen2.5:3b --turns 30 --seeds 3
"""

from __future__ import annotations

import argparse
import json
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path

from rotguard.core import Message, Provenance, TextBlock
from rotguard.probes import run_audit

from .core import STRATEGIES, SYSTEM, score
from .models import Client, make_client, print_estimate, purpose
from .suites import ALL_SUITES, SUITES

KEEP = 6  # messages the summarizer keeps verbatim (core.summarize default)


def neutral(history: list[dict], prefix: str) -> list[Message]:
    out = []
    for i, m in enumerate(history):
        kind = "tool" if m["content"].startswith("[tool result]") else m["role"]
        out.append(
            Message(
                id=f"{prefix}{i}",
                role=m["role"],
                blocks=[TextBlock(text=m["content"])],
                provenance=Provenance(kind=kind),
            )
        )
    return out


class HarnessCaller:
    def __init__(self, client: Client, purpose_name: str = "audit"):
        self.client, self.purpose_name = client, purpose_name

    def ask(self, context: list[Message], question: str) -> str:
        msgs = [{"role": m.role, "content": m.text} for m in context]
        with purpose(self.client, self.purpose_name):
            return self.client.complete(
                SYSTEM, [*msgs, {"role": "user", "content": question}], max_tokens=500
            )


@dataclass
class Row:
    suite: str
    scenario: str
    model: str
    loss: float | None
    answerable: int
    facts: int
    covered: bool  # did the audit extract the very fact the probe asks about?
    probe_failed: bool
    # the audit's verdict on the probed fact: True lost, False kept, None not checked
    probe_fact_lost: bool | None = None
    checked: list[dict] | None = None  # every fact the audit checked, with before/after


def auc(scores: list[float], failed: list[bool]) -> float | None:
    """P(random failed probe has a higher loss than a random passed one); ties count half."""
    pos = [s for s, f in zip(scores, failed, strict=True) if f]
    neg = [s for s, f in zip(scores, failed, strict=True) if not f]
    if not pos or not neg:
        return None
    return sum((p > n) + 0.5 * (p == n) for p in pos for n in neg) / (len(pos) * len(neg))


def permutation_p(
    scores: list[float], failed: list[bool], n: int = 5000, seed: int = 0
) -> float | None:
    observed = auc(scores, failed)
    if observed is None:
        return None
    rng, hits = random.Random(seed), 0
    for _ in range(n):
        shuffled = failed[:]
        rng.shuffle(shuffled)
        hits += (auc(scores, shuffled) or 0) >= observed
    return hits / n


def run(
    client: Client,
    turns: int,
    seeds: int,
    suites: list[str],
    cloze: int = 0,
    limit: int | None = None,
) -> list[Row]:
    rows = []
    for name in suites:
        for seed in range(seeds):
            for sc in ALL_SUITES[name](seed=seed, turns=turns)[:limit]:
                new_hist = STRATEGIES["summarize"](sc.history, client)
                reply = client.complete(
                    SYSTEM, [*new_hist, {"role": "user", "content": sc.probe.question}]
                )
                old = neutral(sc.history, "o")
                summary = Message(
                    id="summary",
                    role="user",
                    blocks=[TextBlock(text=new_hist[0]["content"])],
                    provenance=Provenance(kind="summary"),
                )
                new = [summary, *old[-KEEP:]]
                rep = run_audit(
                    old=old,
                    new=new,
                    removed=old[:-KEEP],
                    entries=[],
                    caller=HarnessCaller(client),
                    cloze=cloze,
                )
                hit = next(
                    (
                        c
                        for c in rep.checked
                        if any(re.search(p, c["answer"], re.I) for p in sc.probe.must)
                    ),
                    None,
                )
                lost = None if hit is None or not hit["before"] else not hit["after"]
                rows.append(
                    Row(
                        sc.suite,
                        f"{sc.id}/s{seed}",
                        client.name,
                        rep.loss,
                        rep.answerable_before,
                        rep.facts_total,
                        hit is not None,
                        not score(sc.probe, reply),
                        lost,
                        rep.checked,
                    )
                )
                print(f"  {rows[-1]}", flush=True)
    return rows


def summarize_rows(rows: list[Row]) -> dict:
    scored = [r for r in rows if r.loss is not None]
    s, f = [r.loss for r in scored], [r.probe_failed for r in scored]
    mean = lambda xs: sum(xs) / len(xs) if xs else None  # noqa: E731
    return {
        "n": len(rows),
        "n_scored": len(scored),
        "probe_failed": sum(r.probe_failed for r in rows),
        "audit_covered_probe_fact": sum(r.covered for r in rows),
        "mean_loss_when_failed": mean([r.loss for r in scored if r.probe_failed]),
        "mean_loss_when_passed": mean([r.loss for r in scored if not r.probe_failed]),
        "auc": auc(s, f),
        "permutation_p": permutation_p(s, f),
    }


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--turns", type=int, default=30)
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--suites", default="all")
    ap.add_argument("--cloze", type=int, default=0, help="fallback facts built from the text")
    ap.add_argument("--out", default="evals/results")
    ap.add_argument(
        "--estimate",
        action="store_true",
        help="dry run on the mock model: count calls, spend nothing",
    )
    a = ap.parse_args(argv)
    client = make_client("mock" if a.estimate else a.model)
    suites = list(SUITES) if a.suites == "all" else a.suites.split(",")
    rows = run(client, a.turns, a.seeds, suites, a.cloze)
    if a.estimate:
        print_estimate(client, f"gate2 {a.model}, suites={suites}, seeds={a.seeds}")
        return
    summary = summarize_rows(rows)
    summary["usage"] = client.totals()
    print(json.dumps(summary, indent=2))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tag = "-cloze" if a.cloze else ""
    stamp = f"{int(time.time())}-{client.name.replace(':', '_')}-t{a.turns}{tag}"
    path = out / f"gate2-{stamp}.jsonl"
    path.write_text("\n".join(json.dumps(r.__dict__) for r in rows))
    print("wrote", path)


if __name__ == "__main__":
    main()
