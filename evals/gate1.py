"""Gate 1: does the ledger help? Each strategy with and without `+ledger`, on the suites that test it.

    uv run python -m evals.gate1 --model anthropic:claude-haiku-5-5 --seeds 3
    uv run python -m evals.gate1 --model mock --estimate        # count calls first, spend nothing

Reported per strategy, without and with the ledger, with the run counts:

- goal retention: pass rate on the `goal_drift` suite after a forced compaction;
- stale-decision rate: share of `superseded` runs that act on the replaced value;
- planted-fact recall: pass rate on the `planted_facts` suite.

The verdict says whether the ledger raised goal retention and planted-fact recall, and lowered the
stale-decision rate, for each strategy. A null result is reported as one.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any

from . import baselines  # noqa: F401  (registers lc-summarize, anthropic-compaction, rg-staged)
from .core import STRATEGIES, Result, run_scenario
from .models import make_client, print_estimate
from .suites import ALL_SUITES

SUITE_FOR = {
    "goal_retention": "goal_drift",
    "stale_decision_rate": "superseded",
    "planted_fact_recall": "planted_facts",
}


def summarize(results: list[Result]) -> dict[str, Any]:
    """Per strategy: each metric without and with the ledger, as value and run count."""
    cells: dict[tuple[str, str, bool], list[Result]] = defaultdict(list)
    for r in results:
        base, _, suffix = r.strategy.partition("+")
        cells[(base, r.suite, suffix == "ledger")].append(r)
    out: dict[str, Any] = {}
    for base in sorted({k[0] for k in cells}):
        row: dict[str, Any] = {}
        for metric, suite in SUITE_FOR.items():
            for led in (False, True):
                rs = cells.get((base, suite, led), [])
                val = None
                if rs:
                    val = sum(r.stale if metric == "stale_decision_rate" else r.passed for r in rs)
                    val /= len(rs)
                row[f"{metric}{'_with_ledger' if led else ''}"] = {"value": val, "n": len(rs)}
        out[base] = row
    return out


def verdict(summary: dict[str, Any]) -> dict[str, Any]:
    """Per metric and strategy: did the ledger help (higher retention and recall, lower stale)?"""
    res: dict[str, Any] = {}
    for base, row in summary.items():
        for metric in SUITE_FOR:
            a, b = row[metric]["value"], row[f"{metric}_with_ledger"]["value"]
            if a is None or b is None:
                continue
            better = b < a if metric == "stale_decision_rate" else b > a
            res.setdefault(metric, {})[base] = (
                "helped" if better else "no change" if a == b else "did not help"
            )
    return res


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--turns", type=int, default=30)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--strategies", default="none,truncate,summarize,lc-summarize,rg-staged")
    ap.add_argument("--out", default="evals/results")
    ap.add_argument(
        "--estimate",
        action="store_true",
        help="dry run on the mock model: count calls, spend nothing",
    )
    a = ap.parse_args(argv)
    client = make_client("mock" if a.estimate else a.model)
    bases = a.strategies.split(",")
    unknown = [b for b in bases if b not in STRATEGIES]
    if unknown:
        raise SystemExit(f"unknown strategies: {unknown}; known: {sorted(STRATEGIES)}")
    results: list[Result] = []
    for suite in SUITE_FOR.values():
        for seed in range(a.seeds):
            for sc in ALL_SUITES[suite](seed=seed, turns=a.turns):
                for base in bases:
                    for strat in (base, f"{base}+ledger"):
                        results.append(run_scenario(sc, strat, client))
    if a.estimate:
        print_estimate(client, f"gate1 {a.model}, strategies={bases}, seeds={a.seeds}")
        return
    summary = summarize(results)
    out_json = {
        "model": client.name,
        "turns": a.turns,
        "seeds": a.seeds,
        "summary": summary,
        "verdict": verdict(summary),
        "usage": client.totals(),
    }
    print(json.dumps(out_json, indent=2))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    stamp = f"gate1-{int(time.time())}-{client.name.replace(':', '_')}-t{a.turns}"
    (out / f"{stamp}.jsonl").write_text("\n".join(json.dumps(asdict(r)) for r in results))
    (out / f"{stamp}.summary.json").write_text(json.dumps(out_json, indent=2))
    print("wrote", out / f"{stamp}.summary.json")


if __name__ == "__main__":
    main()
