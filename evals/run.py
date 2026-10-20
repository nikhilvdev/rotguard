"""Run suites x strategies against a model. Usage:
uv run python -m evals.run --model openai:gpt-4.1-mini --suites all --seeds 3
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

from logquill import FileTransport

from rotguard.log import configure

from . import baselines  # noqa: F401  (registers lc-summarize, anthropic-compaction, rg-staged)
from .core import STRATEGIES, run_scenario
from .models import make_client, print_estimate
from .suites import ALL_SUITES, SUITES


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mock")
    ap.add_argument("--suites", default="all")
    ap.add_argument("--strategies", default=",".join(STRATEGIES))
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument(
        "--turns", type=int, default=30, help="filler exchanges (rot grows with length)"
    )
    ap.add_argument("--out", default="evals/results")
    ap.add_argument(
        "--estimate",
        action="store_true",
        help="dry run on the mock model: count calls, spend nothing",
    )
    a = ap.parse_args(argv)

    client = make_client("mock" if a.estimate else a.model)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    stamp = f"{int(time.time())}-{client.name.replace(':', '_')}"
    configure([FileTransport(str(out / f"{stamp}.log.jsonl"))])
    names = list(SUITES) if a.suites == "all" else a.suites.split(",")
    results = []
    for s in names:
        for seed in range(a.seeds):
            for sc in ALL_SUITES[s](seed=seed, turns=a.turns):
                for strat in a.strategies.split(","):
                    results.append(run_scenario(sc, strat, client))

    if a.estimate:
        print_estimate(
            client, f"{a.model}, suites={a.suites}, strategies={a.strategies}, seeds={a.seeds}"
        )
        return
    agg = defaultdict(lambda: [0, 0, 0])
    for r in results:
        agg[(r.suite, r.strategy)][0] += r.passed
        agg[(r.suite, r.strategy)][1] += 1
        agg[(r.suite, r.strategy)][2] += r.stale
    print(f"model={client.name} turns={a.turns} seeds={a.seeds}")
    for (suite, strat), (p, n, st) in sorted(agg.items()):
        print(f"  {suite:14} {strat:10} {p}/{n} ({p / n:.0%})  stale-decision {st}/{n}")

    f = out / f"{stamp}.jsonl"
    f.write_text("\n".join(json.dumps(asdict(r)) for r in results))
    usage = out / f"{stamp}.usage.json"
    usage.write_text(json.dumps(client.totals(), indent=2))
    print("wrote", f, "and", usage)


if __name__ == "__main__":
    main()
