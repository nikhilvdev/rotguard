"""A small, fixed eval for CI: one scenario per suite, one seed, a hosted or local model.

    uv run python -m evals.nightly --model anthropic:claude-haiku-5-5
    uv run python -m evals.nightly --model ... --write-baseline    # record today's numbers

Two numbers are compared with the baseline in `evals/nightly_baseline.json` and the run fails if
either drops by more than `--margin`:

- `goal_retention`: share of scenarios the agent still gets right after a compaction when the
  ledger is shown (strategy `summarize+ledger`).
- `audit_validity`: of the facts the audit calls "kept", the share whose probe then passed.

With one scenario per suite the numbers are coarse. This catches a break, not a 2% drift.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from . import baselines  # noqa: F401  (registers the extra strategies)
from .core import run_scenario
from .gate2 import run as run_gate2
from .models import Client, make_client
from .suites import ALL_SUITES, SUITES

BASELINE = Path(__file__).with_name("nightly_baseline.json")
STRATEGY = "summarize+ledger"


def measure(client: Client, turns: int = 30) -> dict[str, Any]:
    results = [
        run_scenario(sc, STRATEGY, client)
        for name in SUITES
        for sc in ALL_SUITES[name](seed=0, turns=turns)[:1]
    ]
    rows = run_gate2(client, turns, 1, list(SUITES), limit=1)
    kept = [r for r in rows if r.probe_fact_lost is False]
    return {
        "goal_retention": sum(r.passed for r in results) / len(results) if results else None,
        "audit_validity": (sum(not r.probe_failed for r in kept) / len(kept)) if kept else None,
        "scenarios": len(results),
        "audit_kept_verdicts": len(kept),
    }


def regressions(now: dict[str, Any], baseline: dict[str, Any], margin: float) -> list[str]:
    """Metrics that fell by more than `margin` below the baseline. A metric that could not be
    measured (None) is not a regression: there is nothing to compare."""
    out = []
    for k in ("goal_retention", "audit_validity"):
        a, b = now.get(k), baseline.get(k)
        if a is not None and b is not None and a < b - margin:
            out.append(f"{k}: {a:.2f} is more than {margin:.2f} below the baseline {b:.2f}")
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--turns", type=int, default=30)
    ap.add_argument("--margin", type=float, default=0.15)
    ap.add_argument("--out", default="evals/results")
    ap.add_argument("--baseline", default=str(BASELINE))
    ap.add_argument("--write-baseline", action="store_true")
    a = ap.parse_args(argv)
    client = make_client(a.model)
    now = {"model": client.name, "date": time.strftime("%Y-%m-%d"), **measure(client, a.turns)}
    now["usage"] = client.totals()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "nightly.json").write_text(json.dumps(now, indent=2))
    print(json.dumps({k: v for k, v in now.items() if k != "usage"}, indent=2))
    base = Path(a.baseline)
    if a.write_baseline:
        base.write_text(
            json.dumps(
                {k: now[k] for k in ("model", "date", "goal_retention", "audit_validity")}, indent=2
            )
            + "\n"
        )
        print("wrote", base)
        return 0
    if not base.exists():
        print(f"no baseline at {base}: nothing to compare (run with --write-baseline once)")
        return 0
    bad = regressions(now, json.loads(base.read_text()), a.margin)
    for line in bad:
        print("REGRESSION", line)
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
