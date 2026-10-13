"""Re-score stored results with the current scorers (no model calls).
Usage: uv run python -m evals.rescore evals/results/<run>.jsonl [...]
Assumes seed 0 and the --turns used for the run only affect history, not probes."""

import json
import sys
from collections import defaultdict
from pathlib import Path

from .core import score
from .suites import SUITES


def main(paths):
    probes = {(s.suite, s.id): s.probe for b in SUITES.values() for s in b(seed=0, turns=1)}
    for path in paths:
        agg = defaultdict(lambda: [0, 0])
        for line in Path(path).read_text().splitlines():
            r = json.loads(line)
            ok = score(probes[(r["suite"], r["scenario"])], r["reply"])
            agg[(r["model"], r["suite"], r["strategy"])][0] += ok
            agg[(r["model"], r["suite"], r["strategy"])][1] += 1
        for (m, su, st), (p, n) in sorted(agg.items()):
            print(f"{m:26} {su:14} {st:10} {p}/{n}")


if __name__ == "__main__":
    main(sys.argv[1:])
