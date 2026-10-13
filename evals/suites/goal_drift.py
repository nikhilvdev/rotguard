"""Suite 3: after compaction, does the agent still know its objective without re-asking?"""

from ..core import Probe, Scenario
from ._util import pad, rng_for

GOALS = [
    ("migrate the billing service from Python 2 to Python 3.12", r"python 3\.12|billing"),
    ("reduce p95 latency of the search endpoint below 200 ms", r"p95|latency|200 ?ms"),
    (
        "generate a vendor comparison report for the three cloud providers",
        r"vendor|comparison|cloud provider",
    ),
]
PROBE = (
    "Continue. State your overall objective in one sentence and your next step. "
    "Do not ask me questions."
)


def build(seed: int = 0, turns: int = 30, words: int = 120):
    out = []
    for k, (goal, must) in enumerate(GOALS):
        rng = rng_for(seed, k)
        h = [
            {"role": "user", "content": f"Your objective for this whole session: {goal}."},
            {"role": "assistant", "content": "Understood. I'll begin."},
        ]
        pad(h, rng, turns, words)
        # Fail if the agent asks for clarification instead of continuing.
        out.append(
            Scenario(
                "goal_drift",
                f"goal{k}",
                h,
                Probe(
                    PROBE,
                    [must],
                    [r"could you (clarify|remind)|what (is|was) (the|my) (task|objective|goal)"],
                ),
                ledger=[{"kind": "goal", "text": goal}],
            )
        )
    return out
