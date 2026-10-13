"""Suite 2: change a decision mid-run; the agent must follow the new one."""

from ..core import Probe, Scenario
from ._util import pad, rng_for

CASES = [
    (
        "use PostgreSQL for the cache layer",
        "actually, switch the cache layer to Redis",
        "Which store should the cache layer use?",
        r"redis",
        r"postgres",
    ),
    (
        "name the release branch release/2.4",
        "rename the release branch to release/2.5",
        "What is the release branch called?",
        r"2\.5",
        r"2\.4",
    ),
    (
        "send the report to alice@example.com",
        "send the report to bob@example.com instead",
        "Who receives the report?",
        r"\bbob\b",
        r"\balice\b",
    ),
]


def build(seed: int = 0, turns: int = 25, words: int = 120):
    out = []
    for k, (old, new, q, must, must_not) in enumerate(CASES):
        rng = rng_for(seed, k)
        h = [
            {"role": "user", "content": f"Please {old}."},
            {"role": "assistant", "content": "Done, noted."},
        ]
        pad(h, rng, turns // 2, words)
        h += [
            {"role": "user", "content": f"Change of plan: {new}."},
            {"role": "assistant", "content": "Updated."},
        ]
        pad(h, rng, turns // 2, words)
        out.append(
            Scenario(
                "superseded",
                f"case{k}",
                h,
                Probe(q, [must], stale=[must_not]),
                ledger=[
                    {"kind": "decision", "text": f"Please {old}.", "key": "d"},
                    {"kind": "decision", "text": f"Change of plan: {new}.", "key": "d"},
                ],
            )
        )
    return out
