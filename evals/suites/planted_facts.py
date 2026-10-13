"""Suite 1: plant a constraint early, pad with tool output, then ask for it."""

from ..core import Probe, Scenario
from ._util import pad, rng_for

FACTS = [
    (
        "Never write to the `prod_orders` table; use `staging_orders` only.",
        "Which table may you write to?",
        [r"staging_orders"],
    ),
    (
        "The deploy window closes at 17:40 UTC on Fridays.",
        "When does the Friday deploy window close?",
        [r"17:40"],
    ),
    (
        "The customer's budget cap is 12,500 EUR.",
        "What is the customer's budget cap?",
        [r"12[, ]?500"],
    ),
    (
        "Use port 8443 for the internal gateway, never 443.",
        "Which port should the internal gateway use?",
        [r"8443"],
    ),
]


def build(seed: int = 0, turns: int = 30, words: int = 120):
    out = []
    for k, (fact, q, must) in enumerate(FACTS):
        rng = rng_for(seed, k)
        h = [
            {
                "role": "user",
                "content": f"Task: audit the order service. Important constraint: {fact}",
            },
            {"role": "assistant", "content": "Understood. Starting the audit."},
        ]
        pad(h, rng, turns, words)
        out.append(
            Scenario(
                "planted_facts",
                f"fact{k}",
                h,
                Probe(q, must, stale=[r"\bprod_orders\b"] if k == 0 else []),
                ledger=[{"kind": "constraint", "text": fact}],
            )
        )
    return out
