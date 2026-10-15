"""Held-out scenarios, written after the audit fixes were developed and never used to tune them.

Different facts and phrasing from the three tuning suites, and some facts that hold no identifier
or number. Not part of `--suites all`; run it by name: `--suites holdout`.
"""

from ..core import Probe, Scenario
from ._util import pad, rng_for

# (suite-like kind, setup, optional change, question, must, stale, ledger entries)
PLANTED = [
    (
        "Remember that the API rate limit is 600 requests per minute for this customer.",
        "What is the API rate limit for this customer?",
        r"\b600\b",
        {"kind": "constraint", "text": "The API rate limit is 600 requests per minute."},
    ),
    (
        "Please rotate the signing key every 45 days, no exceptions.",
        "How often should the signing key be rotated?",
        r"\b45\b",
        {"kind": "constraint", "text": "Rotate the signing key every 45 days."},
    ),
    (
        "All alerts must go to the #ops-oncall channel and nowhere else.",
        "Which channel should alerts go to?",
        r"ops-oncall",
        {"kind": "constraint", "text": "Alerts go to #ops-oncall only."},
    ),
    (
        "The reviewer for this migration is Dana, and she must approve every schema change.",
        "Who must approve each schema change?",
        r"\bdana\b",
        {"kind": "constraint", "text": "Dana must approve every schema change."},
    ),
]

SUPERSEDED = [
    (
        "run the database on port 5432",
        "change of plan, the database now runs on port 6432",
        "Which port does the database run on?",
        r"6432",
        r"5432",
    ),
    (
        "assign the incident to Maria",
        "reassign the incident to Jonas instead",
        "Who is the incident assigned to?",
        r"\bjonas\b",
        r"\bmaria\b",
    ),
    (
        "keep the logs for 30 days",
        "extend the log retention to 90 days",
        "How long are the logs kept?",
        r"\b90\b",
        r"\b30\b",
    ),
]

GOALS = [
    (
        "ship the invoice export feature behind a feature flag",
        r"invoice|export|feature flag",
    ),
    (
        "cut the nightly batch job runtime from four hours to under one hour",
        r"nightly|batch|runtime|hour",
    ),
]
PROBE = (
    "Continue. State your overall objective in one sentence and your next step. "
    "Do not ask me questions."
)
ASKS = r"could you (clarify|remind)|what (is|was) (the|my) (task|objective|goal)"


def build_from(name, plants, changes, goals, seed, turns, words, base=0):
    """Build scenarios from template tuples. `base` keeps each suite's random filler distinct."""
    out = []
    for k, (fact, q, must, entry) in enumerate(plants):
        h = [
            {"role": "user", "content": f"Task: help migrate the billing service. {fact}"},
            {"role": "assistant", "content": "Understood. Starting the migration work."},
        ]
        pad(h, rng_for(seed, base + 100 + k), turns, words)
        out.append(Scenario(name, f"plant{k}", h, Probe(q, [must]), ledger=[entry]))
    for k, (old, new, q, must, stale) in enumerate(changes):
        rng = rng_for(seed, base + 200 + k)
        h = [
            {"role": "user", "content": f"Please {old}."},
            {"role": "assistant", "content": "Done, noted."},
        ]
        pad(h, rng, turns // 2, words)
        h += [
            {"role": "user", "content": f"Update: {new}."},
            {"role": "assistant", "content": "Updated."},
        ]
        pad(h, rng, turns // 2, words)
        ledger = [
            {"kind": "decision", "text": f"Please {old}.", "key": "d"},
            {"kind": "decision", "text": f"Update: {new}.", "key": "d"},
        ]
        out.append(Scenario(name, f"change{k}", h, Probe(q, [must], stale=[stale]), ledger=ledger))
    for k, (goal, must) in enumerate(goals):
        h = [
            {"role": "user", "content": f"Your objective for this whole session: {goal}."},
            {"role": "assistant", "content": "Understood. I'll begin."},
        ]
        pad(h, rng_for(seed, base + 300 + k), turns, words)
        probe = Probe(PROBE, [must], [ASKS])
        out.append(Scenario(name, f"aim{k}", h, probe, ledger=[{"kind": "goal", "text": goal}]))
    return out


def build(seed: int = 0, turns: int = 30, words: int = 120):
    return build_from("holdout", PLANTED, SUPERSEDED, GOALS, seed, turns, words)
