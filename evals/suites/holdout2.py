"""A second held-out set, written after the first one turned out to expose a leakage bug in the
audit (and so stopped being held out). Different facts again; none of it was run against any model
before the final audit configuration was fixed. Select it with `--suites holdout2`.
"""

from .holdout import build_from

PLANTED = [
    (
        "The retention period for audit logs is 7 years.",
        "How long are audit logs retained?",
        r"\b7\b|seven",
        {"kind": "constraint", "text": "Audit logs are retained for 7 years."},
    ),
    (
        "Use the sandbox account named acme-test for every trial run.",
        "Which account should trial runs use?",
        r"acme-test",
        {"kind": "constraint", "text": "Trial runs use the acme-test sandbox account."},
    ),
    (
        "Customer email addresses must never be written to the logs in plain text.",
        "How must customer email addresses be handled in the logs?",
        r"(never|not|don'?t|do not|no).{0,40}(log|plain)",
        {"kind": "constraint", "text": "Never log customer emails in plain text."},
    ),
    (
        "The nightly export runs at 02:15 UTC, so avoid deploys at that time.",
        "At what time does the nightly export run?",
        r"02:15|2:15",
        {"kind": "constraint", "text": "The nightly export runs at 02:15 UTC."},
    ),
]

CHANGES = [
    (
        "set the retry limit to 3",
        "raise the retry limit to 8",
        "What is the retry limit?",
        r"\b8\b|eight",
        r"\b3\b|three",
    ),
    (
        "deploy to the eu-west region",
        "deploy to the ap-south region instead",
        "Which region do we deploy to?",
        r"ap-south",
        r"eu-west",
    ),
    (
        "name the feature flag beta_search",
        "rename the feature flag to search_v2",
        "What is the feature flag called?",
        r"search_v2",
        r"beta_search",
    ),
]

GOALS = [
    (
        "reduce the cold-start time of the checkout service",
        r"cold-?start|checkout",
    ),
    (
        "document every public endpoint in the OpenAPI spec",
        r"document|openapi|endpoint",
    ),
]


def build(seed: int = 0, turns: int = 30, words: int = 120):
    return build_from("holdout2", PLANTED, CHANGES, GOALS, seed, turns, words, base=1000)
