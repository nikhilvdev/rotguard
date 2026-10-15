from . import goal_drift, holdout, holdout2, planted_facts, superseded

# The tuning suites: what `--suites all` runs.
SUITES = {
    "planted_facts": planted_facts.build,
    "superseded": superseded.build,
    "goal_drift": goal_drift.build,
}
# Held-out scenarios, selected by name only, so they are never mixed into tuning runs.
HOLDOUT = {"holdout": holdout.build, "holdout2": holdout2.build}
ALL_SUITES = {**SUITES, **HOLDOUT}
