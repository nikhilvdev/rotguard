from . import goal_drift, planted_facts, superseded

SUITES = {
    "planted_facts": planted_facts.build,
    "superseded": superseded.build,
    "goal_drift": goal_drift.build,
}
