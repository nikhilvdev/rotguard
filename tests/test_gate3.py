import json

from evals.gate3 import Series, Turn, alert_turn, main, replay, summarize_series
from evals.suites import ALL_SUITES
from rotguard.probes import RecallPolicy


def series(arm, scores, task_fail=None, first_miss=None, compacted_from=99):
    s = Series("s", "x", "m", arm, "cued", "ledger", task_fail_turn=task_fail)
    s.turns = [Turn(2 * i, i >= compacted_from, r, False) for i, r in enumerate(scores)]
    s.first_miss = first_miss or {}
    return s


def test_alert_turn_needs_consecutive_scores_below_pass_at():
    s = series("compact", [1.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    assert alert_turn(s, 0.5, 2) == 4 and alert_turn(s, 0.5, 4) is None
    assert alert_turn(series("compact", [None, 0.0, None, 0.0]), 0.5, 2) == 3  # None is skipped


def test_summary_counts_lead_misses_and_false_alarms_against_the_control_arm():
    runs = [
        series("compact", [1.0, 0.0, 0.0, 1.0, 1.0], task_fail=4, first_miss={"a": 1, "b": 2}),
        series("compact", [1.0, 1.0, 1.0, 1.0], task_fail=3),  # missed: no alert before failure
        series("compact", [1.0, 1.0], task_fail=None),  # task never failed
        series("control", [1.0, 0.0, 0.0, 1.0]),  # false alarm
        series("control", [1.0, 1.0, 1.0, 1.0], task_fail=2),  # model failed the task unaided
        series("control", [0.0, 0.0, 0.0], task_fail=1),  # alert, and the task failed anyway
    ]
    out = summarize_series(runs, RecallPolicy(), {"model": "m"})
    d = out["cued/ledger"]["default"]
    assert d["task_failed"] == 2 and d["missed"] == 1
    assert d["alerted_at_least_one_turn_before"] == 1 and d["median_lead_turns"] == 2
    assert d["median_lead_turns_per_fact"] in (2, 3)  # facts missed at turns 1 and 2, fail at 4
    assert d["false_alarms"] == 1  # only the control run whose task passed
    assert d["control_task_failed"] == 2 and d["control_alerts_task_also_failed"] == 1
    assert d["alerts_without_failure"] == 0
    assert out["cued/ledger"]["calibrated"] is not None and out["usage"] == {"model": "m"}


class Recording:
    """A client that records what each summarization was given."""

    name = "recording"

    def __init__(self):
        self.summaries, self.n = [], 0

    def complete(self, system, messages, max_tokens=400):
        if system.startswith("Summarize this agent transcript"):
            self.n += 1
            self.summaries.append(messages[0]["content"])
            return f"SUMMARY-{self.n}"
        return "ok"


def test_compaction_compounds_and_the_control_arm_never_compacts():
    sc = ALL_SUITES["planted_facts"](seed=0, turns=20)[0]
    policy = RecallPolicy(mode="every", every=1, max_probes=10**6)
    client = Recording()
    [compact] = replay(
        sc, client, [("cued", "ledger")], policy, arm="compact", step=2, threshold=8, seed=0
    )
    assert client.n >= 2
    assert "SUMMARY-1" in client.summaries[1]  # the second summary covers the first
    assert any(t.compacted for t in compact.turns)
    client2 = Recording()
    [control] = replay(
        sc, client2, [("cued", "ledger")], policy, arm="control", step=2, threshold=8, seed=0
    )
    assert client2.n == 0 and not any(t.compacted for t in control.turns)


def test_runs_end_to_end_on_the_mock_model(tmp_path, capsys):
    main(
        [
            "--model",
            "mock",
            "--turns",
            "8",
            "--seeds",
            "1",
            "--suites",
            "planted_facts",
            "--out",
            str(tmp_path),
        ]
    )
    out = capsys.readouterr().out
    assert "cued/ledger" in out and "cued/history" in out and "list/ledger" in out
    assert "list/history" not in out  # history facts are always cued
    [summary] = tmp_path.glob("gate3-*.summary.json")
    data = json.loads(summary.read_text())
    assert data["usage"]["total"]["calls"] > 0 and "probe" in data["usage"]["by_purpose"]
    assert list(tmp_path.glob("gate3-*.jsonl"))


def test_estimate_spends_nothing_and_writes_nothing(tmp_path, capsys):
    main(
        [
            "--model",
            "openai:gpt",
            "--turns",
            "6",
            "--seeds",
            "1",
            "--suites",
            "planted_facts",
            "--out",
            str(tmp_path),
            "--estimate",
        ]
    )
    assert "ESTIMATE" in capsys.readouterr().out and not list(tmp_path.iterdir())
