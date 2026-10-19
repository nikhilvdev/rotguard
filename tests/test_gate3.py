from evals.gate3 import Series, Turn, main, summarize_series
from rotguard.probes import RecallPolicy


def series(task, alert, recalls=()):
    s = Series("s", "x", "m", task_fail_turn=task, alert_turn=alert, probe_calls=len(recalls))
    s.turns = [Turn(2 * i, False, r, False) for i, r in enumerate(recalls)]
    return s


def test_summary_counts_lead_misses_and_false_alarms():
    runs = [
        series(5, 3),  # alert 2 turns early
        series(4, 4),  # alert at the failure: not early
        series(6, None),  # missed
        series(None, 2),  # false alarm
        series(None, None, [1.0, 1.0, 0.5, 1.0]),  # quiet run; clean scores feed calibration
    ]
    out = summarize_series(runs, RecallPolicy())
    assert out["task_failed"] == 3 and out["missed"] == 1 and out["false_alarms"] == 1
    assert out["alerted_before_or_at_failure"] == 2 and out["alerted_at_least_one_turn_before"] == 1
    assert out["median_lead_turns"] == 2 and out["no_failure_runs"] == 2
    assert out["calibrated_pass_at"] is not None and out["probe_calls_per_turn"] > 0


def test_runs_end_to_end_on_the_mock_model(tmp_path, capsys):
    main(["--model", "mock", "--turns", "6", "--suites", "planted_facts", "--out", str(tmp_path)])
    assert '"n": 4' in capsys.readouterr().out
    assert list(tmp_path.glob("gate3-*.jsonl"))
