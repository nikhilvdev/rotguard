import json

import pytest

from evals.core import Result
from evals.gate1 import main, summarize, verdict


def res(suite, strategy, passed, stale=False):
    return Result(suite, "s", strategy, "m", 0, passed, "", stale)


def test_summary_and_verdict_compare_each_strategy_with_and_without_the_ledger():
    rs = [
        res("goal_drift", "summarize", False), res("goal_drift", "summarize", True),
        res("goal_drift", "summarize+ledger", True), res("goal_drift", "summarize+ledger", True),
        res("superseded", "summarize", False, stale=True), res("superseded", "summarize", False, stale=True),
        res("superseded", "summarize+ledger", True, stale=False), res("superseded", "summarize+ledger", False, stale=True),
        res("planted_facts", "truncate", True), res("planted_facts", "truncate+ledger", True),
    ]  # fmt: skip
    s = summarize(rs)
    assert s["summarize"]["goal_retention"] == {"value": 0.5, "n": 2}
    assert s["summarize"]["goal_retention_with_ledger"] == {"value": 1.0, "n": 2}
    assert s["summarize"]["stale_decision_rate"]["value"] == 1.0
    assert s["summarize"]["stale_decision_rate_with_ledger"]["value"] == 0.5
    v = verdict(s)
    assert v["goal_retention"]["summarize"] == "helped"
    assert v["stale_decision_rate"]["summarize"] == "helped"  # lower is better
    assert v["planted_fact_recall"]["truncate"] == "no change"  # a null result is reported as one
    assert "summarize" not in v["planted_fact_recall"]  # nothing measured for it: not claimed


def test_a_ledger_that_makes_things_worse_is_called_out():
    s = summarize([res("goal_drift", "none", True), res("goal_drift", "none+ledger", False)])
    assert verdict(s)["goal_retention"]["none"] == "did not help"


def test_runs_end_to_end_on_the_mock_model_and_estimate_spends_nothing(tmp_path, capsys):
    base = ["--model", "mock", "--turns", "6", "--seeds", "1", "--strategies", "none,truncate"]
    main([*base, "--out", str(tmp_path)])
    [summary] = tmp_path.glob("gate1-*.summary.json")
    data = json.loads(summary.read_text())
    assert set(data["summary"]) == {"none", "truncate"} and data["usage"]["total"]["calls"] > 0
    capsys.readouterr()
    other = tmp_path / "est"
    main([*base, "--out", str(other), "--estimate"])
    assert "ESTIMATE" in capsys.readouterr().out and not other.exists()
    with pytest.raises(SystemExit, match="unknown strategies"):
        main([*base[:-1], "bogus", "--out", str(tmp_path)])
