import json
from pathlib import Path

import yaml

from evals.models import make_client
from evals.nightly import main, measure, regressions


def test_regressions_flag_drops_beyond_the_margin_and_ignore_unmeasured_metrics():
    base = {"goal_retention": 0.8, "audit_validity": 1.0}
    assert regressions({"goal_retention": 0.7, "audit_validity": 0.9}, base, 0.15) == []
    bad = regressions({"goal_retention": 0.5, "audit_validity": 0.5}, base, 0.15)
    assert len(bad) == 2 and "goal_retention" in bad[0] and "audit_validity" in bad[1]
    assert regressions({"goal_retention": 0.9, "audit_validity": None}, base, 0.15) == []
    assert regressions({"goal_retention": 0.1}, {"goal_retention": None}, 0.15) == []


def test_measure_runs_one_scenario_per_suite_on_the_mock_model():
    out = measure(make_client("mock"), turns=6)
    assert (
        out["scenarios"] == 3
    )  # the tuning suites; held-out ones are not run and 0 <= out["goal_retention"] <= 1


def test_main_writes_a_baseline_then_passes_and_fails_against_it(tmp_path, capsys):
    base, args = tmp_path / "base.json", ["--model", "mock", "--turns", "6", "--out", str(tmp_path)]
    assert main([*args, "--baseline", str(base)]) == 0  # no baseline yet: nothing to compare
    assert "no baseline" in capsys.readouterr().out
    assert main([*args, "--baseline", str(base), "--write-baseline"]) == 0
    saved = json.loads(base.read_text())
    assert set(saved) == {"model", "date", "goal_retention", "audit_validity"}
    assert main([*args, "--baseline", str(base)]) == 0  # same model, same numbers
    saved["goal_retention"] = 2.0  # an impossible bar
    base.write_text(json.dumps(saved))
    assert main([*args, "--baseline", str(base)]) == 1
    assert "REGRESSION" in capsys.readouterr().out
    assert json.loads((tmp_path / "nightly.json").read_text())["usage"]["total"]["calls"] > 0


def test_the_workflow_is_valid_yaml_with_a_schedule_and_no_plaintext_secret():
    text = Path(".github/workflows/nightly-eval.yml").read_text()
    wf = yaml.safe_load(text)
    assert (
        wf[True]["schedule"][0]["cron"] and "workflow_dispatch" in wf[True]
    )  # `on` parses as True
    assert "${{ secrets.ANTHROPIC_API_KEY }}" in text and "sk-" not in text
