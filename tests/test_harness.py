from evals.core import Probe, score
from evals.run import main
from evals.suites import SUITES


def test_scoring():
    p = Probe("q", [r"redis"], [r"postgres"])
    assert score(p, "Use Redis.")
    assert not score(p, "Use Redis, not Postgres.")
    assert not score(p, "dunno")


def test_suites_deterministic_and_nonempty():
    for build in SUITES.values():
        a, b = build(seed=1, turns=4), build(seed=1, turns=4)
        assert a and [s.history for s in a] == [s.history for s in b]


def test_mock_run_no_compaction_passes(tmp_path, capsys):
    main(["--model", "mock", "--strategies", "none", "--turns", "2", "--out", str(tmp_path)])
    assert "planted_facts" in capsys.readouterr().out


def test_run_writes_logquill_records(tmp_path):
    import json

    from rotguard.log import configure

    main(["--model", "mock", "--strategies", "none", "--turns", "2", "--out", str(tmp_path)])
    configure([])  # close the file transport so records are flushed
    lines = [
        json.loads(line)
        for f in tmp_path.glob("*.log.jsonl")
        for line in f.read_text().splitlines()
    ]
    assert lines and lines[0]["level"] == "INFO"
    assert lines[0]["meta"]["model"] == "mock" and lines[0]["meta"]["suite"]


def test_holdout_suite_is_deterministic_separate_and_scoreable():
    from evals.core import score
    from evals.suites import ALL_SUITES, HOLDOUT, SUITES

    assert "holdout" not in SUITES and "holdout" in ALL_SUITES  # never part of `--suites all`
    a, b = HOLDOUT["holdout"](seed=1, turns=4), HOLDOUT["holdout"](seed=1, turns=4)
    assert len(a) == 9 and [s.history for s in a] == [s.history for s in b]
    assert len({s.id for s in a}) == 9
    by_id = {s.id: s for s in a}
    # a correct answer passes and the stale value fails, for a changed decision
    ch = by_id["change0"]
    assert score(ch.probe, "The database runs on port 6432.")
    assert not score(ch.probe, "It runs on 5432.")
    assert score(by_id["plant1"].probe, "Every 45 days.") and not score(
        by_id["plant1"].probe, "Every 30 days."
    )


def test_openai_key_prefers_the_dedicated_variable_and_rejects_placeholders(monkeypatch):
    import pytest

    from evals.models import openai_key

    real = "sk-" + "x" * 40
    monkeypatch.delenv("ROTGUARD_OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "ollama")  # a placeholder, as for a local server
    with pytest.raises(RuntimeError, match="ROTGUARD_OPENAI_API_KEY"):
        openai_key()
    monkeypatch.setenv("ROTGUARD_OPENAI_API_KEY", real)
    assert openai_key() == real
    monkeypatch.setenv("OPENAI_API_KEY", "sk-" + "y" * 40)
    assert openai_key() == real  # the dedicated one wins


def test_second_holdout_is_separate_and_distinct_from_the_first():
    from evals.suites import ALL_SUITES, SUITES

    assert "holdout2" in ALL_SUITES and "holdout2" not in SUITES
    first = ALL_SUITES["holdout"](seed=0, turns=4)
    second = ALL_SUITES["holdout2"](seed=0, turns=4)
    assert len(second) == 9 and all(s.suite == "holdout2" for s in second)
    texts = lambda ss: {s.history[0]["content"] for s in ss}  # noqa: E731  (the stated fact)
    assert not texts(first) & texts(second)  # no template shared
    assert [s.history for s in second] == [
        s.history for s in ALL_SUITES["holdout2"](seed=0, turns=4)
    ]
