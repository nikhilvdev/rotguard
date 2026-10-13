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
