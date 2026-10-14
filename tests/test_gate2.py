from evals.gate2 import auc, main, permutation_p


def test_auc_and_permutation():
    assert auc([0.9, 0.8, 0.1, 0.2], [True, True, False, False]) == 1.0
    assert auc([0.1, 0.2, 0.9, 0.8], [True, True, False, False]) == 0.0
    assert auc([0.5, 0.5], [True, False]) == 0.5
    assert auc([0.5, 0.6], [True, True]) is None
    assert permutation_p([0.9, 0.8, 0.7, 0.1, 0.2, 0.15], [True] * 3 + [False] * 3, n=500) < 0.1


def test_runs_end_to_end_on_the_mock_model(tmp_path, capsys):
    main(["--model", "mock", "--turns", "3", "--suites", "planted_facts", "--out", str(tmp_path)])
    assert '"n": 4' in capsys.readouterr().out
    assert list(tmp_path.glob("gate2-*.jsonl"))
