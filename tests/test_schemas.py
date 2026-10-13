from rotguard.schemas import DIR, render


def test_checked_in_schemas_are_current():
    for name, text in render().items():
        assert (DIR / name).read_text() == text, (
            f"{name} is stale; run `uv run python -m rotguard.schemas`"
        )
