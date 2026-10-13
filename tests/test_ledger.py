import pytest

from rotguard.ledger import Ledger


def test_key_supersedes_and_never_deletes():
    seen = []
    led = Ledger(on_change=lambda e: seen.append((e.id, e.status)))
    a = led.record("decision", "use Postgres", key="cache")
    b = led.record("decision", "use Redis", key="cache")
    assert [e.text for e in led.active()] == ["use Redis"]
    assert len(led.all()) == 2
    old = next(e for e in led.all() if e.id == a.id)
    assert old.status == "superseded" and old.superseded_by == b.id
    assert ("L1", "superseded") in seen


def test_explicit_supersedes_and_unknown_id():
    led = Ledger()
    a = led.record("constraint", "port 443")
    led.record("constraint", "port 8443", supersedes=a.id)
    assert [e.text for e in led.active()] == ["port 8443"]
    with pytest.raises(ValueError):
        led.record("goal", "x", supersedes="L99")


def test_render_omits_superseded_and_empty_is_blank():
    led = Ledger()
    assert led.render() == ""
    led.record("goal", "ship v1")
    led.record("decision", "old", key="k")
    led.record("decision", "new", key="k")
    out = led.render()
    assert "ship v1" in out and "new" in out and "old" not in out
