from rotguard.schemas import DIR, render


def test_checked_in_schemas_are_current():
    for name, text in render().items():
        assert (DIR / name).read_text() == text, (
            f"{name} is stale; run `uv run python -m rotguard.schemas`"
        )


def test_real_reports_and_events_match_their_schemas():
    """Validate what the producers actually emit, with unknown fields forbidden."""
    from test_cli import fake, make_trace

    from rotguard.probes import AuditPolicy, RecallPolicy
    from rotguard.replay import replay
    from rotguard.report.html import build
    from rotguard.report.models import AuditEvent, RecallEvent, Report

    src = make_trace(":memory:")
    recall = RecallPolicy(mode="every", every=1, signals=False)
    dst, sid = replay(src, "orig", fake(), audit=AuditPolicy(mode="every"), recall=recall)
    report = build(dst, sid)
    Report.model_validate(report)
    assert report["audits"] and report["recall"]  # the validation covered real rows
    for e in dst.events(sid, "audit"):
        AuditEvent.model_validate(e)
    for e in dst.events(sid, "recall"):
        RecallEvent.model_validate(e)
    Report.model_validate(build(src, "orig"))  # a trace with no probes validates too
