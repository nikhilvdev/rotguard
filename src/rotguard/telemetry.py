"""OpenTelemetry events, so rotguard shows up in LangSmith and other tracing tools.

Every event rotguard stores (compaction, audit, recall, signals, marker_miss, handoff, staged) is
also emitted as a span event when `opentelemetry-api` is installed (`pip install rotguard[otel]`)
and a tracer provider is set. Without the package it does nothing. Attributes are an allowlist of
counts, scores, ids and flags: message text, lost answers and error strings are left out unless you
call `configure(include_text=True)`.
"""

from __future__ import annotations

from typing import Any

from rotguard.log import get_logger

log = get_logger(__name__)

_include_text = False

# per event kind: scalar fields to copy, list fields to count, list-of-string fields to copy
_SCALARS: dict[str, tuple[str, ...]] = {
    "compaction": ("channel", "turn", "chars_before", "chars_after"),
    "audit": (
        "channel", "turn", "loss", "loss_all", "facts_total", "answerable_before", "side_calls",
    ),
    "audit_error": ("channel",),
    "recall": ("turn", "score", "alert", "target", "rolling", "side_calls"),
    "recall_error": (),
    "signals": ("turn", "goal_overlap", "repetition", "tool_error_rate"),
    "marker_miss": ("turn",),
    "handoff": ("turn", "before", "after"),
    "staged": (
        "turn", "cleared", "summarized", "tokens_before", "tokens_after",
        "tokens_estimated", "forced",
    ),
}  # fmt: skip
_COUNTS: dict[str, tuple[str, ...]] = {
    "compaction": ("removed_ids", "rewritten_ids", "summary_ids"),
    "audit": ("lost",),
    "recall": ("asked", "missed", "stale"),
}
_STRINGS: dict[str, tuple[str, ...]] = {"signals": ("reasons",), "staged": ("stages",)}


def configure(*, include_text: bool = False) -> None:
    """Choose whether free text (lost answers, error messages) may be added to spans."""
    global _include_text
    _include_text = include_text


def attributes(kind: str, session_id: str, data: dict[str, Any]) -> dict[str, Any] | None:
    """The span attributes for one stored event, or None for kinds that are not emitted."""
    if kind not in _SCALARS:
        return None
    out: dict[str, Any] = {"rotguard.session_id": session_id, "rotguard.kind": kind}
    for k in _SCALARS[kind]:
        if isinstance(data.get(k), (bool, int, float, str)):
            out[f"rotguard.{k}"] = data[k]
    for k in _COUNTS.get(kind, ()):
        if isinstance(data.get(k), list):
            out[f"rotguard.{k}_count"] = len(data[k])
    for k in _STRINGS.get(kind, ()):
        if isinstance(data.get(k), list):
            out[f"rotguard.{k}"] = [str(x) for x in data[k]]
    if _include_text:
        if kind == "audit":
            out["rotguard.lost_answers"] = [str(r.get("answer")) for r in data.get("lost", [])]
        if kind in ("audit_error", "recall_error") and "error" in data:
            out["rotguard.error"] = str(data["error"])
    return out


def emit(kind: str, session_id: str, data: dict[str, Any]) -> None:
    """Emit one stored event as a span event. Never raises."""
    try:
        from opentelemetry import trace
    except ImportError:
        return
    try:
        attrs = attributes(kind, session_id, data)
        if attrs is None:
            return
        span = trace.get_current_span()
        if span.is_recording():  # the agent is being traced: attach to its span
            span.add_event(f"rotguard.{kind}", attrs)
        else:  # otherwise a short span of its own, so exporters still see it
            with trace.get_tracer("rotguard").start_as_current_span(
                f"rotguard.{kind}", attributes=attrs
            ) as s:
                s.add_event(f"rotguard.{kind}", attrs)
    except Exception as exc:  # telemetry must never break the agent
        log.warn("telemetry failed", error=repr(exc))
