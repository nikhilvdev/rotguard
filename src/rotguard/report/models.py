"""Typed shapes of what rotguard reports: the MCP `report` tool's output and the payloads of the
stored `audit` and `recall` events. They exist to generate `schemas/` and are checked against the
real producers in the tests, so a changed payload fails a test instead of silently drifting.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from rotguard.ledger import Entry


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class RecallRow(_Strict):
    turn: int
    score: float | None
    alert: bool
    missed: list[str]
    stale: list[str]
    target: str
    rolling: float | None


class CompactionRow(_Strict):
    turn: int
    channel: str
    removed: int
    rewritten: int
    chars_before: int
    chars_after: int


class AuditRow(_Strict):
    turn: int
    loss: float | None
    loss_all: float | None
    facts_total: int
    lost: list[str]
    suggestions: list[str]


class SignalRow(_Strict):
    turn: int
    reasons: list[str]


class Report(_Strict):
    """What `rotguard.report.html.build` returns, and the MCP `report` tool."""

    session: str
    items: int
    recall: list[RecallRow]
    compactions: list[CompactionRow]
    audits: list[AuditRow]
    signals: list[SignalRow]
    marker_misses: list[int]
    errors: int
    ledger: list[Entry]


class AuditEvent(_Strict):
    """A stored `audit` event: `AuditReport.to_dict()` plus where and when."""

    ts: float
    kind: Literal["audit"]
    channel: str
    turn: int
    loss: float | None
    facts_total: int
    answerable_before: int
    lost: list[dict[str, Any]]
    checked: list[dict[str, Any]]
    loss_all: float | None
    suggestions: list[str]
    side_calls: int


class RecallEvent(_Strict):
    """A stored `recall` event: `RecallResult.to_dict()` plus the rolling score and the alert."""

    ts: float
    kind: Literal["recall"]
    turn: int
    score: float | None
    asked: list[str]
    missed: list[str]
    stale: list[str]
    side_calls: int
    rolling: float | None
    alert: bool
    target: str
