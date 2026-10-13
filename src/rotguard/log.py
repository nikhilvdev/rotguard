"""Logging for rotguard, built on logquill (https://pypi.org/project/logquill/).

Library code calls `get_logger(__name__)`. A logger with no transports emits
nothing, so rotguard is silent until the application (or the eval harness)
attaches transports with `configure()`.
"""

from __future__ import annotations

from typing import Any

from logquill import Level, Logger, Transport

_root = Logger("rotguard", level=Level.INFO)


def configure(transports: list[Transport], level: int | str | Level = Level.INFO) -> Logger:
    """Replace the transports on the shared rotguard logger; children inherit them."""
    global _root
    _root.close()
    _root = Logger("rotguard", level=level, transports=transports)
    return _root


class _LazyLogger:
    """Resolves to a child of the *current* root on each use. logquill children copy the
    root's transports when created, so a module-level logger would miss `configure()`."""

    def __init__(self, name: str):
        self._name = name

    def __getattr__(self, attr: str) -> Any:
        return getattr(_root.child(self._name) if self._name else _root, attr)


def get_logger(name: str = "") -> Logger:
    """Logger named `rotguard.<name>`; safe to create at import time."""
    return _LazyLogger(name.removeprefix("rotguard."))  # type: ignore[return-value]
