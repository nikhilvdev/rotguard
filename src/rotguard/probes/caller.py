"""The one thing the audit needs from a model: answer a question about a given context."""

from __future__ import annotations

from typing import Protocol

from rotguard.core import Message


class SideCaller(Protocol):
    """Ask `question` as if it were the next user turn after `context`.

    A side call: its prompt and answer must never be written into the agent's history.
    An empty `context` is allowed (used for fact extraction).
    """

    def ask(self, context: list[Message], question: str) -> str: ...
