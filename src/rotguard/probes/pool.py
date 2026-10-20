"""A session-long pool of facts from the conversation that the ledger does not cover.

Recall probes with `target="history"` draw from this pool. Facts stay in it after their source
message is compacted away, because whether the summary kept them is exactly what the probe asks.
Asking the same facts again over time gives a recall curve per fact.
"""

from __future__ import annotations

from rotguard.core import Message
from rotguard.ledger import Entry

from .quiz import Fact, cloze_facts, fact_tokens, norm


def ledger_tokens(entries: list[Entry]) -> set[str]:
    """Tokens the ledger already covers (superseded entries too), to keep out of the pool."""
    return {t for e in entries for t in fact_tokens(e.text)[0]}


class FactPool:
    def __init__(self, per_message: int = 2, max_facts: int = 200):
        self.per_message, self.max_facts = per_message, max_facts
        self.facts: list[Fact] = []
        self._seen: set[str] = set()
        self._tokens: set[str] = set()

    def add(self, messages: list[Message], skip: set[str] | None = None) -> None:
        """Add cloze facts from new messages, leaving out tokens in `skip` or already pooled."""
        for m in messages:
            if m.id in self._seen:
                continue
            self._seen.add(m.id)
            if len(self.facts) >= self.max_facts:
                continue
            for f in cloze_facts([m], self.per_message, {*(skip or ()), *self._tokens}):
                f.id = f"pool:{len(self.facts) + 1}"
                self._tokens.add(norm(f.answer))
                self.facts.append(f)
