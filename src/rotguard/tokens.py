"""Token counting for budgets (offload, staged cleanup).

Uses a provider's counting API when one is configured and reachable. Otherwise it estimates
characters / 4, which is rough (code and non-English text tokenize denser), and sets `estimated`
so anything that relies on the number can say so.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from rotguard.core import Message, ToolCallBlock
from rotguard.log import get_logger

log = get_logger(__name__)

CHARS_PER_TOKEN = 4


def estimate(text: str) -> int:
    return max(1, len(text) // CHARS_PER_TOKEN) if text else 0


class TokenCounter:
    """`count(messages)` -> tokens. `provider` takes the messages and returns an exact count."""

    def __init__(self, provider: Callable[[list[Message]], int] | None = None):
        self.provider = provider
        self.estimated = provider is None  # True once any count fell back to the estimate
        self._cache: dict[tuple[str, str], int] = {}

    def message(self, m: Message) -> int:
        key = (m.id, m.content_hash)
        if key not in self._cache:
            calls = sum(estimate(str(b.args)) for b in m.blocks if isinstance(b, ToolCallBlock))
            self._cache[key] = estimate(m.text) + calls
        return self._cache[key]

    def count(self, messages: list[Message]) -> int:
        if self.provider is not None:
            try:
                return int(self.provider(messages))
            except Exception as exc:  # unreachable or rate limited: never fail the agent
                self.estimated = True
                log.warn("token count failed, estimating", error=repr(exc))
        return sum(self.message(m) for m in messages)


def anthropic_counter(client: Any, model: str) -> Callable[[list[Message]], int]:
    """A provider for `TokenCounter` that uses Anthropic's token counting endpoint.

    Messages are sent as plain text turns, so tool blocks are counted as their text: close, not
    exact. It needs a network call per count; cache or call it sparingly.
    """

    def provider(messages: list[Message]) -> int:
        turns: list[dict[str, str]] = []
        for m in messages:
            role = "assistant" if m.role == "assistant" else "user"
            if turns and turns[-1]["role"] == role:
                turns[-1]["content"] += "\n\n" + m.text
            else:
                turns.append({"role": role, "content": m.text or "(empty)"})
        return int(client.messages.count_tokens(model=model, messages=turns).input_tokens)

    return provider
