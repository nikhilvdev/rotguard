"""Baseline compaction strategies beyond the naive ones in `core.py` (importing registers them).

- `lc-summarize`: LangChain's own `SummarizationMiddleware`, called on the history once.
- `anthropic-compaction`: Anthropic's on-demand server-side compaction (Anthropic models only).
  The request shape comes from the SDK's type definitions (`compaction={"type": "summarize"}`,
  beta `compact-2026-09-04`); it has been checked against a fake client, **not the live API**.
- `rg-staged`: rotguard's staged cleanup idea on harness histories: replace old tool outputs with
  stubs first, summarize only if the history is still over budget.

ContextForge is not wired in: it needs the library installed and its API read first.
"""

from __future__ import annotations

from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from rotguard.policies.offload import OffloadPolicy, make_stub
from rotguard.tokens import estimate

from .core import STRATEGIES, SYSTEM, summarize
from .models import Client, purpose

KEEP_LAST = 6  # messages the strategies keep verbatim, as core.summarize does
TOOL_PREFIX = "[tool result]"


class ClientChat(BaseChatModel):
    """Lets LangChain components call an eval `Client`."""

    client: Any

    @property
    def _llm_type(self) -> str:
        return "eval-client"

    def _generate(  # type: ignore[no-untyped-def]
        self, messages: list[BaseMessage], stop=None, run_manager=None, **kw: Any
    ):
        turns = [
            {"role": "assistant" if m.type == "ai" else "user", "content": str(m.content)}
            for m in messages
        ]
        with purpose(self.client, "summarizer"):
            text = self.client.complete(SYSTEM, turns, max_tokens=500)
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=text))])


def lc_summarize(keep_last: int = KEEP_LAST):
    """LangChain's `SummarizationMiddleware`, asked to compact the whole history once."""

    def f(history: list[dict], client: Client) -> list[dict]:
        from langchain.agents.middleware import SummarizationMiddleware

        msgs = [
            (AIMessage if m["role"] == "assistant" else HumanMessage)(
                content=m["content"], id=f"m{i}"
            )
            for i, m in enumerate(history)
        ]
        mw = SummarizationMiddleware(
            model=ClientChat(client=client), trigger=("messages", 1), keep=("messages", keep_last)
        )
        update = mw.before_model({"messages": msgs}, None)  # type: ignore[arg-type]
        if not update:
            return history
        kept = update["messages"][1:]  # the first item removes everything else
        return [
            {"role": "assistant" if m.type == "ai" else "user", "content": str(m.content)}
            for m in kept
        ]

    return f


def anthropic_compaction(keep_last: int = KEEP_LAST):
    """Compact everything but the last messages with Anthropic's on-demand compaction."""

    def f(history: list[dict], client: Client) -> list[dict]:
        inner = getattr(client, "inner", client)
        if not hasattr(inner, "compact"):
            raise RuntimeError("anthropic-compaction needs an anthropic: model")
        old, recent = history[:-keep_last], history[-keep_last:]
        with purpose(client, "summarizer"):
            block = inner.compact(SYSTEM, old)
        return [{"role": "assistant", "content": [block]}, *recent]

    return f


def rg_staged(budget_tokens: int = 1500, clear_keep: int = 3, keep_last: int = KEEP_LAST):
    """Clear old tool outputs first; summarize only if still over budget."""
    policy = OffloadPolicy()

    def tokens(h: list[dict]) -> int:
        return sum(estimate(str(m["content"])) for m in h)

    def f(history: list[dict], client: Client) -> list[dict]:
        if tokens(history) <= budget_tokens:
            return history
        new = list(history)
        tool = [i for i, m in enumerate(new) if str(m["content"]).startswith(TOOL_PREFIX)]
        for i in tool[: max(0, len(tool) - clear_keep)]:
            new[i] = {**new[i], "content": make_stub(f"offload:h{i}", new[i]["content"], policy)}
        if tokens(new) <= budget_tokens:
            return new
        return summarize(keep_last)(new, client)

    return f


STRATEGIES.update(
    {
        "lc-summarize": lc_summarize(),
        "anthropic-compaction": anthropic_compaction(),
        "rg-staged": rg_staged(),
    }
)
