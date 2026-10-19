"""Anthropic SDK wrapper: watch requests, recognise server-side compaction, audit it.

Both compaction modes (threshold `compact_20260112`, and on-demand `compaction={...}`)
put an assistant `compaction` content block into `messages`, and the API ignores everything before
it. So the context the model really sees is "from the last compaction block on", and a compaction
shows up as that effective context shrinking between two requests. Observing requests is enough:
the old context is the previous request, the new one is the current request.

    rg = RotguardAnthropic(anthropic.Anthropic(), audit=AuditPolicy(mode="every"))
    resp = rg.beta.messages.create(model=..., messages=..., betas=[...], context_management=...)

Nothing is changed in what is sent. Audit side calls reuse the request's `model`, `system`, `tools`
and `betas`, because the API requires tools to be declared when the history contains tool blocks and
requires the beta header for a history that contains a compaction block.
"""

from __future__ import annotations

import contextlib
import hashlib
import uuid
from collections.abc import Callable
from typing import Any

from rotguard.core import Message, Provenance, TextBlock, ToolCallBlock, ToolResultBlock
from rotguard.ledger import Ledger
from rotguard.log import get_logger
from rotguard.probes import AuditPolicy, RecallPolicy, RecallResult
from rotguard.session import Session
from rotguard.store import SQLiteStore

log = get_logger(__name__)


def _plain(x: Any) -> Any:
    """SDK block or message object -> plain dict (None fields dropped); dicts pass through."""
    if hasattr(x, "model_dump"):
        return x.model_dump(exclude_none=True)
    return x


def _flat(content: Any) -> str:
    if isinstance(content, str):
        return content
    return "\n".join(b.get("text", "") for b in (_plain(c) for c in content) if isinstance(b, dict))


def _neutral(role: str, blocks: list[dict[str, Any]], summary: bool = False) -> Message:
    out: list[Any] = []
    for b in blocks:
        t = b.get("type")
        if t == "text":
            out.append(TextBlock(text=b.get("text", "")))
        elif t == "compaction":
            out.append(TextBlock(text=b.get("content") or ""))
        elif t == "tool_use":
            out.append(ToolCallBlock(id=b["id"], name=b["name"], args=b.get("input", {})))
        elif t == "tool_result":
            out.append(
                ToolResultBlock(call_id=b["tool_use_id"], content=_flat(b.get("content", "")))
            )
    only_results = bool(out) and all(isinstance(x, ToolResultBlock) for x in out)
    kind = "summary" if summary else "tool" if only_results else role
    return Message(
        id="",
        role="assistant" if role == "assistant" else "user",
        blocks=out,
        provenance=Provenance(kind=kind),  # type: ignore[arg-type]
        extra={"raw_message": {"role": role, "content": blocks}},
    )


def to_neutral(messages: list[Any]) -> list[Message]:
    """The context the model effectively sees: from the last compaction block onward."""
    plain = [
        {"role": m["role"], "content": [_plain(b) for b in _as_blocks(m["content"])]}
        for m in messages
    ]
    start, pos = 0, 0
    for i, m in enumerate(plain):
        for j, b in enumerate(m["content"]):
            if b.get("type") == "compaction":
                start, pos = i, j
    out: list[Message] = []
    for i, m in enumerate(plain[start:], start):
        blocks = m["content"]
        if i == start and any(b.get("type") == "compaction" for b in blocks):
            block = blocks[pos]
            digest = hashlib.sha256((block.get("content") or "").encode()).hexdigest()[:12]
            summary = _neutral(m["role"], [block], summary=True)
            summary.id = f"compaction-{digest}"
            out.append(summary)
            blocks = blocks[pos + 1 :]
            if not blocks:
                continue
        out.append(_neutral(m["role"], blocks))
    seen: dict[str, int] = {}
    for msg in out:
        if not msg.id:
            n = seen[msg.content_hash] = seen.get(msg.content_hash, 0) + 1
            msg.id = f"a-{msg.content_hash}-{n}"
    return out


def _as_blocks(content: Any) -> list[Any]:
    return [{"type": "text", "text": content}] if isinstance(content, str) else list(content)


class AnthropicCaller:
    """Side calls through the same client, with the request's model, system, tools and betas."""

    def __init__(
        self,
        client: Any,
        *,
        model: str,
        system: Any = None,
        tools: Any = None,
        betas: Any = None,
        max_tokens: int = 1024,
    ):
        self.client, self.model, self.system = client, model, system
        self.tools, self.betas, self.max_tokens = tools, betas, max_tokens

    def ask(self, context: list[Message], question: str) -> str:
        messages = [m.extra["raw_message"] for m in context]
        messages.append({"role": "user", "content": question})
        kw: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": messages,
        }
        for name in ("system", "tools", "betas"):
            if getattr(self, name):
                kw[name] = getattr(self, name)
        resp = self.client.beta.messages.create(**kw)
        return "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")


class RotguardAnthropic:
    def __init__(
        self,
        client: Any,
        store: SQLiteStore | str | None = None,
        session_id: str | None = None,
        audit: AuditPolicy | None = None,
        recall: RecallPolicy | None = None,
        on_alert: Callable[[RecallResult], None] | None = None,
    ):
        self._client = client
        self.store = store if isinstance(store, SQLiteStore) else SQLiteStore(store or ":memory:")
        self.session_id = session_id or uuid.uuid4().hex
        self.ledger = Ledger(
            on_change=lambda e: self.store.upsert_entry(self.session_id, e.model_dump())
        )
        self.session = Session(self.store, self.session_id, self.ledger, audit, recall, on_alert)
        self.beta = _Beta(self)

    def __getattr__(self, name: str) -> Any:  # everything else is the plain client
        return getattr(self._client, name)

    def _observe(self, kwargs: dict[str, Any]) -> None:
        try:
            self.session.turn += 1
            self.session.observe_history(to_neutral(kwargs.get("messages", [])))
            caller = AnthropicCaller(
                self._client,
                model=kwargs["model"],
                system=kwargs.get("system"),
                tools=kwargs.get("tools"),
                betas=kwargs.get("betas"),
            )
            self.session.run_pending(caller)
        except Exception as exc:  # never break the agent
            with contextlib.suppress(Exception):
                log.warn("rotguard observe failed", error=repr(exc))


class _Messages:
    def __init__(self, owner: RotguardAnthropic):
        self._owner = owner

    def create(self, **kwargs: Any) -> Any:
        self._owner._observe(kwargs)
        return self._owner._client.beta.messages.create(**kwargs)

    def stream(self, **kwargs: Any) -> Any:
        self._owner._observe(kwargs)
        return self._owner._client.beta.messages.stream(**kwargs)


class _Beta:
    def __init__(self, owner: RotguardAnthropic):
        self.messages = _Messages(owner)
