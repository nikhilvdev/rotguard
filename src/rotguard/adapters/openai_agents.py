"""OpenAI Agents SDK: wrap a session so rotguard sees what it trims or summarizes.

    session = RotguardSession(SQLiteSession("t1"), client, model="gpt-4.1", audit=AuditPolicy(...))
    Runner.run(agent, "hi", session=session)

Works with any object that has the SDK's session methods (`get_items`, `add_items`, `pop_item`,
`clear_items`), such as a trimming or summarizing session, so the SDK stays an optional import.
`get_items` runs once per run, just before the model is called, so that is where a compaction done
by the inner session shows up. Side calls use `client.responses.create` and never enter the session.
Nothing is injected into the model's input here, so the marker check (which needs an injected
instruction) is not available on this adapter.
"""

from __future__ import annotations

import hashlib
import inspect
import uuid
from collections.abc import Callable
from typing import Any

from rotguard.core import Message, Provenance, TextBlock
from rotguard.ledger import Ledger
from rotguard.log import get_logger
from rotguard.probes import AuditPolicy, RecallPolicy, RecallResult
from rotguard.session import Session
from rotguard.store import SQLiteStore

log = get_logger(__name__)


def _text(item: dict[str, Any]) -> str:
    if item.get("type") == "function_call":
        return f"{item.get('name', '')}({item.get('arguments', '')})"
    if item.get("type") == "function_call_output":
        return str(item.get("output", ""))
    content = item.get("content", "")
    if isinstance(content, str):
        return content
    return "".join(p.get("text", "") for p in content if isinstance(p, dict))


def to_neutral(items: list[dict[str, Any]]) -> list[Message]:
    out: list[Message] = []
    seen: dict[str, int] = {}
    for item in items:
        kind = item.get("type")
        role = "assistant" if item.get("role") == "assistant" or kind == "function_call" else "user"
        prov = "tool" if kind == "function_call_output" else role
        m = Message(
            id=str(item.get("id") or ""),
            role=role,  # type: ignore[arg-type]
            blocks=[TextBlock(text=_text(item))],
            provenance=Provenance(kind=prov),  # type: ignore[arg-type]
            extra={"raw_message": item},
        )
        if not m.id:
            n = seen[m.content_hash] = seen.get(m.content_hash, 0) + 1
            m.id = f"o-{hashlib.sha256(m.content_hash.encode()).hexdigest()[:12]}-{n}"
        out.append(m)
    return out


class OpenAICaller:
    """Side calls through the Responses API with the same model."""

    def __init__(self, client: Any, model: str, instructions: str | None = None):
        self.client, self.model, self.instructions = client, model, instructions

    def ask(self, context: list[Message], question: str) -> str:
        items = [m.extra["raw_message"] for m in context]
        kw: dict[str, Any] = {
            "model": self.model,
            "input": [*items, {"role": "user", "content": question}],
        }
        if self.instructions:
            kw["instructions"] = self.instructions
        return str(self.client.responses.create(**kw).output_text)


class RotguardSession:
    def __init__(
        self,
        inner: Any,
        client: Any,
        model: str,
        store: SQLiteStore | str | None = None,
        session_id: str | None = None,
        audit: AuditPolicy | None = None,
        recall: RecallPolicy | None = None,
        on_alert: Callable[[RecallResult], None] | None = None,
        instructions: str | None = None,
    ):
        self._inner = inner
        self.session_id = session_id or getattr(inner, "session_id", None) or uuid.uuid4().hex
        self.store = store if isinstance(store, SQLiteStore) else SQLiteStore(store or ":memory:")
        self.ledger = Ledger(
            on_change=lambda e: self.store.upsert_entry(self.session_id, e.model_dump())
        )
        self.session = Session(self.store, self.session_id, self.ledger, audit, recall, on_alert)
        self._caller = OpenAICaller(client, model, instructions)

    def __getattr__(self, name: str) -> Any:  # anything else is the inner session
        return getattr(self._inner, name)

    async def get_items(self, limit: int | None = None) -> list[Any]:
        items = self._inner.get_items(limit)
        items = await items if inspect.isawaitable(items) else items
        try:
            self.session.turn += 1
            self.session.observe_history(to_neutral(list(items)))
            self.session.run_pending(self._caller)
        except Exception as exc:  # never break the agent
            log.warn("rotguard observe failed", error=repr(exc))
        return list(items)

    async def add_items(self, items: list[Any]) -> None:
        out = self._inner.add_items(items)
        if inspect.isawaitable(out):
            await out

    async def pop_item(self) -> Any:
        out = self._inner.pop_item()
        return await out if inspect.isawaitable(out) else out

    async def clear_items(self) -> None:
        out = self._inner.clear_items()
        if inspect.isawaitable(out):
            await out
