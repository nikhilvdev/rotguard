"""LangChain v1 adapter: `RotguardMiddleware` and message conversion.

Hook mapping (see the plan, section 7.1):
  before_agent    load the session's ledger from the store
  before_model    observe the stored history; record any compaction
  wrap_model_call observe the final request; add the ledger to this call only
  after_model     remember the reply's id as the source of ledger entries
  after_agent     final observation, flush
Every hook fails open: an error is logged and the agent carries on unchanged.
"""

from __future__ import annotations

import contextlib
import functools
import uuid
from collections.abc import Awaitable, Callable
from typing import Any, Literal

from langchain.agents.middleware import AgentMiddleware
from langchain.tools import tool
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage, ToolMessage

from rotguard.core import (
    Message,
    Observer,
    Provenance,
    TextBlock,
    ToolCallBlock,
    ToolResultBlock,
    default_provenance,
)
from rotguard.ledger import Entry, Ledger
from rotguard.log import get_logger
from rotguard.store import SQLiteStore

log = get_logger(__name__)

LEDGER_MESSAGE_NAME = "rotguard_ledger"


# --- conversion -------------------------------------------------------------------------


def _text(content: Any) -> tuple[str, bool]:
    """Plain text of LangChain content, and whether the original was a block list."""
    if isinstance(content, str):
        return content, False
    parts = [
        b if isinstance(b, str) else b.get("text", "")
        for b in content
        if isinstance(b, str) or b.get("type") == "text"
    ]
    return "".join(parts), True


def to_neutral(m: AnyMessage, fallback_id: str) -> Message:
    text, listed = _text(m.content)
    extra: dict[str, Any] = {"lc_type": m.type}
    if listed:
        extra["content_raw"] = m.content
    for attr in ("additional_kwargs", "response_metadata"):
        if getattr(m, attr, None):
            extra[attr] = getattr(m, attr)
    if getattr(m, "name", None):
        extra["name"] = m.name
    blocks: list[Any] = []
    if isinstance(m, ToolMessage):
        role: Literal["system", "user", "assistant", "tool"] = "tool"
        blocks.append(ToolResultBlock(call_id=m.tool_call_id, name=m.name, content=text))
        extra["status"] = m.status
    else:
        role = {"human": "user", "ai": "assistant", "system": "system"}.get(m.type, "user")  # type: ignore[assignment]
        if text or not getattr(m, "tool_calls", None):
            blocks.append(TextBlock(text=text))
        for tc in getattr(m, "tool_calls", None) or []:
            blocks.append(ToolCallBlock(id=tc["id"] or "", name=tc["name"], args=tc["args"]))
    prov = None
    if m.additional_kwargs.get("lc_source") == "summarization":
        prov = Provenance(kind="summary")
    return Message(id=m.id or fallback_id, role=role, blocks=blocks, provenance=prov, extra=extra)


def to_neutral_list(messages: list[AnyMessage]) -> list[Message]:
    """Convert a list; messages without ids get a stable id from content and occurrence count."""
    seen: dict[str, int] = {}
    out = []
    for m in messages:
        probe = to_neutral(m, "")
        if m.id is None:
            n = seen[probe.content_hash] = seen.get(probe.content_hash, 0) + 1
            probe.id = f"auto-{probe.content_hash}-{n}"
        out.append(probe)
    return out


def from_neutral(m: Message) -> AnyMessage:
    e = m.extra
    content = e.get("content_raw", m.text if m.role != "tool" else "")
    kw = {"id": m.id, "additional_kwargs": e.get("additional_kwargs", {})}
    if m.role == "tool":
        r = next(b for b in m.blocks if isinstance(b, ToolResultBlock))
        return ToolMessage(
            content=r.content,
            tool_call_id=r.call_id,
            name=r.name,
            status=e.get("status", "success"),
            **kw,
        )
    if m.role == "assistant":
        calls = [
            {"id": b.id, "name": b.name, "args": b.args, "type": "tool_call"}
            for b in m.blocks
            if isinstance(b, ToolCallBlock)
        ]
        return AIMessage(
            content=content,
            tool_calls=calls,
            name=e.get("name"),
            response_metadata=e.get("response_metadata", {}),
            **kw,
        )
    cls = SystemMessage if m.role == "system" else HumanMessage
    return cls(content=content, name=e.get("name"), **kw)


# --- middleware -------------------------------------------------------------------------


def _warn(**meta: Any) -> None:
    with contextlib.suppress(Exception):  # logging must never break the agent either
        log.warn("rotguard hook failed", **meta)


def _failopen(fn: Callable[..., Any]) -> Callable[..., Any]:
    @functools.wraps(fn)
    def inner(self: Any, *a: Any, **k: Any) -> Any:
        try:
            return fn(self, *a, **k)
        except Exception as exc:  # never break the agent
            _warn(hook=fn.__name__, error=repr(exc))
            return None

    return inner


class RotguardMiddleware(AgentMiddleware):
    """Observe context changes, keep a ledger, and show the ledger to the model on every call.

    Place it after any compacting middleware (`SummarizationMiddleware`,
    `ContextEditingMiddleware`) in the `middleware=[...]` list so it sees their effect.
    One instance serves one session.
    """

    def __init__(
        self,
        store: SQLiteStore | str | None = None,
        session_id: str | None = None,
        inject_ledger: bool = True,
        ledger_tool: bool = True,
    ):
        super().__init__()
        self.store = store if isinstance(store, SQLiteStore) else SQLiteStore(store or ":memory:")
        self.session_id = session_id or uuid.uuid4().hex
        self.inject_ledger = inject_ledger
        self.ledger = Ledger(
            on_change=lambda e: self.store.upsert_entry(self.session_id, e.model_dump())
        )
        self._history, self._request = Observer("history"), Observer("request")
        self._turn = 0
        self._last_assistant_id: str | None = None
        self.tools = [self._make_tool()] if ledger_tool else []

    def _make_tool(self) -> Any:
        @tool("ledger_record")
        def ledger_record(
            kind: Literal["goal", "decision", "constraint", "question"],
            text: str,
            key: str | None = None,
            supersedes: str | None = None,
        ) -> str:
            """Record a goal, decision, constraint or open question so it survives context
            compaction. Give the same `key` as an earlier entry (or its id in `supersedes`) when
            this overrides it."""
            try:
                e = self.ledger.record(
                    kind,
                    text,
                    key=key,
                    supersedes=supersedes,
                    source_message_id=self._last_assistant_id,
                    turn=self._turn,
                )
            except ValueError as exc:
                return f"not recorded: {exc}"
            return f"recorded {e.id}"

        return ledger_record

    def _row(self, m: Message) -> dict[str, Any]:
        prov = m.provenance or default_provenance(m, self._turn)
        return {
            "id": m.id,
            "turn": self._turn,
            "kind": prov.kind,
            "role": m.role,
            "tool_name": prov.tool_name,
            "call_id": prov.call_id,
            "parents": prov.parents,
            "hash": m.content_hash,
            "text": m.text,
        }

    # observation
    def _observe(self, observer: Observer, messages: list[AnyMessage]) -> None:
        neutral = to_neutral_list(messages)
        obs = observer.observe(neutral, self._turn)
        if observer is self._history:
            if obs.added:
                self.store.add_items(
                    self.session_id,
                    [self._row(m) for m in obs.added],
                )
            self.store.mark_removed(self.session_id, obs.removed, self._turn)
            self.store.save_snapshot(self.session_id, self._turn, [m.id for m in neutral])
        if obs.compaction:
            self.store.add_event(self.session_id, "compaction", obs.compaction.to_dict())
            log.info(
                "compaction",
                channel=observer.channel,
                turn=self._turn,
                removed=len(obs.removed),
                rewritten=len(obs.rewritten),
            )

    @_failopen
    def before_agent(self, state: Any, runtime: Any) -> dict[str, Any] | None:
        if not self.ledger.all():
            self.ledger.load([Entry(**e) for e in self.store.entries(self.session_id)])
        return None

    @_failopen
    def before_model(self, state: Any, runtime: Any) -> dict[str, Any] | None:
        self._turn += 1
        self._observe(self._history, state["messages"])
        return None

    def _prepare(self, request: Any) -> Any:
        try:
            self._observe(self._request, request.messages)
            text = self.ledger.render() if self.inject_ledger else ""
            if text:
                note = HumanMessage(content=text, name=LEDGER_MESSAGE_NAME)
                return request.override(messages=[*request.messages, note])
        except Exception as exc:
            _warn(hook="wrap_model_call", error=repr(exc))
        return request

    def wrap_model_call(self, request: Any, handler: Callable[[Any], Any]) -> Any:
        return handler(self._prepare(request))

    async def awrap_model_call(self, request: Any, handler: Callable[[Any], Awaitable[Any]]) -> Any:
        return await handler(self._prepare(request))

    @_failopen
    def after_model(self, state: Any, runtime: Any) -> dict[str, Any] | None:
        last = state["messages"][-1]
        if isinstance(last, AIMessage):
            self._last_assistant_id = last.id
        return None

    @_failopen
    def after_agent(self, state: Any, runtime: Any) -> dict[str, Any] | None:
        self._observe(self._history, state["messages"])
        return None

    async def abefore_agent(self, state: Any, runtime: Any) -> dict[str, Any] | None:
        self.before_agent(state, runtime)
        return None

    async def abefore_model(self, state: Any, runtime: Any) -> dict[str, Any] | None:
        self.before_model(state, runtime)
        return None

    async def aafter_model(self, state: Any, runtime: Any) -> dict[str, Any] | None:
        self.after_model(state, runtime)
        return None

    async def aafter_agent(self, state: Any, runtime: Any) -> dict[str, Any] | None:
        self.after_agent(state, runtime)
        return None
