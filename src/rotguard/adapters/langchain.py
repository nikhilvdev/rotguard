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

from rotguard.core import Message, Provenance, TextBlock, ToolCallBlock, ToolResultBlock
from rotguard.ledger import Entry, Ledger
from rotguard.log import get_logger
from rotguard.probes import AuditPolicy
from rotguard.session import Session
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


class LangChainCaller:
    """Side calls on the agent's own model, with its tools and system prompt, so the history that
    contains tool calls stays valid for providers that require tools to be declared."""

    def __init__(self, model: Any, tools: list[Any] | None = None, system: Any = None):
        self.model, self.tools, self.system = model, tools or [], system

    def ask(self, context: list[Message], question: str) -> str:
        messages: list[AnyMessage] = [
            *([self.system] if self.system else []),
            *[from_neutral(m) for m in context],
            HumanMessage(content=question),
        ]
        model = self.model
        if self.tools:
            with contextlib.suppress(Exception):  # a model that can't bind tools is asked as it is
                model = model.bind_tools(self.tools)
        return _text(model.invoke(messages).content)[0]


def make_ledger_tool(
    ledger: Ledger, get_turn: Callable[[], int], get_source: Callable[[], str | None]
) -> Any:
    """The `ledger_record` tool an agent calls to keep a goal, decision or constraint alive."""

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
            e = ledger.record(
                kind,
                text,
                key=key,
                supersedes=supersedes,
                source_message_id=get_source(),
                turn=get_turn(),
            )
        except ValueError as exc:
            return f"not recorded: {exc}"
        return f"recorded {e.id}"

    return ledger_record


class RotguardMiddleware(AgentMiddleware):
    """Observe context changes, keep a ledger, show it to the model on every call, and optionally
    audit what each compaction lost.

    Place it after any compacting middleware (`SummarizationMiddleware`,
    `ContextEditingMiddleware`) in the `middleware=[...]` list so it sees their effect.
    One instance serves one session. Audits are off unless you pass an `AuditPolicy`, because each
    one costs about three extra model calls.
    """

    def __init__(
        self,
        store: SQLiteStore | str | None = None,
        session_id: str | None = None,
        inject_ledger: bool = True,
        ledger_tool: bool = True,
        audit: AuditPolicy | None = None,
    ):
        super().__init__()
        self.store = store if isinstance(store, SQLiteStore) else SQLiteStore(store or ":memory:")
        self.session_id = session_id or uuid.uuid4().hex
        self.inject_ledger = inject_ledger
        self.ledger = Ledger(on_change=self._save_entry)
        self.session = Session(self.store, self.session_id, self.ledger, audit)
        self._last_assistant_id: str | None = None
        self.tools = (
            [
                make_ledger_tool(
                    self.ledger, lambda: self.session.turn, lambda: self._last_assistant_id
                )
            ]
            if ledger_tool
            else []
        )

    @_failopen
    def before_agent(self, state: Any, runtime: Any) -> dict[str, Any] | None:
        if not self.ledger.all():
            self.ledger.load([Entry(**e) for e in self.store.entries(self.session_id)])
        return None

    @_failopen
    def before_model(self, state: Any, runtime: Any) -> dict[str, Any] | None:
        self.session.turn += 1
        self.session.observe_history(to_neutral_list(state["messages"]))
        return None

    def _save_entry(self, entry: Entry) -> None:
        self.store.upsert_entry(self.session_id, entry.model_dump())

    def _prepare(self, request: Any) -> Any:
        try:
            self.session.observe_request(to_neutral_list(request.messages))
            text = self.ledger.render() if self.inject_ledger else ""
            self.session.run_pending(
                LangChainCaller(request.model, request.tools, request.system_message), text
            )
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
        self.session.observe_history(to_neutral_list(state["messages"]))
        self.session.flush()
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
