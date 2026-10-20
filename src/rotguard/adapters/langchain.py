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
from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    HumanMessage,
    RemoveMessage,
    SystemMessage,
    ToolMessage,
)
from langgraph.graph.message import REMOVE_ALL_MESSAGES

from rotguard.core import Message, Provenance, TextBlock, ToolCallBlock, ToolResultBlock
from rotguard.ledger import Entry, Ledger
from rotguard.log import get_logger
from rotguard.policies import DedupPolicy, OffloadPolicy, StagedCleanup
from rotguard.policies.dedup import find_duplicates
from rotguard.policies.dedup import stub as dedup_stub
from rotguard.policies.offload import (
    OFFLOADED,
    make_stub,
    needs_offload,
    offload_id,
    read_page,
    store_output,
)
from rotguard.policies.staged import SUMMARY_ID_PREFIX, run_staged
from rotguard.probes import AuditPolicy, LedgerExtractor, RecallPolicy, RecallResult
from rotguard.session import Session
from rotguard.store import Store, open_store
from rotguard.tokens import TokenCounter

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
    ledger: Ledger,
    get_turn: Callable[[], int],
    get_source: Callable[[], str | None],
    on_pin: Callable[[str], None] | None = None,
) -> Any:
    """The `ledger_record` tool an agent calls to keep a goal, decision or constraint alive."""

    @tool("ledger_record")
    def ledger_record(
        kind: Literal["goal", "decision", "constraint", "question"],
        text: str,
        key: str | None = None,
        supersedes: str | None = None,
        pin: bool = False,
    ) -> str:
        """Record a goal, decision, constraint or open question so it survives context
        compaction. Give the same `key` as an earlier entry (or its id in `supersedes`) when
        this overrides it. Set `pin` to also keep the message that recorded it word for word
        through cleanup."""
        try:
            e = ledger.record(
                kind,
                text,
                key=key,
                supersedes=supersedes,
                source_message_id=get_source(),
                turn=get_turn(),
                origin="tool",
            )
        except ValueError as exc:
            return f"not recorded: {exc}"
        if pin and on_pin and (src := get_source()):
            on_pin(src)
        return f"recorded {e.id}"

    return ledger_record


def make_recall_output_tool(store: Store, session_id: str) -> Any:
    """The `recall_output` tool: read a tool output that was offloaded, in pages."""

    @tool("recall_output")
    def recall_output(id: str, start_line: int = 0, max_lines: int = 200) -> str:
        """Read a tool output that was moved out of the context to keep it small. Pass the id from
        the stub that replaced it, and page through long outputs with `start_line`."""
        row = store.item(session_id, id)
        if row is None or row["kind"] != OFFLOADED:
            return f"no offloaded output with id {id!r}"
        return read_page(row["text"], start_line, max_lines)

    return recall_output


def model_summarizer(model: Any) -> Callable[[list[Message]], str]:
    """A `StagedCleanup.summarizer` that asks a chat model to summarize the messages."""

    def summarize(messages: list[Message]) -> str:
        transcript = "\n".join(f"{m.role}: {m.text[:2000]}" for m in messages)
        reply = model.invoke(
            [
                HumanMessage(
                    content="Summarize this agent transcript so the work can continue from the "
                    "summary alone. Keep goals, decisions, constraints, names, numbers and "
                    "identifiers.\n\n<transcript>\n" + transcript + "\n</transcript>"
                )
            ]
        )
        return str(reply.text)

    return summarize


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
        store: Store | str | None = None,
        session_id: str | None = None,
        inject_ledger: bool = True,
        ledger_tool: bool = True,
        audit: AuditPolicy | None = None,
        recall: RecallPolicy | None = None,
        on_alert: Callable[[RecallResult], None] | None = None,
        offload: OffloadPolicy | None = None,
        dedup: DedupPolicy | None = None,
        staged: StagedCleanup | None = None,
        counter: TokenCounter | None = None,
        extract: LedgerExtractor | None = None,
    ):
        super().__init__()
        self.store = open_store(store)
        self.session_id = session_id or uuid.uuid4().hex
        self.inject_ledger = inject_ledger
        self.ledger = Ledger(on_change=self._save_entry)
        self.session = Session(self.store, self.session_id, self.ledger, audit, recall, on_alert)
        self._last_assistant_id: str | None = None
        self.offload, self.dedup, self.staged, self.extract = offload, dedup, staged, extract
        self.counter = counter or TokenCounter()
        self._warned: set[str] = set()
        self.tools: list[Any] = (
            [
                make_ledger_tool(
                    self.ledger,
                    lambda: self.session.turn,
                    lambda: self._last_assistant_id,
                    self.session.pin,
                )
            ]
            if ledger_tool
            else []
        )
        if offload or staged:  # both leave stubs that point at recall_output
            self.tools.append(make_recall_output_tool(self.store, self.session_id))

    def pin(self, message_id: str) -> None:
        """Keep this message through offload, dedup, staged cleanup and handoff."""
        self.session.pin(message_id)

    def _warn_once(self, key: str, msg: str) -> None:
        if key not in self._warned:
            self._warned.add(key)
            _warn(hook=key, error=msg)

    @_failopen
    def before_agent(self, state: Any, runtime: Any) -> dict[str, Any] | None:
        if not self.ledger.all():
            self.ledger.load([Entry(**e) for e in self.store.entries(self.session_id)])
        return None

    @_failopen
    def before_model(self, state: Any, runtime: Any) -> dict[str, Any] | None:
        self.session.turn += 1
        neutral = to_neutral_list(state["messages"])
        self.session.observe_history(neutral)
        fresh = self.session.take_handoff(neutral)  # RecallPolicy(action="handoff") and an alert
        if fresh is None:
            fresh = self._staged_cleanup(neutral)
        if fresh is not None:
            return {
                "messages": [
                    RemoveMessage(id=REMOVE_ALL_MESSAGES),
                    *[from_neutral(m) for m in fresh],
                ]
            }
        return None

    def _staged_cleanup(self, neutral: list[Message]) -> list[Message] | None:
        """F5d, and `action="compact"`: the cleaned message list, or None to leave things alone."""
        forced = self.session.compact_pending
        self.session.compact_pending = False  # one-shot, like the alert that asked for it
        if self.staged is None:
            if forced:
                self._warn_once("compact", 'action="compact" needs a StagedCleanup; logging only')
            return None
        # Another summarizer is active (SummarizationMiddleware, Anthropic compaction): two
        # summarizers would fight over the same history, so only stage 1 runs.
        other = any(
            m.provenance
            and m.provenance.kind == "summary"
            and not m.id.startswith(SUMMARY_ID_PREFIX)
            for m in neutral
        )
        if other:
            self._warn_once("staged", "another summarizer is active: staged cleanup stage 2 is off")
        res = run_staged(
            neutral,
            self.staged,
            self.counter,
            self.session.pinned,
            force=forced,
            allow_summary=not other,
        )
        if res is None:
            return None
        for o in res.offloaded:
            store_output(
                self.store,
                self.session_id,
                o.item_id,
                o.text,
                turn=self.session.turn,
                tool_name=o.tool_name,
                call_id=o.call_id,
            )
        self.session.event(
            "staged",
            {
                "turn": self.session.turn,
                "stages": res.stages,
                "cleared": res.cleared,
                "summarized": res.summarized,
                "tokens_before": res.tokens_before,
                "tokens_after": res.tokens_after,
                "tokens_estimated": self.counter.estimated,
                "forced": forced,
            },
        )
        return res.messages

    def _offload(self, request: Any, result: Any) -> Any:
        """F5a: store a large tool output and return a stub with the same tool_call_id."""
        if (
            self.offload is None
            or not isinstance(result, ToolMessage)
            or not isinstance(result.content, str)
            or not needs_offload(result.content, self.offload)
        ):
            return result
        call = request.tool_call
        item_id = offload_id(result.tool_call_id)
        store_output(
            self.store,
            self.session_id,
            item_id,
            result.content,
            turn=self.session.turn,
            tool_name=call.get("name"),
            call_id=result.tool_call_id,
        )
        return result.model_copy(
            update={"content": make_stub(item_id, result.content, self.offload)}
        )

    def wrap_tool_call(self, request: Any, handler: Callable[[Any], Any]) -> Any:
        result = handler(request)
        try:
            return self._offload(request, result)
        except Exception as exc:
            _warn(hook="wrap_tool_call", error=repr(exc))
            return result

    async def awrap_tool_call(self, request: Any, handler: Callable[[Any], Awaitable[Any]]) -> Any:
        result = await handler(request)
        try:
            return self._offload(request, result)
        except Exception as exc:
            _warn(hook="awrap_tool_call", error=repr(exc))
            return result

    def _dedup(self, messages: list[Any]) -> list[Any]:
        """F5b, for this call only: later copies of an identical tool output become a pointer."""
        if self.dedup is None:
            return messages
        outs = [
            (
                m.id or f"#{i}",
                m.content if isinstance(m, ToolMessage) and isinstance(m.content, str) else None,
            )
            for i, m in enumerate(messages)
        ]
        dups = find_duplicates(outs, self.dedup, self.session.pinned)
        return [
            m.model_copy(update={"content": dedup_stub(dups[i])}) if i in dups else m
            for i, m in enumerate(messages)
        ]

    def _pinned_note(self, messages: list[Any]) -> str:
        """Pinned messages another tool compacted out of this request, for this call only."""
        present = {m.id for m in messages}
        parts = []
        for pid in sorted(self.session.pinned - present):
            row = self.store.item(self.session_id, pid)
            if row and row["kind"] != OFFLOADED:
                parts.append(f"[rotguard pinned message {pid} ({row['kind']})]:\n{row['text']}")
        return "\n".join(parts)

    def _extract_entries(self, messages: list[Any]) -> None:
        """F1x: propose ledger entries from the last user message and the reply just given."""
        if self.extract is None or not self.extract.due(self.session.turn):
            return
        try:
            last_user = next(m for m in reversed(messages[:-1]) if isinstance(m, HumanMessage))
            self.extract.run(
                to_neutral_list([last_user, messages[-1]]),
                self.ledger,
                turn=self.session.turn,
                source_message_id=messages[-1].id,
            )
        except StopIteration:
            return
        except Exception as exc:  # an extractor must never break the agent
            _warn(hook="extract", error=repr(exc))

    def _save_entry(self, entry: Entry) -> None:
        self.store.upsert_entry(self.session_id, entry.model_dump())

    def _prepare(self, request: Any) -> Any:
        try:
            messages = self._dedup(list(request.messages))
            self.session.observe_request(to_neutral_list(messages))
            text = self.ledger.render() if self.inject_ledger else ""
            parts = (text, self.session.marker_note(), self._pinned_note(messages))
            shown = "\n".join(t for t in parts if t)
            self.session.run_pending(
                LangChainCaller(request.model, request.tools, request.system_message), text, shown
            )
            changed = any(a is not b for a, b in zip(messages, request.messages, strict=True))
            if shown:
                note = HumanMessage(content=shown, name=LEDGER_MESSAGE_NAME)
                return request.override(messages=[*messages, note])
            if changed:
                return request.override(messages=messages)
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
            self.session.observe_reply(last.text)
            self._extract_entries(state["messages"])
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
        result: dict[str, Any] | None = self.before_model(state, runtime)
        return result

    async def aafter_model(self, state: Any, runtime: Any) -> dict[str, Any] | None:
        self.after_model(state, runtime)
        return None

    async def aafter_agent(self, state: Any, runtime: Any) -> dict[str, Any] | None:
        self.after_agent(state, runtime)
        return None
