"""F5d: staged cleanup. Clear old tool outputs first; summarize only if still over budget.

LangChain's built-in middleware cannot express "clear first, then summarize only if needed".
Stage 1 offloads every tool result except the newest `clear_keep` (the full text goes to the store,
so nothing is lost). Stage 2 summarizes everything before the last `keep_messages` messages,
except system messages and pinned ones. The result is a new message list for the caller to put in
place of the old one. Tool calls and results stay paired throughout.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass, field

from rotguard.core import Message, Provenance, TextBlock, ToolResultBlock
from rotguard.handoff import pinned_closure, tail_start
from rotguard.tokens import TokenCounter

from .offload import OffloadPolicy, is_stub, make_stub, offload_id

SUMMARY_ID_PREFIX = "staged-summary-"


@dataclass
class StagedCleanup:
    budget_tokens: int
    clear_keep: int = 3  # newest tool results left whole by stage 1
    summarizer: Callable[[list[Message]], str] | None = None  # None: stage 2 is off
    keep_messages: int = 6  # newest messages stage 2 leaves whole
    offload: OffloadPolicy = field(default_factory=OffloadPolicy)  # shapes the stage 1 stubs


@dataclass
class Offloaded:
    item_id: str
    text: str
    tool_name: str | None
    call_id: str


@dataclass
class StagedResult:
    messages: list[Message]
    stages: list[str]
    cleared: int
    summarized: int
    tokens_before: int
    tokens_after: int
    offloaded: list[Offloaded]


def run_staged(
    messages: list[Message],
    cfg: StagedCleanup,
    counter: TokenCounter,
    pinned: set[str] | frozenset[str] = frozenset(),
    *,
    force: bool = False,
    allow_summary: bool = True,
) -> StagedResult | None:
    """The cleaned message list, or None when under budget (or nothing could be cleaned).

    `force` cleans even when under budget, and runs stage 2 whatever stage 1 achieved.
    """
    before = counter.count(messages)
    if before <= cfg.budget_tokens and not force:
        return None
    cur = list(messages)
    stages: list[str] = []
    offloaded: list[Offloaded] = []

    tool_idx = [i for i, m in enumerate(cur) if m.role == "tool"]
    for i in tool_idx[: len(tool_idx) - cfg.clear_keep] if cfg.clear_keep > 0 else tool_idx:
        m = cur[i]
        r = next((b for b in m.blocks if isinstance(b, ToolResultBlock)), None)
        if m.id in pinned or r is None or not r.content or is_stub(r.content):
            continue
        item_id = offload_id(r.call_id)
        offloaded.append(Offloaded(item_id, r.content, r.name, r.call_id))
        stub = r.model_copy(update={"content": make_stub(item_id, r.content, cfg.offload)})
        cur[i] = m.model_copy(update={"blocks": [stub]})
    if offloaded:
        stages.append("clear")
    summarized = 0

    if (force or counter.count(cur) > cfg.budget_tokens) and allow_summary and cfg.summarizer:
        system = [m for m in cur if m.role == "system"]
        body = [m for m in cur if m.role != "system"]
        start = tail_start(body, cfg.keep_messages, user_first=False)  # the summary leads
        kept = pinned_closure(body[:start], pinned)
        covered = [m for m in body[:start] if m.id not in kept]
        if covered:
            sid = (
                SUMMARY_ID_PREFIX
                + hashlib.sha256("|".join(m.id for m in covered).encode()).hexdigest()[:8]
            )
            summary = Message(
                id=sid,
                role="user",
                blocks=[TextBlock(text="[summary of earlier work]\n" + cfg.summarizer(covered))],
                provenance=Provenance(kind="summary", parents=[m.id for m in covered]),
                extra={"additional_kwargs": {"lc_source": "summarization"}},
            )
            cur = [*system, summary, *[m for m in body[:start] if m.id in kept], *body[start:]]
            stages.append("summarize")
            summarized = len(covered)

    if not stages:
        return None
    return StagedResult(
        cur, stages, len(offloaded), summarized, before, counter.count(cur), offloaded
    )
