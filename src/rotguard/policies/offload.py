"""F5a: move large tool outputs into the store and leave a short stub in the context.

The stub keeps the tool call's id, so call and result stay paired. The full text is stored as an
item of kind "offloaded" and nothing is deleted: `recall_output(id)` reads it back in pages.
Offloading rewrites stored history, so it is opt-in.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from rotguard.store import Store
from rotguard.tokens import estimate

OFFLOADED = "offloaded"
STUB_MARK = "[rotguard: output offloaded"


@dataclass
class OffloadPolicy:
    max_tokens: int = 2000  # a tool output above this many tokens (estimated) is offloaded
    preview_lines: int = 10  # lines of the output kept in the stub
    preview_chars: int = 1000  # and at most this many characters, for outputs with huge lines


def offload_id(call_id: str) -> str:
    return f"offload:{call_id}"


def is_stub(text: str) -> bool:
    return STUB_MARK in text[-400:]


def needs_offload(text: str, policy: OffloadPolicy) -> bool:
    return not is_stub(text) and estimate(text) > policy.max_tokens


def make_stub(item_id: str, text: str, policy: OffloadPolicy) -> str:
    lines = text.splitlines()
    head = "\n".join(lines[: policy.preview_lines])
    if len(head) > policy.preview_chars:
        head = head[: policy.preview_chars] + "…"
    return (
        f"{head}\n{STUB_MARK}: {len(lines)} lines, about {estimate(text)} tokens, stored as "
        f"{item_id}. Call recall_output('{item_id}') to read the rest.]"
    )


def store_output(
    store: Store,
    session: str,
    item_id: str,
    text: str,
    *,
    turn: int = 0,
    tool_name: str | None = None,
    call_id: str | None = None,
) -> None:
    """Keep the full text. Repeating it for the same id changes nothing."""
    store.add_items(
        session,
        [
            {
                "id": item_id,
                "turn": turn,
                "kind": OFFLOADED,
                "role": "tool",
                "tool_name": tool_name,
                "call_id": call_id,
                "parents": [],
                "hash": hashlib.sha256(text.encode()).hexdigest()[:16],
                "text": text,
            }
        ],
    )


def read_page(text: str, start_line: int = 0, max_lines: int = 200) -> str:
    """`max_lines` lines of `text` from `start_line`, and where the next page starts."""
    lines = text.splitlines()
    start = min(max(0, start_line), len(lines))
    end = min(len(lines), start + max(1, max_lines))
    body = "\n".join(lines[start:end])
    if end < len(lines):
        return f"{body}\n[lines {start}-{end - 1} of {len(lines)}; next: start_line={end}]"
    return f"{body}\n[lines {start}-{max(start, end - 1)} of {len(lines)}: end]"
