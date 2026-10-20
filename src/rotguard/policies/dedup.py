"""F5b: collapse repeated identical tool outputs in the request for one model call.

Only tool results are touched, never user or assistant messages, and only the copy sent to the
model: stored history is unchanged. The first copy stays; later ones become a short pointer, with
their tool call id intact so call and result stay paired.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass


@dataclass
class DedupPolicy:
    min_chars: int = 200  # shorter outputs ("ok", "done") are left alone


def stub(first_id: str) -> str:
    return f"[rotguard: same output as {first_id}]"


def find_duplicates(
    outputs: list[tuple[str, str | None]],
    policy: DedupPolicy,
    pinned: Iterable[str] = (),
) -> dict[int, str]:
    """`outputs` is (id, tool output text or None for other messages) in request order.

    Returns, for each later copy, its index mapped to the id of the first identical output.
    Pinned messages are never replaced (they can still be the first copy).
    """
    keep = set(pinned)
    first: dict[str, str] = {}
    out: dict[int, str] = {}
    for i, (mid, text) in enumerate(outputs):
        if text is None or len(text) < policy.min_chars:
            continue
        if text in first:
            if mid not in keep:
                out[i] = first[text]
        else:
            first[text] = mid
    return out
