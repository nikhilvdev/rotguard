"""Framework-neutral message model. Adapters convert to and from this."""

from __future__ import annotations

import hashlib
import json
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field

Kind = Literal["user", "assistant", "tool", "system", "summary", "ledger"]


class Provenance(BaseModel):
    """Where an item in the context came from."""

    kind: Kind
    turn: int = 0
    tool_name: str | None = None
    call_id: str | None = None
    parents: list[str] = Field(default_factory=list)  # for summaries: the items replaced


class TextBlock(BaseModel):
    type: Literal["text"] = "text"
    text: str


class ToolCallBlock(BaseModel):
    type: Literal["tool_call"] = "tool_call"
    id: str
    name: str
    args: dict[str, Any] = Field(default_factory=dict)


class ToolResultBlock(BaseModel):
    type: Literal["tool_result"] = "tool_result"
    call_id: str
    name: str | None = None
    content: str


Block = Annotated[TextBlock | ToolCallBlock | ToolResultBlock, Field(discriminator="type")]


class Message(BaseModel):
    id: str
    role: Literal["system", "user", "assistant", "tool"]
    blocks: list[Block] = Field(default_factory=list)
    provenance: Provenance | None = None
    extra: dict[str, Any] = Field(default_factory=dict)  # framework fields kept for round-trips

    @property
    def text(self) -> str:
        parts = []
        for b in self.blocks:
            if isinstance(b, TextBlock):
                parts.append(b.text)
            elif isinstance(b, ToolResultBlock):
                parts.append(b.content)
        return "\n".join(parts)

    @property
    def content_hash(self) -> str:
        payload = json.dumps([b.model_dump() for b in self.blocks], sort_keys=True, default=str)
        return hashlib.sha256(f"{self.role}|{payload}".encode()).hexdigest()[:16]


def default_provenance(m: Message, turn: int) -> Provenance:
    tool_name = call_id = None
    for b in m.blocks:
        if isinstance(b, ToolResultBlock):
            tool_name, call_id = b.name, b.call_id
    return Provenance(kind=m.role, turn=turn, tool_name=tool_name, call_id=call_id)
