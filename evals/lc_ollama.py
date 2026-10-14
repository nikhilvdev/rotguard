"""A minimal LangChain chat model over the local Ollama client, for end-to-end tests with a real
model. Text only: no tool calling."""

from __future__ import annotations

from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from .core import SYSTEM
from .models import OllamaClient


def _text(content: Any) -> str:
    if isinstance(content, str):
        return content
    return "".join(b if isinstance(b, str) else b.get("text", "") for b in content)


class OllamaChat(BaseChatModel):
    model: str = "qwen2.5:3b"
    num_ctx: int = 16384
    calls: int = 0

    @property
    def _llm_type(self) -> str:
        return "ollama-text"

    def _generate(self, messages, stop=None, run_manager=None, **kw) -> ChatResult:
        system = "\n".join(_text(m.content) for m in messages if m.type == "system") or SYSTEM
        turns = [
            {"role": "assistant" if m.type == "ai" else "user", "content": _text(m.content)}
            for m in messages
            if m.type != "system"
        ]
        self.calls += 1
        text = OllamaClient(self.model, self.num_ctx).complete(system, turns, max_tokens=300)
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=text))])

    def bind_tools(self, tools, **kw):
        return self
