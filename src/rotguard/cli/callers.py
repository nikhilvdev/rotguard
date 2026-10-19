"""Side callers for the CLI, built from `--model provider:name`.

Providers are imported on use, so the base install needs none of them:
`anthropic:<model>` (the anthropic package), `openai:<model>` (openai), `ollama:<model>` (openai
package pointed at a local Ollama server).
"""

from __future__ import annotations

from typing import Any

from rotguard.core import Message
from rotguard.probes import SideCaller


def chat_messages(context: list[Message], question: str) -> list[dict[str, str]]:
    """Alternating user/assistant turns that end with `question`, as providers require."""
    turns = [(m.role, m.text) for m in context] + [("user", question)]
    out: list[dict[str, str]] = []
    for role, text in turns:
        role = "assistant" if role == "assistant" else "user"
        if out and out[-1]["role"] == role:
            out[-1]["content"] += "\n\n" + text
        else:
            out.append({"role": role, "content": text})
    if out[0]["role"] != "user":
        out.insert(0, {"role": "user", "content": "(earlier context omitted)"})
    return out


class _Anthropic:
    def __init__(self, model: str):
        import anthropic

        self.client, self.model = anthropic.Anthropic(), model

    def ask(self, context: list[Message], question: str) -> str:
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            messages=chat_messages(context, question),  # type: ignore[arg-type]
        )
        return "".join(getattr(b, "text", "") for b in resp.content)


class _OpenAI:
    def __init__(self, model: str, base_url: str | None = None, api_key: str | None = None):
        import openai

        kw: dict[str, Any] = {k: v for k, v in (("base_url", base_url), ("api_key", api_key)) if v}
        self.client, self.model = openai.OpenAI(**kw), model

    def ask(self, context: list[Message], question: str) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=chat_messages(context, question),  # type: ignore[arg-type]
        )
        return resp.choices[0].message.content or ""


def make_caller(spec: str) -> SideCaller:
    provider, _, model = spec.partition(":")
    if not model:
        raise ValueError(f"--model must look like provider:name, got {spec!r}")
    if provider == "anthropic":
        return _Anthropic(model)
    if provider == "openai":
        return _OpenAI(model)
    if provider == "ollama":
        return _OpenAI(model, "http://localhost:11434/v1", "ollama")
    raise ValueError(f"unknown provider {provider!r}: use anthropic, openai or ollama")
