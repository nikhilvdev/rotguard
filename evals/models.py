"""Minimal model clients for the eval harness. Text in, text out."""

from __future__ import annotations

from typing import Protocol

Message = dict  # {"role": "user" | "assistant", "content": str}


class Client(Protocol):
    name: str

    def complete(self, system: str, messages: list[Message], max_tokens: int = 400) -> str: ...


class OpenAIClient:
    def __init__(self, model: str = "gpt-4.1-mini"):
        from openai import OpenAI

        self.name = f"openai:{model}"
        self._model = model
        self._c = OpenAI()

    def complete(self, system, messages, max_tokens=400):
        r = self._c.chat.completions.create(
            model=self._model,
            max_completion_tokens=max_tokens,
            messages=[{"role": "system", "content": system}, *messages],
        )
        return r.choices[0].message.content or ""


class AnthropicClient:
    def __init__(self, model: str = "claude-haiku-5-5"):
        import anthropic

        self.name = f"anthropic:{model}"
        self._model = model
        self._c = anthropic.Anthropic()

    def complete(self, system, messages, max_tokens=400):
        r = self._c.messages.create(
            model=self._model, max_tokens=max_tokens, system=system, messages=messages
        )
        return "".join(b.text for b in r.content if b.type == "text")


class OllamaClient:
    """Local model via Ollama's native API, so num_ctx can be set (the OpenAI-compatible
    endpoint can't, and Ollama's default window would silently truncate long contexts)."""

    def __init__(self, model: str, num_ctx: int = 32768, host: str = "http://localhost:11434"):
        self.name = f"ollama:{model}"
        self._model, self._num_ctx, self._host = model, num_ctx, host

    def complete(self, system, messages, max_tokens=400):
        import json
        import urllib.request

        body = {
            "model": self._model,
            "stream": False,
            "messages": [{"role": "system", "content": system}, *messages],
            "options": {"num_ctx": self._num_ctx, "num_predict": max_tokens, "temperature": 0},
        }
        req = urllib.request.Request(
            f"{self._host}/api/chat",
            json.dumps(body).encode(),
            {"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=1800) as r:
            data = json.load(r)
        if data.get("prompt_eval_count", 0) >= self._num_ctx - max_tokens:
            raise RuntimeError(f"context hit num_ctx={self._num_ctx}; result would be truncated")
        return data["message"]["content"]


class MockClient:
    """Deterministic stand-in: recalls whatever literal text is present in the context."""

    name = "mock"

    def complete(self, system, messages, max_tokens=400):
        return "\n".join(m["content"] for m in messages)[-2000:]


def make_client(spec: str) -> Client:
    kind, _, model = spec.partition(":")
    if kind == "mock":
        return MockClient()
    if kind == "ollama":
        return OllamaClient(model)
    if kind == "openai":
        return OpenAIClient(model or "gpt-4.1-mini")
    if kind == "anthropic":
        return AnthropicClient(model or "claude-haiku-5-5")
    raise ValueError(f"unknown client spec: {spec}")
