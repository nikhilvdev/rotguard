"""Minimal model clients for the eval harness. Text in, text out."""

from __future__ import annotations

import contextlib
import os
from collections.abc import Iterator
from typing import Any, Protocol

Message = dict  # {"role": "user" | "assistant", "content": str, or a list of API blocks}
COMPACTION_BETA = "compact-2026-09-04"  # Anthropic on-demand compaction (SDK type definitions)


class Client(Protocol):
    name: str

    def complete(self, system: str, messages: list[Message], max_tokens: int = 400) -> str: ...


def openai_key() -> str:
    """The key for hosted OpenAI runs.

    `ROTGUARD_OPENAI_API_KEY` wins over `OPENAI_API_KEY`, because the latter is often set to a
    placeholder (for example "ollama") to point tools at a local server.
    """
    key = os.environ.get("ROTGUARD_OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY") or ""
    if not key.startswith("sk-") or len(key) < 30:
        raise RuntimeError(
            "no real OpenAI key found: set ROTGUARD_OPENAI_API_KEY in your environment "
            "(OPENAI_API_KEY is unset or looks like a placeholder)"
        )
    return key


class OpenAIClient:
    def __init__(self, model: str = "gpt-4.1-mini"):
        from openai import OpenAI

        self.name = f"openai:{model}"
        self._model = model
        self._c = OpenAI(api_key=openai_key())

    def complete(self, system, messages, max_tokens=400):
        r = self._c.chat.completions.create(
            model=self._model,
            max_completion_tokens=max_tokens,
            messages=[{"role": "system", "content": system}, *messages],
        )
        u = getattr(r, "usage", None)
        self.last_usage = (u.prompt_tokens, u.completion_tokens) if u else None
        return r.choices[0].message.content or ""


class AnthropicClient:
    def __init__(self, model: str = "claude-haiku-5-5"):
        import anthropic

        self.name = f"anthropic:{model}"
        self._model = model
        self._c = anthropic.Anthropic()

    def complete(self, system, messages, max_tokens=400):
        kw = {
            "model": self._model,
            "max_tokens": max_tokens,
            "system": system,
            "messages": messages,
        }
        if any(isinstance(m["content"], list) for m in messages):  # holds a compaction block
            r = self._c.beta.messages.create(betas=[COMPACTION_BETA], **kw)
        else:
            r = self._c.messages.create(**kw)
        self.last_usage = (r.usage.input_tokens, r.usage.output_tokens)
        return "".join(b.text for b in r.content if b.type == "text")

    def compact(self, system, messages):
        """On-demand compaction: the signed block that stands in for `messages` (as returned)."""
        r = self._c.beta.messages.create(
            model=self._model,
            max_tokens=1024,
            betas=[COMPACTION_BETA],
            compaction={"type": "summarize"},
            system=system,
            messages=messages,
        )
        self.last_usage = (r.usage.input_tokens, r.usage.output_tokens)
        return next(b for b in r.content if b.type == "compaction").model_dump()


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
        self.last_usage = (data.get("prompt_eval_count", 0), data.get("eval_count", 0))
        return data["message"]["content"]


class MockClient:
    """Deterministic stand-in: recalls whatever literal text is present in the context."""

    name = "mock"

    def complete(self, system, messages, max_tokens=400):
        return "\n".join(m["content"] for m in messages)[-2000:]


PURPOSES = ("agent", "summarizer", "audit", "probe")


class Metered:
    """Wraps a client and counts calls and tokens per purpose.

    Tokens come from the provider's usage fields when the client reports them (`last_usage`);
    otherwise they are estimated as characters / 4 and counted under `estimated_calls`.
    """

    def __init__(self, inner: Client):
        self.inner, self.name = inner, inner.name
        self.tally: dict[str, dict[str, int]] = {}
        self._purpose = "agent"

    @contextlib.contextmanager
    def purpose(self, name: str) -> Iterator[None]:
        before, self._purpose = self._purpose, name
        try:
            yield
        finally:
            self._purpose = before

    def complete(self, system: str, messages: list[Message], max_tokens: int = 400) -> str:
        self.inner.__dict__.pop("last_usage", None)
        out = self.inner.complete(system, messages, max_tokens)
        usage = getattr(self.inner, "last_usage", None)
        t = self.tally.setdefault(
            self._purpose, {"calls": 0, "input": 0, "output": 0, "estimated_calls": 0}
        )
        t["calls"] += 1
        if usage is None:
            usage = (
                len(system) // 4 + sum(len(str(m["content"])) for m in messages) // 4,
                len(out) // 4,
            )
            t["estimated_calls"] += 1
        t["input"] += usage[0]
        t["output"] += usage[1]
        return out

    def totals(self) -> dict[str, Any]:
        total = {k: sum(t[k] for t in self.tally.values()) for k in ("calls", "input", "output")}
        agent = self.tally.get("agent", {"input": 0, "output": 0})
        side = {
            k: sum(self.tally.get(p, {"input": 0, "output": 0})[k] for p in ("audit", "probe"))
            for k in ("input", "output")
        }
        base = agent["input"] + agent["output"]
        return {
            "model": self.name,
            "by_purpose": self.tally,
            "total": total,
            "side_call_tokens_pct_of_agent": round(100 * (side["input"] + side["output"]) / base, 1)
            if base
            else None,
            "note": "tokens are provider-reported unless estimated_calls > 0 (chars / 4)",
        }


def purpose(client: Any, name: str) -> Any:
    """`with purpose(client, "audit"):` tags calls on a Metered client; a no-op on any other."""
    fn = getattr(client, "purpose", None)
    return fn(name) if fn else contextlib.nullcontext()


def make_client(spec: str) -> Metered:
    kind, _, model = spec.partition(":")
    if kind == "mock":
        return Metered(MockClient())
    if kind == "ollama":
        return Metered(OllamaClient(model))
    if kind == "openai":
        return Metered(OpenAIClient(model or "gpt-4.1-mini"))
    if kind == "anthropic":
        return Metered(AnthropicClient(model or "claude-haiku-5-5"))
    raise ValueError(f"unknown client spec: {spec}")


def print_estimate(client: Metered, what: str) -> None:
    """What a dry run on the mock model says a real run would cost, before anything is spent."""
    t = client.totals()
    print(f"ESTIMATE for {what} (dry run on the mock model; nothing was spent):")
    for p in PURPOSES:
        if p in client.tally:
            c = client.tally[p]
            print(f"  {p:11} {c['calls']:6} calls  ~{c['input']:>10,} in  ~{c['output']:>8,} out")
    n = t["total"]["calls"]
    print(f"  {'total':11} {n:6} calls  (tokens are chars/4; real tokenizers differ)")
    print("  The mock model finds fewer facts than a real one, so audit and probe counts are")
    print("  lower bounds: budget for more.")
