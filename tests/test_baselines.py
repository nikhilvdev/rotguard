import pytest

from evals import baselines  # noqa: F401  (registers the strategies)
from evals.baselines import anthropic_compaction, lc_summarize, rg_staged
from evals.core import STRATEGIES
from evals.models import COMPACTION_BETA, AnthropicClient, Metered, make_client


class Recording:
    name = "recording"

    def __init__(self):
        self.calls = []

    def complete(self, system, messages, max_tokens=400):
        self.calls.append((system, messages))
        return "SUMMARY-TEXT"


def history(n=20, tool_every=2):
    out = []
    for i in range(n):
        body = "[tool result] listing\n" + "row " * 400 if i % tool_every else f"message {i}"
        out.append({"role": "user" if i % 2 == 0 else "assistant", "content": body})
    return out


def test_all_baselines_are_registered_next_to_the_naive_ones():
    assert {"none", "truncate", "summarize"} <= set(STRATEGIES)
    assert {"lc-summarize", "anthropic-compaction", "rg-staged"} <= set(STRATEGIES)


def test_rg_staged_under_budget_does_nothing_and_clears_before_it_summarizes():
    h = history()
    c = Recording()
    assert rg_staged(budget_tokens=10**6)(h, c) == h and not c.calls
    cleared = rg_staged(budget_tokens=3300, clear_keep=3)(h, c)  # enough to clear, not to summarize
    assert not c.calls and len(cleared) == len(h)
    stubs = [m for m in cleared if "output offloaded" in m["content"]]
    assert stubs and sum(len(m["content"]) > 1500 for m in cleared) == 3  # newest 3 kept whole
    assert [m["role"] for m in cleared] == [m["role"] for m in h]
    # still over budget after clearing: one summarizer call, on the stubbed history
    out = rg_staged(budget_tokens=200, clear_keep=3)(h, c)
    assert len(c.calls) == 1 and out[0]["content"].startswith("[summary of earlier work]")
    assert "output offloaded" in c.calls[0][1][0]["content"]  # summarized the cleared history


def test_lc_summarize_runs_the_real_middleware_and_keeps_the_recent_messages():
    h = history(16, tool_every=99)
    c = Recording()
    out = lc_summarize(keep_last=4)(h, c)
    assert c.calls and len(out) == 1 + 4
    assert "SUMMARY-TEXT" in out[0]["content"] and out[-4:] == h[-4:]


class FakeBeta:
    def __init__(self):
        self.requests = []
        self.messages = self

    def create(self, **kw):
        self.requests.append(kw)
        block = type(
            "B",
            (),
            {
                "type": "compaction",
                "model_dump": lambda self: {
                    "type": "compaction",
                    "content": "S",
                    "signature": "sig",
                },
            },
        )()
        text = type("T", (), {"type": "text", "text": "ok"})()
        usage = type("U", (), {"input_tokens": 11, "output_tokens": 3})()
        return type(
            "R", (), {"content": [block if kw.get("compaction") else text], "usage": usage}
        )()


def anthropic_with_fake(monkeypatch):
    pytest.importorskip("anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    c = AnthropicClient("claude-haiku-5-5")
    fake = FakeBeta()
    c._c = type("C", (), {"beta": fake, "messages": fake})()
    return c, fake


def test_anthropic_compaction_request_shape_matches_the_sdk_types_not_the_live_api(monkeypatch):
    client, fake = anthropic_with_fake(monkeypatch)
    out = anthropic_compaction(keep_last=2)(history(6, tool_every=99), Metered(client))
    req = fake.requests[0]
    assert req["compaction"] == {"type": "summarize"} and req["betas"] == [COMPACTION_BETA]
    assert len(req["messages"]) == 4  # everything but the last two
    assert out[0] == {
        "role": "assistant",
        "content": [{"type": "compaction", "content": "S", "signature": "sig"}],
    }
    assert out[1:] == history(6, tool_every=99)[-2:]
    # a later request that carries the block is routed through the beta endpoint
    client.complete("sys", out)
    assert fake.requests[1]["betas"] == [COMPACTION_BETA] and "compaction" not in fake.requests[1]
    client.complete("sys", [{"role": "user", "content": "plain"}])
    assert "betas" not in fake.requests[2]


def test_anthropic_compaction_needs_an_anthropic_model():
    with pytest.raises(RuntimeError, match="anthropic: model"):
        anthropic_compaction()(history(8), make_client("mock"))
