import json
import re
from types import SimpleNamespace

import pytest

pytest.importorskip("anthropic")

from anthropic.types.beta import BetaCompactionBlock, BetaTextBlock  # noqa: E402
from test_audit import cued_answer  # noqa: E402

from rotguard.adapters.anthropic import RotguardAnthropic, to_neutral
from rotguard.probes import AuditPolicy
from rotguard.report import blame

BETAS = ["compact-2026-01-12"]
TOOLS = [
    {"name": "grep", "description": "search", "input_schema": {"type": "object", "properties": {}}}
]


def flat(content):
    if isinstance(content, str):
        return content
    return "\n".join(
        b.get("text") or b.get("content") or "" for b in content if isinstance(b, dict)
    )


class FakeClient:
    """Stands in for anthropic.Anthropic(): knows only what is in the context a call carries."""

    def __init__(self, answers, ledger_answers=(), sources=()):
        self.answers, self.calls, self.ledger_answers = answers, [], list(ledger_answers)
        self.sources = list(sources)
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self.create))

    def create(self, **kw):
        self.calls.append(kw)
        last = flat(kw["messages"][-1]["content"])
        text = "\n".join(flat(m["content"]) for m in kw["messages"][:-1])
        if "about to be removed" in last:
            reply = json.dumps([{"question": q, "answer": a} for q, a in self.answers.items()])
        elif "Answer from the conversation above only" in last:
            lines = []
            for m in (re.match(r"(\d+)\. (.*)", ln) for ln in last.splitlines()):
                if m and m.group(2).startswith("List every goal"):
                    known = [a for a in self.ledger_answers if a in text]
                    lines.append(f"{m.group(1)}. " + ("; ".join(known) or "unknown"))
                elif m and (cued := cued_answer(m.group(2), text, self.sources)) is not None:
                    lines.append(f"{m.group(1)}. {cued}")
                elif m and m.group(2) in self.answers:
                    a = self.answers[m.group(2)]
                    lines.append(f"{m.group(1)}. {a if a in text else 'unknown'}")
            reply = "\n".join(lines)
        else:
            reply = "ok"
        return SimpleNamespace(
            content=[BetaTextBlock(type="text", text=reply)], stop_reason="end_turn"
        )


def block(content="SUMMARY: the user is auditing orders."):
    return BetaCompactionBlock(type="compaction", content=content, signature="sig-123")


def user(t):
    return {"role": "user", "content": t}


def test_effective_context_starts_at_the_last_compaction_block():
    msgs = [
        user("old 1"),
        {"role": "assistant", "content": "old 2"},
        user("old 3"),
        {"role": "assistant", "content": [block(), BetaTextBlock(type="text", text="reply after")]},
        user("new question"),
    ]
    ctx = to_neutral(msgs)
    assert [m.text for m in ctx] == [
        "SUMMARY: the user is auditing orders.",
        "reply after",
        "new question",
    ]
    assert ctx[0].provenance.kind == "summary" and ctx[0].id.startswith("compaction-")
    # the signed block is kept exactly as the API returned it, with no None fields added
    assert ctx[0].extra["raw_message"]["content"] == [
        {"type": "compaction", "content": block().content, "signature": "sig-123"}
    ]


def test_on_demand_swap_and_tool_blocks():
    msgs = [
        {"role": "assistant", "content": [block()]},
        user("continue"),
        {
            "role": "assistant",
            "content": [{"type": "tool_use", "id": "t1", "name": "grep", "input": {"q": "x"}}],
        },
        {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "found 3"}],
        },
    ]
    ctx = to_neutral(msgs)
    assert [m.provenance.kind for m in ctx] == ["summary", "user", "assistant", "tool"]
    assert ctx[3].text == "found 3"
    assert len({m.id for m in ctx}) == 4


def test_ids_are_stable_across_requests_even_for_repeated_messages():
    a = to_neutral([user("same"), user("same")])
    b = to_neutral([user("same"), user("same"), user("new")])
    assert [m.id for m in a] == [m.id for m in b][:2] and a[0].id != a[1].id


def run_conversation(audit, recall=None, on_alert=None):
    client = FakeClient(
        {"Which table may be written to?": "staging_orders"},
        ["staging_orders"],
        ["Only write to staging_orders."],
    )
    rg = RotguardAnthropic(client, audit=audit, recall=recall, on_alert=on_alert)
    if recall:
        rg.ledger.record("constraint", "Only write to staging_orders.")
    history = [
        user("Constraint: only write to staging_orders."),
        {"role": "assistant", "content": "ok"},
        user("keep going"),
        {"role": "assistant", "content": "done"},
    ]
    first = rg.beta.messages.create(
        model="claude-opus-5-5",
        max_tokens=100,
        messages=history,
        betas=BETAS,
        tools=TOOLS,
        system="be brief",
    )
    # threshold compaction: the reply starts with a lossy compaction block; the API drops what is before it
    history += [
        {"role": "assistant", "content": [block(), BetaTextBlock(type="text", text="carrying on")]},
        user("what next?"),
    ]
    rg.beta.messages.create(
        model="claude-opus-5-5",
        max_tokens=100,
        messages=history,
        betas=BETAS,
        tools=TOOLS,
        system="be brief",
    )
    return client, rg, first


def test_compaction_is_recorded_with_provenance_and_blame_finds_the_source():
    _, rg, _ = run_conversation(None)
    events = rg.store.events(rg.session_id, "compaction")
    assert len(events) == 1 and events[0]["summary_ids"][0].startswith("compaction-")
    b = blame(rg.store, rg.session_id, "staging_orders")
    assert b.origins and b.origins[0]["role"] == "user" and b.origins[0]["turn"] == 1
    assert not rg.store.events(rg.session_id, "audit")


def test_audit_reports_loss_and_side_calls_carry_tools_betas_and_the_signed_block():
    client, rg, first = run_conversation(AuditPolicy(mode="ci"))
    assert first.content[0].text == "ok"
    reports = rg.store.events(rg.session_id, "audit")
    assert reports[0]["loss"] == 1.0 and reports[0]["lost"][0]["answer"] == "staging_orders"
    quiz = [
        c
        for c in client.calls
        if "Answer from the conversation above only" in flat(c["messages"][-1]["content"])
    ]
    assert len(quiz) == 2 and all(
        c["tools"] == TOOLS and c["betas"] == BETAS and c["system"] == "be brief" for c in quiz
    )
    new_ctx = quiz[1]["messages"]
    assert new_ctx[0]["content"][0] == {
        "type": "compaction",
        "content": block().content,
        "signature": "sig-123",
    }
    main = [
        c
        for c in client.calls
        if c not in quiz and "about to be removed" not in flat(c["messages"][-1]["content"])
    ]
    assert len(main) == 2 and "context_management" not in main[0]


def test_wrapper_fails_open_and_passes_other_attributes_through():
    client = FakeClient({})
    client.other = "x"
    rg = RotguardAnthropic(client, audit=AuditPolicy(mode="ci"))
    assert rg.other == "x"
    rg.store.add_items = lambda *a, **k: 1 / 0
    resp = rg.beta.messages.create(model="m", max_tokens=5, messages=[user("hi")])
    assert resp.content[0].text == "ok"


def test_recall_probe_sees_the_constraint_until_compaction_drops_it():
    from rotguard.probes import RecallPolicy

    policy = RecallPolicy(mode="every", every=1, signals=False, background=False)
    client, rg, _ = run_conversation(None, policy)
    scores = [e["score"] for e in rg.store.events(rg.session_id, "recall")]
    assert scores == [1.0, 0.0]
    probes = [
        c
        for c in client.calls
        if "Complete the constraint currently in effect" in flat(c["messages"][-1]["content"])
    ]
    assert len(probes) == 2 and all(c["tools"] == TOOLS and c["betas"] == BETAS for c in probes)
