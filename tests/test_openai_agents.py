import asyncio

from test_audit import FakeCaller

from rotguard.adapters.openai_agents import RotguardSession, to_neutral
from rotguard.probes import AuditPolicy, RecallPolicy


class TrimmingInner:
    """Stands in for a trimming session: keeps only the last `keep` items."""

    session_id = "inner"

    def __init__(self, keep=4):
        self.items, self.keep = [], keep

    async def get_items(self, limit=None):
        return list(self.items)

    async def add_items(self, items):
        self.items = [*self.items, *items][-self.keep :]

    async def pop_item(self):
        return self.items.pop() if self.items else None

    async def clear_items(self):
        self.items = []


class FakeResponses:
    def __init__(self, caller):
        self.caller, self.calls = caller, []

    def create(self, model, input, instructions=None):
        *items, question = input
        self.calls.append(question["content"])
        reply = self.caller.ask(to_neutral(items), question["content"])
        return type("R", (), {"output_text": reply})


class FakeClient:
    def __init__(self, caller):
        self.responses = FakeResponses(caller)


def run(session, n):
    async def go():
        await session.add_items(
            [{"role": "user", "content": "Constraint: only use staging_orders."}]
        )
        for i in range(n):
            await session.get_items()
            await session.add_items(
                [
                    {"role": "assistant", "content": f"working on step {i}"},
                    {"role": "user", "content": f"follow-up {i}"},
                ]
            )
        await session.get_items()

    asyncio.run(go())


def test_items_convert_to_neutral_messages_with_stable_ids_and_kinds():
    items = [
        {"role": "user", "content": "hi"},
        {"type": "function_call", "name": "ls", "arguments": "{}", "call_id": "c1"},
        {"type": "function_call_output", "call_id": "c1", "output": "a.txt"},
        {"role": "assistant", "content": [{"type": "output_text", "text": "done"}]},
    ]
    msgs = to_neutral(items)
    assert [m.role for m in msgs] == ["user", "assistant", "user", "assistant"]
    assert [m.provenance.kind for m in msgs] == ["user", "assistant", "tool", "assistant"]
    assert msgs[3].text == "done" and msgs[1].text == "ls({})"
    assert [m.id for m in to_neutral(items)] == [m.id for m in msgs]  # stable across calls
    assert len({m.id for m in msgs}) == 4


def test_audit_and_recall_see_a_trim_done_by_the_inner_session():
    alerts = []
    caller = FakeCaller(
        [{"question": "Which table may be written to?", "answer": "staging_orders"}],
        {"Which table may be written to?": "staging_orders"},
        ["staging_orders"],
    )
    client = FakeClient(caller)
    s = RotguardSession(
        TrimmingInner(),
        client,
        "m",
        audit=AuditPolicy(mode="ci"),
        recall=RecallPolicy(mode="every", every=1, alert_after=2, background=False, signals=False),
        on_alert=alerts.append,
    )
    s.ledger.record("constraint", "Only write to staging_orders.")
    run(s, 4)
    store, sid = s.store, s.session_id
    assert sid == "inner"
    audits = store.events(sid, "audit")
    assert audits and "staging_orders" in audits[0]["lost"][0]["answer"]
    scores = [e["score"] for e in store.events(sid, "recall")]
    assert scores[0] == 1.0 and scores[-1] == 0.0 and len(alerts) == 1
    # side calls went to the client, not into the wrapped session
    assert not any("Answer from the conversation" in str(i) for i in s._inner.items)


def test_a_broken_observer_does_not_break_get_items():
    class Boom(TrimmingInner):
        async def get_items(self, limit=None):
            return [{"role": "user", "content": object()}]  # unreadable content

    s = RotguardSession(Boom(), FakeClient(FakeCaller([], {})), "m")
    items = asyncio.run(s.get_items())
    assert len(items) == 1 and items[0]["role"] == "user"  # the agent still gets its items
