from rotguard.core import Message, Provenance, TextBlock, ToolCallBlock, ToolResultBlock
from rotguard.handoff import NOTE, handoff
from rotguard.ledger import Ledger


def text(i, role, t="x"):
    return Message(id=i, role=role, blocks=[TextBlock(text=t)], provenance=Provenance(kind=role))


def call(i, cid):
    return Message(
        id=i,
        role="assistant",
        blocks=[ToolCallBlock(id=cid, name="ls", args={})],
        provenance=Provenance(kind="assistant"),
    )


def result(i, cid):
    return Message(
        id=i,
        role="tool",
        blocks=[ToolResultBlock(call_id=cid, name="ls", content="a.txt")],
        provenance=Provenance(kind="tool"),
    )


def ids(ms):
    return [m.id for m in ms]


def ledger():
    led = Ledger()
    led.record("constraint", "Only write to staging_orders.")
    return led


def test_keeps_system_a_ledger_note_and_the_tail_and_leaves_the_input_alone():
    msgs = [text("s", "system"), text("u1", "user"), text("a1", "assistant"), text("u2", "user")]
    before = ids(msgs)
    out = handoff(msgs, ledger(), keep=1)
    assert ids(out) == ["s", "handoff-ledger", "u2"] and ids(msgs) == before
    note = out[1].text
    assert note.startswith(NOTE) and "Only write to staging_orders." in note
    assert out[1].provenance.kind == "ledger"


def test_no_note_when_the_ledger_is_empty():
    out = handoff([text("u1", "user"), text("a1", "assistant")], Ledger(), keep=1)
    assert ids(out) == ["u1", "a1"]  # widened back to a user message


def test_tail_starts_on_a_user_message_and_keeps_tool_pairs_together():
    msgs = [
        text("u1", "user"),
        call("a1", "c1"),
        result("t1", "c1"),
        text("u2", "user"),
        call("a2", "c2"),
        result("t2", "c2"),
        text("a3", "assistant"),
    ]
    out = handoff(msgs, Ledger(), keep=2)  # naive tail would be [t2, a3]: a result with no call
    assert ids(out) == ["u2", "a2", "t2", "a3"]


def test_widens_for_a_tool_call_when_no_user_message_precedes_it():
    msgs = [call("a1", "c1"), result("t1", "c1"), text("a2", "assistant")]
    assert ids(handoff(msgs, Ledger(), keep=2)) == ["a1", "t1", "a2"]
    parallel = [text("u", "user"), call("a1", "c1"), call("a2", "c2"), result("t2", "c2")]
    assert ids(handoff(parallel, Ledger(), keep=1)) == ["u", "a1", "a2", "t2"]


def test_keep_larger_than_history_returns_everything():
    msgs = [text("u1", "user"), text("a1", "assistant")]
    assert ids(handoff(msgs, Ledger(), keep=50)) == ["u1", "a1"]
