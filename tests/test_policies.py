from rotguard.core import Message, Provenance, TextBlock, ToolCallBlock, ToolResultBlock
from rotguard.handoff import handoff, pinned_closure, tail_start
from rotguard.ledger import Ledger
from rotguard.policies import DedupPolicy, OffloadPolicy, StagedCleanup
from rotguard.policies.dedup import find_duplicates
from rotguard.policies.offload import (
    is_stub,
    make_stub,
    needs_offload,
    offload_id,
    read_page,
    store_output,
)
from rotguard.policies.staged import run_staged
from rotguard.store import SQLiteStore
from rotguard.testing import assert_history_integrity
from rotguard.tokens import TokenCounter, estimate


def text(i, role, t="x"):
    return Message(id=i, role=role, blocks=[TextBlock(text=t)], provenance=Provenance(kind=role))


def call(i, cid):
    return Message(id=i, role="assistant", blocks=[ToolCallBlock(id=cid, name="ls", args={})])


def result(i, cid, body="out"):
    return Message(
        id=i, role="tool", blocks=[ToolResultBlock(call_id=cid, name="ls", content=body)]
    )


def ids(ms):
    return [m.id for m in ms]


BIG = "\n".join(f"line {n} of a long listing" for n in range(400))  # about 2,000 tokens


# --- A4: tokens -------------------------------------------------------------------------------


def test_token_estimate_is_deterministic_and_flagged():
    c = TokenCounter()
    msgs = [text("a", "user", "x" * 400)]
    assert c.count(msgs) == 100 == c.count(msgs) and c.estimated
    assert estimate("") == 0 and estimate("abc") == 1


def test_a_failing_provider_falls_back_to_the_estimate():
    def boom(_):
        raise RuntimeError("down")

    c = TokenCounter(boom)
    assert not c.estimated
    assert c.count([text("a", "user", "x" * 40)]) == 10 and c.estimated
    assert TokenCounter(lambda ms: 7).count([text("a", "user")]) == 7


# --- F5a: offload -----------------------------------------------------------------------------


def test_big_output_becomes_a_stub_with_the_same_call_id_and_pages_back():
    policy = OffloadPolicy(max_tokens=500, preview_lines=3)
    assert needs_offload(BIG, policy) and not needs_offload("short", policy)
    stub = make_stub(offload_id("c1"), BIG, policy)
    assert stub.splitlines()[:3] == BIG.splitlines()[:3] and is_stub(stub)
    assert "offload:c1" in stub and not needs_offload(stub, policy)  # a stub is never re-offloaded
    page = read_page(BIG, 0, 200)
    assert page.endswith("next: start_line=200]") and "line 199 " in page
    assert read_page(BIG, 399, 200).endswith("of 400: end]")
    assert read_page(BIG, 10_000).endswith("end]")  # past the end is harmless


def test_stored_output_is_kept_in_full_and_storing_twice_changes_nothing():
    store = SQLiteStore()
    store_output(store, "s", "offload:c1", BIG, turn=3, tool_name="ls", call_id="c1")
    store_output(store, "s", "offload:c1", "different", turn=4)
    row = store.item("s", "offload:c1")
    assert row and row["text"] == BIG and row["kind"] == "offloaded" and row["call_id"] == "c1"
    assert store.item("s", "nope") is None


# --- F5b: dedup -------------------------------------------------------------------------------


def test_duplicates_collapse_to_the_first_copy_and_pinned_or_short_ones_are_kept():
    long = "y" * 300
    outs = [("m1", long), ("m2", None), ("m3", long), ("m4", "ok"), ("m5", "ok"), ("m6", long)]
    assert find_duplicates(outs, DedupPolicy()) == {2: "m1", 5: "m1"}
    assert find_duplicates(outs, DedupPolicy(), pinned=["m3"]) == {5: "m1"}


# --- F5d: staged cleanup ----------------------------------------------------------------------


def convo(n_tools=5, big=BIG):
    ms = [text("sys", "system"), text("u0", "user")]
    for k in range(n_tools):
        ms += [call(f"a{k}", f"c{k}"), result(f"t{k}", f"c{k}", big), text(f"u{k + 1}", "user")]
    return ms


class Summarizer:
    def __init__(self):
        self.calls = []

    def __call__(self, msgs):
        self.calls.append(ids(msgs))
        return "SUMMARY"


def test_under_budget_nothing_happens():
    r = run_staged(convo(), StagedCleanup(budget_tokens=10**6), TokenCounter())
    assert r is None


def test_clearing_is_enough_so_no_summarizer_call_is_made():
    s = Summarizer()
    cfg = StagedCleanup(budget_tokens=4000, clear_keep=1, summarizer=s)
    r = run_staged(convo(), cfg, TokenCounter())
    assert r and r.stages == ["clear"] and s.calls == [] and r.tokens_after < r.tokens_before
    assert r.cleared == 4 and [o.call_id for o in r.offloaded] == ["c0", "c1", "c2", "c3"]
    kept = [m for m in r.messages if m.id == "t4"][0]
    assert kept.blocks[0].content == BIG  # the newest result is whole
    assert_history_integrity(r.messages)  # stubs keep their tool_call_id


def test_still_over_budget_summarizes_exactly_once_and_keeps_pinned_and_pairs():
    s = Summarizer()
    cfg = StagedCleanup(budget_tokens=50, clear_keep=1, summarizer=s, keep_messages=4)
    msgs = convo()
    r = run_staged(msgs, cfg, TokenCounter(), pinned={"u1"})
    assert r and r.stages == ["clear", "summarize"] and len(s.calls) == 1
    out = ids(r.messages)
    assert out[0] == "sys" and out[1].startswith("staged-summary-") and "u1" in out
    assert "u1" not in s.calls[0] and "u0" in s.calls[0]  # pinned is not summarized away
    summary = r.messages[1]
    assert summary.provenance.kind == "summary" and set(summary.provenance.parents) == set(
        s.calls[0]
    )
    assert_history_integrity(r.messages)
    assert r.messages[-1].id == msgs[-1].id  # the tail is whole


def test_force_runs_even_under_budget_and_no_summarizer_means_stage_two_is_off():
    s = Summarizer()
    forced = run_staged(
        convo(),
        StagedCleanup(budget_tokens=10**6, summarizer=s, keep_messages=4),
        TokenCounter(),
        force=True,
    )
    assert forced and "summarize" in forced.stages
    off = run_staged(convo(), StagedCleanup(budget_tokens=50, clear_keep=0), TokenCounter())
    assert off and off.stages == ["clear"]
    nothing = run_staged(
        convo(), StagedCleanup(budget_tokens=50, summarizer=s), TokenCounter(), allow_summary=False
    )
    assert nothing and "summarize" not in nothing.stages


# --- F5c: pinned messages survive handoff -------------------------------------------------------


def test_pinned_messages_and_their_pairs_survive_a_handoff():
    msgs = [
        text("u0", "user"),
        call("a0", "c0"),
        result("t0", "c0"),
        text("u1", "user"),
        text("a1", "assistant"),
        text("u2", "user"),
    ]
    out = handoff(msgs, Ledger(), keep=1, pinned={"t0"})
    assert ids(out) == ["a0", "t0", "u2"]  # the pinned result brings its call along
    assert_history_integrity(out)
    assert pinned_closure(msgs, {"a0"}) == {"a0", "t0"} and tail_start(msgs, 1) == 5


def test_a_tool_loop_with_no_later_user_message_can_still_be_summarized():
    s = Summarizer()
    msgs = [text("u0", "user")]
    for k in range(6):
        msgs += [call(f"a{k}", f"c{k}"), result(f"t{k}", f"c{k}", "r")]
    cfg = StagedCleanup(budget_tokens=0, clear_keep=0, summarizer=s, keep_messages=4)
    r = run_staged(msgs, cfg, TokenCounter(), force=True)
    assert r and "summarize" in r.stages and s.calls
    assert r.messages[0].provenance.kind == "summary"  # a user-role summary leads the history
    assert_history_integrity(r.messages)
    # a handoff with no ledger note still has to start on a user message
    assert (
        tail_start(msgs, 3) == 0 and tail_start(msgs, 3, user_first=False) == 9
    )  # pulled back to the call


def test_a_stub_is_always_much_smaller_even_for_one_enormous_line():
    one_line = "x" * 20_000
    stub = make_stub(offload_id("c9"), one_line, OffloadPolicy(max_tokens=100, preview_lines=10))
    assert len(stub) < 1400 and stub.startswith("x" * 1000 + "…") and is_stub(stub)
