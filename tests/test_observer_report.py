from rotguard.core import Message, Observer, Provenance, TextBlock
from rotguard.report import blame, diff
from rotguard.store import SQLiteStore


def msg(i, text, kind=None):
    return Message(
        id=i,
        role="user",
        blocks=[TextBlock(text=text)],
        provenance=Provenance(kind=kind) if kind else None,
    )


def run(store, obs, turn, messages):
    o = obs.observe(messages, turn)
    store.add_items(
        "s",
        [
            {
                "id": m.id,
                "turn": turn,
                "kind": m.provenance.kind,
                "role": m.role,
                "parents": m.provenance.parents,
                "hash": m.content_hash,
                "text": m.text,
            }
            for m in o.added
        ],
    )
    store.mark_removed("s", o.removed, turn)
    store.save_snapshot("s", turn, [m.id for m in messages])
    return o


def test_detects_summary_compaction_and_parents():
    obs = Observer()
    ctx = [msg("a", "constraint: use staging_orders"), msg("b", "x"), msg("c", "y")]
    assert obs.observe(ctx, 1).compaction is None
    o = obs.observe([msg("s1", "summary: use staging_orders", "summary"), ctx[2]], 2)
    assert o.compaction and o.compaction.removed_ids == ["a", "b"]
    assert o.compaction.summary_ids == ["s1"]
    assert o.added[0].provenance.parents == ["a", "b"]


def test_detects_in_place_rewrite():
    obs = Observer("request")
    obs.observe([msg("t", "huge tool output")], 1)
    o = obs.observe([msg("t", "[cleared]")], 2)
    assert o.rewritten == ["t"] and o.compaction and not o.removed


def test_diff_and_blame_through_two_compactions():
    store, obs = SQLiteStore(), Observer()
    fact = msg("f", "Never write to prod_orders; use staging_orders.")
    run(store, obs, 1, [fact, msg("a", "noise 1"), msg("b", "noise 2")])
    run(store, obs, 2, [fact, msg("a", "noise 1"), msg("b", "noise 2"), msg("c", "noise 3")])
    # round 1: fact + noise folded into summary s1
    run(
        store,
        obs,
        3,
        [msg("s1", "Earlier: only use staging_orders.", "summary"), msg("c", "noise 3")],
    )
    # round 2: s1 folded into s2
    run(
        store,
        obs,
        4,
        [msg("s2", "Still: use staging_orders only.", "summary"), msg("d", "noise 4")],
    )

    d = diff(store, "s", 2, 3)
    assert set(d.removed) == {"f", "a", "b"} and d.added == ["s1"]
    assert set(d.rewritten["s1"]) == {"f", "a", "b"}

    b = blame(store, "s", "staging_orders")
    assert [o["id"] for o in b.origins] == ["f"]  # original source, not a summary
    assert {s["id"] for s in b.carried_by} == {"s1", "s2"}
    assert "f" in b.ancestors  # reachable through both summaries
