import pytest
from test_cli import make_trace

from rotguard.adapters.mcp import RotguardTools
from rotguard.store import SQLiteStore


def tools():
    store = make_trace(":memory:")
    return RotguardTools(store, "orig"), store


def test_ledger_record_supersedes_and_persists_to_the_store():
    t, store = tools()
    assert t.ledger_record("constraint", "use staging_orders", key="table") == "recorded L2"
    assert t.ledger_record("constraint", "use archive_orders", key="table") == "recorded L3"
    assert t.ledger_record("goal", "x", supersedes="L99").startswith("not recorded")
    saved = {e["id"]: e["status"] for e in store.entries("orig")}
    assert saved == {"L1": "active", "L2": "superseded", "L3": "active"}  # nothing deleted
    assert RotguardTools(store, "orig").ledger.render().count("archive_orders") == 1  # reloads


def test_recall_returns_compacted_items_by_id_or_text():
    t, _ = tools()
    by_id = t.recall("f")
    assert "removed at turn 3" in by_id and "only write to staging_orders" in by_id
    assert t.recall("STAGING_orders").startswith("f (user, turn 1")  # case-insensitive text match
    assert t.recall("no such thing") == "nothing found" and t.recall("") == "nothing found"


def test_recall_caps_long_text_and_the_number_of_hits():
    store = SQLiteStore()
    row = {"turn": 1, "kind": "tool", "role": "user", "parents": [], "hash": "h"}
    store.add_items("s", [{**row, "id": f"i{n}", "text": "needle " + "x" * 3000} for n in range(5)])
    out = RotguardTools(store, "s").recall("needle")
    assert out.count("…") == 3 and out.endswith("(+2 more)")


def test_blame_and_report_return_plain_data():
    t, _ = tools()
    assert t.blame("staging_orders") == {"origins": ["f"], "carried_by": [], "source_items": []}
    rep = t.report()
    assert rep["session"] == "orig" and len(rep["compactions"]) == 1


async def test_real_mcp_server_exposes_and_routes_the_four_tools():
    pytest.importorskip("mcp.server.mcpserver")
    from rotguard.adapters.mcp import make_server

    t, store = tools()
    server = make_server(t)
    names = [x.name for x in await server.list_tools()]
    assert names == ["ledger_record", "recall", "blame", "report"]
    res = await server.call_tool("ledger_record", {"kind": "goal", "text": "ship v1"})
    assert res.structured_content == {"result": "recorded L2"} and not res.is_error
    assert any(e["text"] == "ship v1" for e in store.entries("orig"))
    res = await server.call_tool("recall", {"query": "noise 2"})
    assert "b (user" in res.structured_content["result"]
