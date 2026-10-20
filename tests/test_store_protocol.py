from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage
from test_langchain import FakeChat

from rotguard.adapters.langchain import RotguardMiddleware
from rotguard.store import SQLiteStore, Store, open_store


class OnlyTheProtocol:
    """A store that exposes exactly the `Store` methods and nothing else (no SQLite attributes)."""

    def __init__(self):
        self._inner = SQLiteStore()
        self.calls = set()
        for name in [
            "add_items",
            "mark_removed",
            "items",
            "item",
            "save_snapshot",
            "snapshot_at",
            "upsert_entry",
            "entries",
            "add_event",
            "events",
            "sessions",
            "close",
        ]:
            setattr(self, name, self._wrap(name))

    def _wrap(self, name):
        fn = getattr(self._inner, name)

        def call(*a, **k):
            self.calls.add(name)
            return fn(*a, **k)

        return call


def test_the_middleware_runs_on_a_store_that_only_has_the_protocol_methods():
    store: Store = OnlyTheProtocol()
    rg = RotguardMiddleware(store)
    rg.ledger.record("goal", "ship v1")
    agent = create_agent(
        model=FakeChat(replies=[AIMessage(content="ok")]), tools=[], middleware=[rg]
    )
    agent.invoke({"messages": [HumanMessage(content="hello")]})
    assert rg.store is store
    assert {"add_items", "save_snapshot", "upsert_entry"} <= store.calls
    assert store.sessions() == [rg.session_id]


def test_open_store_uses_what_it_is_given_and_opens_sqlite_otherwise():
    mine = OnlyTheProtocol()
    assert open_store(mine) is mine
    assert isinstance(open_store(None), SQLiteStore) and isinstance(
        open_store(":memory:"), SQLiteStore
    )
