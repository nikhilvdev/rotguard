from .protocol import Store
from .sqlite import SQLiteStore


def open_store(store: Store | str | None) -> Store:
    """A store passed in is used as is; a path (or None, for memory) opens a SQLite store."""
    return SQLiteStore(store or ":memory:") if store is None or isinstance(store, str) else store


__all__ = ["SQLiteStore", "Store", "open_store"]
