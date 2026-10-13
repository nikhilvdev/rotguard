# rotguard schemas

Generated from the Pydantic models by `uv run python -m rotguard.schemas`; do not edit by hand.
`tests/test_schemas.py` fails if they drift.

- `message.schema.json`: the framework-neutral message (blocks, provenance)
- `ledger_entry.schema.json`: one ledger entry (kind, status, supersession, source)

These are the contract for any non-Python port, together with the fixtures in `evals/`.
