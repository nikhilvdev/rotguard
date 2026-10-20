"""JSON schemas for the neutral message model, ledger entries, reports and stored events.

The files in `schemas/` are the language-neutral contract: a port to another language has to
produce and accept the same shapes. Regenerate with `uv run python -m rotguard.schemas`.
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel

from rotguard.core import Message
from rotguard.ledger import Entry
from rotguard.report.models import AuditEvent, RecallEvent, Report

SCHEMAS: dict[str, type[BaseModel]] = {
    "message.schema.json": Message,
    "ledger_entry.schema.json": Entry,
    "report.schema.json": Report,
    "audit_event.schema.json": AuditEvent,
    "recall_event.schema.json": RecallEvent,
}
DIR = Path(__file__).resolve().parents[2] / "schemas"


def render() -> dict[str, str]:
    return {
        name: json.dumps(model.model_json_schema(), indent=2) + "\n"
        for name, model in SCHEMAS.items()
    }


def main() -> None:
    DIR.mkdir(exist_ok=True)
    for name, text in render().items():
        (DIR / name).write_text(text)
        print("wrote", DIR / name)


if __name__ == "__main__":
    main()
