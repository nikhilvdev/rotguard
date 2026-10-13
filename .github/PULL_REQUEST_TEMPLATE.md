## Summary

<!-- What does this change, and why? -->

## Checklist

- [ ] `ruff check .` passes
- [ ] `mypy` passes (`--strict`)
- [ ] `pytest` passes, and new surface area has test coverage
- [ ] Public API additions are documented in `README.md` with a runnable example
- [ ] `CHANGELOG.md` has an entry under `Unreleased`
- [ ] If this changes the message model or ledger entry shape, `schemas/` is regenerated
      (`uv run python -m rotguard.schemas`)
- [ ] Any claim about quality or cost is backed by a number from `evals/`

## Scope

<!-- One concern per PR where possible. Call out here if this
     intentionally spans more than one. -->
