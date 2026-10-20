# Contributing

Thanks for considering a contribution to `rotguard`. This project also follows a
[Code of Conduct](CODE_OF_CONDUCT.md); taking part in issues, PRs and discussions means agreeing to it.

## Setup

```bash
uv sync --extra langchain --extra eval
pre-commit install

uv run ruff check .
uv run mypy
uv run pytest
```

The tests use fake models and need no API keys. The benchmark harness in `evals/` calls real
models (a local Ollama model, or a hosted one with your own key) and is not part of `pytest`:

```bash
uv run python -m evals.run --model ollama:qwen2.5:3b --turns 40 --strategies none,summarize
uv run python -m evals.rescore evals/results/<run>.jsonl   # re-score stored replies, no model calls
```

## Pull request strategy

- **Branch from `main`**, name branches by intent: `feat/…`, `fix/…`, `docs/…`, `chore/…`.
- **Keep PRs scoped to one concern.** A refactor mixed into a feature is harder to review and revert.
- **Definition of done** before a PR is ready for review:
  1. Type hints throughout, `mypy --strict` clean
  2. Unit tests cover the new surface; existing tests still pass
  3. Public API additions documented in the README with a runnable example
  4. `CHANGELOG.md` has an entry under `Unreleased`
  5. If the message model or ledger entry shape changed, regenerate `schemas/`
     (`uv run python -m rotguard.schemas`)
  6. Any claim about quality or cost is backed by a number from `evals/`; say which run
- **Adapters** import their framework only inside `rotguard/adapters/`. The core must stay
  framework-free, and every adapter needs round-trip tests.
- **Fail open.** A hook that errors must log and let the agent's call go ahead unchanged.
- **CI must be green** and **at least one review approval** is required before merge.
- **Squash-merge** into `main`. Commit messages and PR titles are imperative: "Add rotating store",
  not "Added".

## Licensing of contributions

rotguard is dual-licensed: [GNU AGPL-3.0](LICENSE) for everyone, and a commercial license from the
maintainer. By opening a pull request you confirm that you wrote the change (or have the right to
submit it) and that it may be distributed under the AGPL-3.0 and under the commercial license the
maintainer offers. If you can't agree to that, please open an issue before sending code.

## Reporting issues

Bug reports and feature requests are welcome via GitHub issues. For security issues, see
[.github/SECURITY.md](.github/SECURITY.md).
