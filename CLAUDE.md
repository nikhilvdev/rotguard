# CLAUDE.md

rotguard measures and guards against context rot in LLM agents. The plan is `rotguard.md`; read its
section 5 (architecture) and 7 (LangChain integration) before changing core or adapter code.

## Commands

```bash
uv sync --extra langchain --extra eval
uv run pytest                       # fake models only, no API keys
uv run ruff check . && uv run ruff format .
uv run mypy                         # strict, src/rotguard only
uv run python -m rotguard.schemas   # regenerate schemas/ after changing Message or Entry
uv run python -m evals.run --model ollama:qwen2.5:3b --turns 40 --strategies none,summarize
uv run python -m evals.rescore evals/results/<run>.jsonl   # re-score stored replies, no model calls
```

## Layout

- `src/rotguard/core` neutral message model and the observer (framework-free)
- `src/rotguard/ledger`, `store`, `report` ledger with supersession, SQLite store, diff and blame
- `src/rotguard/adapters/langchain.py` the only place that imports LangChain
- `src/rotguard/log.py` logging via logquill; use `get_logger(__name__)`, never `logging` directly
- `evals/` benchmark harness (a dev tool, not shipped); `schemas/` generated JSON schemas

## Rules that are easy to break

- **Fail open.** Adapter hooks catch errors, log them and let the agent's call proceed unchanged.
- **Non-destructive.** The ledger is added to the model request for that call only
  (`request.override`); it must never be written into stored history.
- **Never delete ledger entries.** A newer entry marks the older one `superseded`.
- **Middleware order.** rotguard must be listed after `SummarizationMiddleware` and
  `ContextEditingMiddleware`; `tests/test_langchain.py::test_ordering_rules` pins this.
- **logquill children copy transports when created**, so module-level loggers go through the lazy
  proxy in `log.py`. The released logquill 1.0.0 has `warn` (not `warning`) and no `llm_call` yet.
- **Eval claims need numbers.** Anything stated about quality or cost must come from an `evals/` run.
  Note which model, how many turns, and the sample size; most cells here are 3 to 4 scenarios.
