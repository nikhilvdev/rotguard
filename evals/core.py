"""Scenario, compaction strategies and scoring for the Phase 0 harness."""

from __future__ import annotations

import random
import re
import time
from dataclasses import dataclass, field

from rotguard.ledger import Ledger
from rotguard.log import get_logger

from .models import Client, Message, purpose

log = get_logger("evals")

SYSTEM = "You are a coding and research agent working through a long task. Answer concisely."

FILLER_TOPICS = [
    "grep results for `handle_request` across 41 files",
    "pytest output: 212 passed, 3 skipped, warnings summary",
    "directory listing of build/ with 87 entries",
    "API response for /v2/orders?page=7 (200 OK, 50 items)",
    "stack trace from a retried HTTP call, retry 2 of 3",
    "diff of utils/format.py, 64 lines changed",
]


def filler(rng: random.Random, n_words: int) -> str:
    topic = rng.choice(FILLER_TOPICS)
    words = [
        f"{rng.choice(['id', 'ref', 'path', 'row', 'key'])}{rng.randint(100, 9999)}"
        for _ in range(n_words)
    ]
    return f"[tool result] {topic}\n" + " ".join(words)


@dataclass
class Probe:
    question: str
    must: list[str]  # regexes that must all match the reply
    must_not: list[str] = field(default_factory=list)  # regexes that must not match
    # Stale values: mentioning one is fine (e.g. "Redis, not Postgres") unless it comes
    # before the first `must` match, i.e. the reply leads with the wrong answer.
    stale: list[str] = field(default_factory=list)


@dataclass
class Scenario:
    suite: str
    id: str
    history: list[Message]
    probe: Probe
    # What an agent using `ledger_record` would have recorded (oracle: assumes faithful recording).
    ledger: list[dict] = field(default_factory=list)


@dataclass
class Result:
    suite: str
    scenario: str
    strategy: str
    model: str
    context_chars: int
    passed: bool
    reply: str
    stale: bool = False  # the reply acted on a value that had been replaced


def used_stale(probe: Probe, reply: str) -> bool:
    """The reply gives a replaced value instead of the current one, or leads with it.

    Mentioning the old value after the right one ("Redis, not Postgres") does not count.
    """
    olds = [m for p in probe.stale if (m := re.search(p, reply, re.I))]
    if not olds:
        return False
    news = [m for p in probe.must if (m := re.search(p, reply, re.I))]
    return len(news) < len(probe.must) or min(m.start() for m in olds) < min(
        m.start() for m in news
    )


def score(probe: Probe, reply: str) -> bool:
    hits = [re.search(p, reply, re.I) for p in probe.must]
    if not all(hits):
        return False
    if any(re.search(p, reply, re.I) for p in probe.must_not):
        return False
    first_ok = min(h.start() for h in hits)
    return not any((m := re.search(p, reply, re.I)) and m.start() < first_ok for p in probe.stale)


# --- compaction strategies: history -> history -------------------------------------------


def no_compaction(history, client):
    return history


def truncate(keep_last: int = 6):
    def f(history, client):
        return history[-keep_last:]

    return f


def summarize(keep_last: int = 6):
    """Naive LLM summary of everything except the last `keep_last` messages."""

    def f(history, client: Client):
        old, recent = history[:-keep_last], history[-keep_last:]
        text = "\n".join(f"{m['role']}: {m['content']}" for m in old)
        with purpose(client, "summarizer"):
            summary = client.complete(
                "Summarize this agent transcript so the work can continue from the summary alone.",
                [{"role": "user", "content": text}],
                max_tokens=500,
            )
        return [{"role": "user", "content": f"[summary of earlier work]\n{summary}"}, *recent]

    return f


STRATEGIES = {"none": no_compaction, "truncate": truncate(), "summarize": summarize()}


def run_scenario(sc: Scenario, strategy: str, client: Client) -> Result:
    base, _, suffix = strategy.partition("+")
    history = STRATEGIES[base](sc.history, client)
    question = sc.probe.question
    if suffix == "ledger":
        led = Ledger()
        for e in sc.ledger:
            led.record(**e)
        question = f"{led.render()}\n\n{question}"  # ledger goes at the end of the request
    messages = [*history, {"role": "user", "content": question}]
    t0 = time.perf_counter()
    reply = client.complete(SYSTEM, messages)
    latency_ms = (time.perf_counter() - t0) * 1000
    passed = score(sc.probe, reply)
    # TODO: switch to log.llm_call(...) once logquill releases it (unreleased after 1.0.0).
    log.info(
        "probe",
        model=client.name,
        latency_ms=round(latency_ms, 1),
        suite=sc.suite,
        scenario=sc.id,
        strategy=strategy,
        passed=passed,
        context_chars=sum(len(m["content"]) for m in history),
    )
    return Result(
        suite=sc.suite,
        scenario=sc.id,
        strategy=strategy,
        model=client.name,
        context_chars=sum(len(m["content"]) for m in history),
        passed=passed,
        reply=reply,
        stale=used_stale(sc.probe, reply),
    )
