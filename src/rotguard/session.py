"""One rotguard session: observers, store writes, and compaction audits.

Framework adapters own only the translation to and from their message types; everything else
lives here so LangChain, LangGraph and the Anthropic wrapper behave the same way.
"""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

from rotguard.core import (
    Message,
    Observation,
    Observer,
    Provenance,
    TextBlock,
    default_provenance,
)
from rotguard.handoff import handoff
from rotguard.ledger import Entry, Ledger
from rotguard.log import get_logger
from rotguard.probes import (
    AuditPolicy,
    FactPool,
    RecallMonitor,
    RecallPolicy,
    RecallResult,
    SideCaller,
    ledger_tokens,
    passive_signals,
    run_audit,
    run_recall,
)
from rotguard.probes.quiz import Fact, norm
from rotguard.store import Store
from rotguard.telemetry import emit

log = get_logger(__name__)


@dataclass
class _Pending:
    channel: str
    turn: int
    old: list[Message]
    new: list[Message]
    removed: list[Message]


class Session:
    def __init__(
        self,
        store: Store,
        session_id: str,
        ledger: Ledger,
        audit: AuditPolicy | None = None,
        recall: RecallPolicy | None = None,
        on_alert: Callable[[RecallResult], None] | None = None,
    ):
        self.store, self.session_id, self.ledger = store, session_id, ledger
        self.policy = audit or AuditPolicy(mode="off")
        self._gate = self.policy.gate()
        self.recall = RecallMonitor(recall or RecallPolicy())
        self._on_alert = on_alert
        self.handoff_pending = False
        self.compact_pending = False  # an alert asked for a staged cleanup (action="compact")
        self.pinned: set[str] = {e["id"] for e in store.events(session_id, "pin")}
        self.pool = FactPool()  # facts from the conversation, for target="history"
        self.history, self.request = Observer("history"), Observer("request")
        self.turn = 0
        self._prev_hist: list[Message] = []
        self._prev_req: list[Message] = []
        self._edit_seen = False
        self._pending: list[_Pending] = []
        self._pool: ThreadPoolExecutor | None = None
        self._futures: list[Future[None]] = []

    # --- observation ---------------------------------------------------------------------

    def _row(self, m: Message) -> dict[str, Any]:
        prov = m.provenance or default_provenance(m, self.turn)
        return {
            "id": m.id,
            "turn": self.turn,
            "kind": prov.kind,
            "role": m.role,
            "tool_name": prov.tool_name,
            "call_id": prov.call_id,
            "parents": prov.parents,
            "hash": m.content_hash,
            "text": m.text,
        }

    def _record(self, obs: Observation, channel: str) -> None:
        if obs.compaction:
            self.event("compaction", obs.compaction.to_dict())
            log.info(
                "compaction",
                channel=channel,
                turn=self.turn,
                removed=len(obs.removed),
                rewritten=len(obs.rewritten),
            )

    def observe_history(self, neutral: list[Message]) -> Observation:
        """The stored history. A summary or trim here is a compaction."""
        obs = self.history.observe(neutral, self.turn)
        if self.recall.policy.mode != "off":
            self.pool.add(obs.added, ledger_tokens(self.ledger.all()))
        if obs.added:
            self.store.add_items(self.session_id, [self._row(m) for m in obs.added])
        self.store.mark_removed(self.session_id, obs.removed, self.turn)
        self.store.save_snapshot(self.session_id, self.turn, [m.id for m in neutral])
        self._record(obs, "history")
        if obs.compaction and obs.removed:
            gone = set(obs.removed)
            self._pending.append(
                _Pending(
                    "history",
                    self.turn,
                    self._prev_hist,
                    list(neutral),
                    [m for m in self._prev_hist if m.id in gone],
                )
            )
        self._prev_hist = list(neutral)
        return obs

    def observe_request(self, neutral: list[Message]) -> Observation:
        """The final request, after other middleware. Catches in-place clearing of tool output,
        which is applied to a copy and never reaches the stored history."""
        obs = self.request.observe(neutral, self.turn)
        self._record(obs, "request")
        if obs.rewritten and not obs.removed and not self._edit_seen:
            self._edit_seen = True  # the same clearing repeats every call: audit it once
            hit = set(obs.rewritten)
            self._pending.append(
                _Pending(
                    "request",
                    self.turn,
                    self._prev_req,
                    list(neutral),
                    [m for m in self._prev_req if m.id in hit],
                )
            )
        self._prev_req = list(neutral)
        return obs

    # --- audits --------------------------------------------------------------------------

    def run_pending(self, caller: SideCaller, ledger_text: str = "", note: str = "") -> None:
        """Start audits for compactions seen since the last call, as the policy allows.

        `ledger_text` is the ledger as the audit quiz prefix. `note` is everything the adapter adds
        to this request (ledger plus marker); recall probes see the request with it, because that
        is what the model will actually be given.
        """
        pending, self._pending = self._pending, []
        context = list(self._prev_req or self._prev_hist)
        if note:
            context.append(
                Message(
                    id="rotguard-note",
                    role="user",
                    blocks=[TextBlock(text=note)],
                    provenance=Provenance(kind="ledger"),
                )
            )
        self.run_recall_check(context, caller, ledger_shown=bool(ledger_text))
        for p in pending:
            if not self._gate.allow():
                continue
            entries: list[Entry] = [e.model_copy() for e in self.ledger.active()]

            def job(p: _Pending = p, entries: list[Entry] = entries) -> None:
                self._execute(p, caller, ledger_text, entries)

            self._submit(job, self.policy.mode == "ci" or not self.policy.background)

    def _submit(self, job: Callable[[], None], inline: bool) -> None:
        if inline:
            job()
            return
        self._pool = self._pool or ThreadPoolExecutor(1, thread_name_prefix="rotguard-audit")
        self._futures.append(self._pool.submit(job))

    # --- recall probes -------------------------------------------------------------------

    def _recall_target(self, ledger_shown: bool) -> str:
        t = self.recall.policy.target
        if t != "auto":
            return t
        if ledger_shown:  # the model can always answer ledger questions from the ledger
            return "history"
        return "ledger" if self.ledger.active() else "history"

    def run_recall_check(
        self, context: list[Message], caller: SideCaller, ledger_shown: bool = False
    ) -> None:
        """Start a recall probe over `context` if the policy says one is due this turn."""
        if self.recall.policy.mode == "off":
            return
        entries: list[Entry] = [e.model_copy() for e in self.ledger.all()]
        if self.recall.policy.signals:
            sig = passive_signals(context, [e for e in entries if e.status == "active"])
            if sig.flag:
                self.recall.flag()
                self.event("signals", {"turn": self.turn, **sig.__dict__})
        if not self.recall.due(self.turn):
            return
        turn = self.turn
        target = self._recall_target(ledger_shown)
        facts = None
        if target == "history":
            covered = ledger_tokens(entries)  # a fact the ledger recorded since is no longer ours
            facts = [f for f in self.pool.facts if norm(f.answer) not in covered]

        def job() -> None:
            self._execute_recall(context, entries, caller, turn, facts, target)

        self._submit(job, not self.recall.policy.background)

    def event(self, kind: str, data: dict[str, Any]) -> None:
        """Store an event and, when OpenTelemetry is set up, emit it as a span event."""
        self.store.add_event(self.session_id, kind, data)
        emit(kind, self.session_id, data)

    def pin(self, message_id: str) -> None:
        """Keep a message through offload, dedup, staged cleanup and handoff, and re-attach its
        text to a request if another tool compacted it away."""
        if message_id not in self.pinned:
            self.pinned.add(message_id)
            self.event("pin", {"id": message_id})

    def take_handoff(self, context: list[Message]) -> list[Message] | None:
        """If an alert asked for a handoff, the fresh context to use instead of `context`.

        The caller must replace its history with the result; this only builds it and records
        the event. Returns None when no handoff is pending. A pending handoff that would not
        shrink the context yet stays pending, and a passing recall probe cancels it.
        """
        if not self.handoff_pending:
            return None
        fresh = handoff(
            context, self.ledger, keep=self.recall.policy.handoff_keep, pinned=self.pinned
        )
        if len(fresh) >= len(context):  # nothing would be dropped: stay pending until it would
            return None
        self.handoff_pending = False
        self.event(
            "handoff",
            {"turn": self.turn, "before": len(context), "after": len(fresh)},
        )
        log.info("handoff", turn=self.turn, before=len(context), after=len(fresh))
        return fresh

    def marker_note(self) -> str:
        """The standing instruction for the marker check, or "" when it is off (tier 2)."""
        m = self.recall.policy.marker
        return f"[rotguard marker: begin every reply with {m}]" if m else ""

    def observe_reply(self, text: str) -> None:
        """Tier 2: a reply without the marker means the standing instruction was dropped."""
        m = self.recall.policy.marker
        if m and self.recall.policy.mode != "off" and m not in text:
            self.recall.flag()
            self.event("marker_miss", {"turn": self.turn})
            log.info("marker missed", turn=self.turn)

    def _execute_recall(
        self,
        context: list[Message],
        entries: list[Entry],
        caller: SideCaller,
        turn: int,
        facts: list[Fact] | None = None,
        target: str = "ledger",
    ) -> None:
        try:
            r = run_recall(
                context=context,
                entries=entries,
                caller=caller,
                sample=self.recall.policy.sample,
                rng=self.recall.rng,
                turn=turn,
                style=self.recall.policy.style,
                facts=facts,
            )
            alert = self.recall.record(r)
            if r.score is not None and r.score >= self.recall.policy.pass_at:
                self.handoff_pending = False  # recall recovered: no need to restart
                self.compact_pending = False
            self.event(
                "recall",
                {**r.to_dict(), "rolling": self.recall.rolling, "alert": alert, "target": target},
            )
            log.info("recall", turn=turn, score=r.score, missed=len(r.missed), alert=alert)
            if alert and self.recall.policy.action == "handoff":
                self.handoff_pending = True
            if alert and self.recall.policy.action == "compact":
                self.compact_pending = True
            if alert and self._on_alert:
                self._on_alert(r)
        except Exception as exc:  # a probe must never break the agent
            self.event("recall_error", {"error": repr(exc)})
            log.warn("recall failed", error=repr(exc))

    def _execute(
        self, p: _Pending, caller: SideCaller, ledger_text: str, entries: list[Entry]
    ) -> None:
        try:
            report = run_audit(
                old=p.old,
                new=p.new,
                removed=p.removed,
                entries=entries,
                caller=caller,
                max_facts=self.policy.max_facts,
                cloze=self.policy.cloze,
                prefix=ledger_text,
            )
            self.event("audit", {"channel": p.channel, "turn": p.turn, **report.to_dict()})
            log.info(
                "audit", channel=p.channel, turn=p.turn, loss=report.loss, lost=len(report.lost)
            )
        except Exception as exc:  # an audit must never break the agent
            self.event("audit_error", {"channel": p.channel, "error": repr(exc)})
            log.warn("audit failed", error=repr(exc))

    def flush(self, timeout: float | None = None) -> None:
        """Wait for background audits."""
        futures, self._futures = self._futures, []
        for f in futures:
            f.result(timeout)

    def close(self) -> None:
        self.flush()
        if self._pool:
            self._pool.shutdown(wait=True)
