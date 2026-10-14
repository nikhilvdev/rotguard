"""One rotguard session: observers, store writes, and compaction audits.

Framework adapters own only the translation to and from their message types; everything else
lives here so LangChain, LangGraph and the Anthropic wrapper behave the same way.
"""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

from rotguard.core import Message, Observation, Observer, default_provenance
from rotguard.ledger import Entry, Ledger
from rotguard.log import get_logger
from rotguard.probes import AuditPolicy, SideCaller, run_audit
from rotguard.store import SQLiteStore

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
        self, store: SQLiteStore, session_id: str, ledger: Ledger, audit: AuditPolicy | None = None
    ):
        self.store, self.session_id, self.ledger = store, session_id, ledger
        self.policy = audit or AuditPolicy(mode="off")
        self._gate = self.policy.gate()
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
            self.store.add_event(self.session_id, "compaction", obs.compaction.to_dict())
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

    def run_pending(self, caller: SideCaller, ledger_text: str = "") -> None:
        """Start audits for compactions seen since the last call, as the policy allows."""
        pending, self._pending = self._pending, []
        for p in pending:
            if not self._gate.allow():
                continue
            entries: list[Entry] = [e.model_copy() for e in self.ledger.active()]

            def job(p: _Pending = p, entries: list[Entry] = entries) -> None:
                self._execute(p, caller, ledger_text, entries)

            if self.policy.mode == "ci" or not self.policy.background:
                job()
            else:
                self._pool = self._pool or ThreadPoolExecutor(
                    1, thread_name_prefix="rotguard-audit"
                )
                self._futures.append(self._pool.submit(job))

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
                prefix=ledger_text,
            )
            self.store.add_event(
                self.session_id, "audit", {"channel": p.channel, "turn": p.turn, **report.to_dict()}
            )
            log.info(
                "audit", channel=p.channel, turn=p.turn, loss=report.loss, lost=len(report.lost)
            )
        except Exception as exc:  # an audit must never break the agent
            self.store.add_event(
                self.session_id, "audit_error", {"channel": p.channel, "error": repr(exc)}
            )
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
