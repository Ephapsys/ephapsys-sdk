# SPDX-License-Identifier: Apache-2.0
"""Local append-only journal for A2A message decisions.

Records the decision taken for each received A2A message — verified,
rejected, guardrail-blocked, quarantine alert, system event — to a
local JSONL file. Useful for audit, post-incident review, and selective
rollback when a sender is later determined to have been compromised.

The journal is intentionally minimal: append-only, one JSON record per
line, no rotation, no locking. Concurrent writers should each use their
own journal path (typical pattern: one journal per agent).
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Optional


DEFAULT_JOURNAL_PATH = "~/.ephapsys/a2a-journal.jsonl"

# Decision values that callers and downstream tooling can rely on.
DECISION_VERIFIED = "verified"
DECISION_REJECTED = "rejected"
DECISION_GUARDRAIL_BLOCKED = "guardrail_blocked"
DECISION_QUARANTINE_ALERT = "quarantine_alert"
DECISION_SYSTEM_EVENT = "system_event"

ALL_DECISIONS = frozenset(
    {
        DECISION_VERIFIED,
        DECISION_REJECTED,
        DECISION_GUARDRAIL_BLOCKED,
        DECISION_QUARANTINE_ALERT,
        DECISION_SYSTEM_EVENT,
    }
)


class MessageJournal:
    """Append-only JSONL journal of A2A message decisions.

    Each :meth:`record` call appends one JSON object to the journal file
    with shape::

        {
            "ts": int,                    # unix seconds
            "agent_id": str,              # the local agent recording the decision
            "message_id": str,            # the A2A message id
            "from_agent_id": str,
            "decision": str,              # one of ALL_DECISIONS
            "reason": Optional[str],
            "extra": dict                 # caller-supplied details
        }

    Reads via :meth:`read` are best-effort and skip malformed lines.
    """

    def __init__(self, path: str):
        self.path = Path(os.path.expanduser(path))
        self.path.parent.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_env(cls) -> "MessageJournal":
        return cls(path=os.getenv("EPHAPSYS_A2A_JOURNAL_PATH", DEFAULT_JOURNAL_PATH))

    def record(
        self,
        *,
        agent_id: str,
        message_id: str,
        decision: str,
        from_agent_id: str = "",
        reason: Optional[str] = None,
        extra: Optional[Dict[str, Any]] = None,
        ts: Optional[int] = None,
    ) -> Dict[str, Any]:
        if decision not in ALL_DECISIONS:
            raise ValueError(
                f"unknown decision '{decision}'; expected one of {sorted(ALL_DECISIONS)}"
            )
        entry: Dict[str, Any] = {
            "ts": int(ts if ts is not None else time.time()),
            "agent_id": str(agent_id or ""),
            "message_id": str(message_id or ""),
            "from_agent_id": str(from_agent_id or ""),
            "decision": decision,
            "reason": reason,
            "extra": dict(extra or {}),
        }
        line = json.dumps(entry, separators=(",", ":"), sort_keys=True)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")
        return entry

    def read(self, *, since_ts: Optional[int] = None) -> Iterator[Dict[str, Any]]:
        if not self.path.exists():
            return
        with self.path.open("r", encoding="utf-8") as fh:
            for raw in fh:
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    entry = json.loads(raw)
                except Exception:
                    continue
                if since_ts is not None and int(entry.get("ts", 0)) < since_ts:
                    continue
                yield entry

    def count_by_decision(self) -> Dict[str, int]:
        counts: Dict[str, int] = {d: 0 for d in ALL_DECISIONS}
        for entry in self.read():
            d = entry.get("decision")
            if isinstance(d, str) and d in counts:
                counts[d] += 1
        return counts


def consume(journal: MessageJournal, *, since_ts: Optional[int] = None) -> Iterable[Dict[str, Any]]:
    """Convenience wrapper to iterate journal entries (e.g., for batch sync)."""
    return journal.read(since_ts=since_ts)
