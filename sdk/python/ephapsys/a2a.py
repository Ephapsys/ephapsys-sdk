# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import dataclasses
import hashlib
import hmac
import json
import logging
import os
import re
import secrets
import time
from typing import Any, Callable, Dict, Iterable, List, Optional

import requests

from .journal import (
    DECISION_GUARDRAIL_BLOCKED,
    DECISION_QUARANTINE_ALERT,
    DECISION_REJECTED,
    DECISION_SYSTEM_EVENT,
    DECISION_VERIFIED,
    MessageJournal,
)


logger = logging.getLogger("ephapsys.sdk")


SYSTEM_SENDER = "__system__"
SYSTEM_QUARANTINE_TYPE = "system.message_quarantine"
SYSTEM_STATUS_CHANGE_TYPE = "system.status_change"

# Sender statuses considered safe to deliver inbound messages from.
ACCEPTED_SENDER_STATUSES = frozenset({"ENABLED", "REGISTERED", "PERSONALIZED"})


@dataclasses.dataclass
class VerifiedMessage:
    """Result of verifying an inbound A2A message.

    Attributes
    ----------
    message:
        The original message dict as returned by ``A2AClient.inbox``.
    verified:
        ``True`` only if the sender check and (when requested) the
        payload guardrail scan both pass.
    reason:
        Short machine-readable reason when ``verified`` is ``False``
        (e.g. ``"sender_revoked"``, ``"sender_disabled"``,
        ``"sender_lookup_failed"``, ``"guardrail_blocked"``). ``None``
        when verified.
    sender_status:
        Upper-cased sender status as reported by the AOC, when known.
    is_system:
        ``True`` for ``system.*`` messages from the platform sender. The
        sender check is skipped for these and ``verified`` is always
        ``True``; callers typically dispatch them through dedicated
        callbacks rather than the regular message path.
    guardrail_hits:
        List of ``{"pattern": str, "snippet": str}`` records when the
        payload triggered a guardrail. Empty when the scan was skipped
        or no patterns matched.
    """

    message: Dict[str, Any]
    verified: bool
    reason: Optional[str]
    sender_status: Optional[str]
    is_system: bool
    guardrail_hits: List[Dict[str, Any]]


def _walk_strings(obj: Any) -> Iterable[str]:
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from _walk_strings(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            yield from _walk_strings(v)


def _payload_injection_hits(payload: Any) -> List[Dict[str, Any]]:
    """Scan string fields in ``payload`` for known prompt-injection patterns.

    Reuses the patterns shipped with the SDK's inference guardrails so
    the same defenses applied to model inputs are also applied to A2A
    message contents.
    """
    try:
        from .agent import PROMPT_INJECTION_PATTERNS  # type: ignore[attr-defined]
    except Exception:
        return []
    hits: List[Dict[str, Any]] = []
    for value in _walk_strings(payload):
        for pattern in PROMPT_INJECTION_PATTERNS:
            try:
                if re.search(pattern, value, flags=re.IGNORECASE):
                    hits.append({"pattern": pattern, "snippet": value[:120]})
            except re.error:
                continue
    return hits


class A2AClient:
    """
    Minimal Agent-to-Agent API client.

    Authentication:
      - preferred: AOC_A2A_TOKEN (Bearer)
      - fallback: explicit token argument
    """

    def __init__(
        self,
        base_url: str,
        token: Optional[str] = None,
        timeout: float = 30.0,
        *,
        org_id: Optional[str] = None,
        sign_requests: bool = False,
        hmac_secret: Optional[str] = None,
    ):
        self.base_url = (base_url or "").rstrip("/")
        if not self.base_url:
            raise RuntimeError("A2AClient requires a non-empty base_url")
        self.token = token or os.getenv("AOC_A2A_TOKEN", "") or os.getenv("AOC_MODULATION_TOKEN", "")
        if not self.token:
            raise RuntimeError("Missing token. Provide token or set AOC_A2A_TOKEN.")
        self.timeout = float(timeout)
        self.org_id = (org_id or os.getenv("AOC_ORG_ID", "")).strip()
        self.sign_requests = bool(sign_requests)
        self.hmac_secret = hmac_secret or os.getenv("A2A_HMAC_SECRET", "")
        if self.sign_requests and not self.org_id:
            raise RuntimeError("A2A signed mode requires AOC_ORG_ID (or org_id constructor argument).")
        if self.sign_requests and not self.hmac_secret:
            raise RuntimeError("A2A signed mode requires A2A_HMAC_SECRET (or hmac_secret constructor argument).")

    @classmethod
    def from_env(cls) -> "A2AClient":
        base_url = os.getenv("AOC_BASE_URL") or os.getenv("AOC_API_URL") or "http://localhost:7001"
        token = os.getenv("AOC_A2A_TOKEN", "") or os.getenv("AOC_MODULATION_TOKEN", "")
        timeout = float(os.getenv("AOC_HTTP_TIMEOUT", "30"))
        org_id = os.getenv("AOC_ORG_ID", "")
        sign_requests = os.getenv("A2A_SIGN_REQUESTS", "0") == "1"
        hmac_secret = os.getenv("A2A_HMAC_SECRET", "")
        return cls(
            base_url=base_url,
            token=token,
            timeout=timeout,
            org_id=org_id,
            sign_requests=sign_requests,
            hmac_secret=hmac_secret,
        )

    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json",
        }

    @staticmethod
    def _canonical_send_payload(body: Dict[str, Any]) -> str:
        payload = {
            "from_agent_id": body.get("from_agent_id", ""),
            "to_agent_id": body.get("to_agent_id", ""),
            "message_type": body.get("message_type", "event"),
            "payload": body.get("payload", {}) or {},
            "correlation_id": body.get("correlation_id", "") or "",
            "ttl_seconds": int(body.get("ttl_seconds", 0) or 0),
        }
        return json.dumps(payload, separators=(",", ":"), sort_keys=True)

    def _signed_headers(self, *, method: str, path: str, body: Dict[str, Any]) -> Dict[str, str]:
        headers = self._headers()
        if not self.sign_requests:
            return headers

        ts = int(time.time())
        nonce = secrets.token_hex(16)
        canonical = "\n".join(
            [
                str(ts),
                nonce,
                method.upper(),
                path,
                self.org_id,
                self._canonical_send_payload(body),
            ]
        )
        sig = hmac.new(self.hmac_secret.encode("utf-8"), canonical.encode("utf-8"), hashlib.sha256).hexdigest()
        headers["x-a2a-ts"] = str(ts)
        headers["x-a2a-nonce"] = nonce
        headers["x-a2a-sig"] = sig
        return headers

    def send_message(
        self,
        *,
        from_agent_id: str,
        to_agent_id: str,
        payload: Dict[str, Any],
        message_type: str = "event",
        correlation_id: Optional[str] = None,
        ttl_seconds: Optional[int] = None,
    ) -> Dict[str, Any]:
        body: Dict[str, Any] = {
            "from_agent_id": from_agent_id,
            "to_agent_id": to_agent_id,
            "payload": payload or {},
            "message_type": message_type or "event",
        }
        if correlation_id:
            body["correlation_id"] = correlation_id
        if ttl_seconds is not None:
            body["ttl_seconds"] = int(ttl_seconds)

        path = "/a2a/messages"
        r = requests.post(
            f"{self.base_url}{path}",
            headers=self._signed_headers(method="POST", path=path, body=body),
            json=body,
            timeout=self.timeout,
        )
        if not r.ok:
            raise RuntimeError(f"A2A send failed: {r.status_code} {r.text}")
        return r.json()

    def inbox(self, *, agent_id: str, limit: int = 50, include_acked: bool = False) -> Dict[str, Any]:
        r = requests.get(
            f"{self.base_url}/a2a/messages/inbox",
            headers=self._headers(),
            params={"agent_id": agent_id, "limit": int(limit), "include_acked": bool(include_acked)},
            timeout=self.timeout,
        )
        if not r.ok:
            raise RuntimeError(f"A2A inbox failed: {r.status_code} {r.text}")
        return r.json()

    def ack_message(self, *, message_id: str, agent_id: str) -> Dict[str, Any]:
        r = requests.post(
            f"{self.base_url}/a2a/messages/{message_id}/ack",
            headers=self._headers(),
            json={"agent_id": agent_id},
            timeout=self.timeout,
        )
        if not r.ok:
            raise RuntimeError(f"A2A ack failed: {r.status_code} {r.text}")
        return r.json()

    # -- Inbound verification -------------------------------------------------

    def _check_sender_status(self, sender_agent_id: str) -> Dict[str, Any]:
        """Look up the sender's current status via the AOC.

        Returns ``{"ok": bool, "status": Optional[str], "state": dict,
        "error": Optional[str]}``. Does not raise on HTTP failure — the
        caller decides how to treat lookup errors (this implementation
        treats them as fail-closed: the message is rejected with a
        ``sender_lookup_failed`` reason).
        """
        try:
            r = requests.get(
                f"{self.base_url}/agents/{sender_agent_id}/status",
                headers=self._headers(),
                timeout=self.timeout,
            )
        except requests.RequestException as exc:
            return {"ok": False, "status": None, "state": {}, "error": f"network: {exc}"}
        if not r.ok:
            return {
                "ok": False,
                "status": None,
                "state": {},
                "error": f"http {r.status_code}",
            }
        try:
            data = r.json() or {}
        except ValueError:
            return {"ok": False, "status": None, "state": {}, "error": "invalid_json"}
        return {
            "ok": True,
            "status": str(data.get("status") or "").upper(),
            "state": data.get("state") or {},
            "error": None,
        }

    def verify_message(
        self,
        message: Dict[str, Any],
        *,
        scan_guardrails: bool = True,
    ) -> VerifiedMessage:
        """Verify a single inbound A2A message.

        For ``system.*`` messages from the platform sender, the sender
        check is skipped and ``verified`` is ``True`` (callers dispatch
        them through dedicated handlers).

        For agent-to-agent messages this performs:

        1. **Sender status check** via ``GET /agents/{id}/status``.
           Messages from senders whose current status is not in
           :data:`ACCEPTED_SENDER_STATUSES` (e.g. ``REVOKED`` /
           ``DISABLED``) are rejected with a structured reason. Lookup
           errors are treated as fail-closed.
        2. **Payload guardrail scan** (when ``scan_guardrails`` is true)
           against the SDK's ``PROMPT_INJECTION_PATTERNS``. Any hit
           causes rejection with ``reason="guardrail_blocked"``.
        """
        msg_type = (message.get("message_type") or "").lower()
        sender = str(message.get("from_agent_id") or "")

        if sender == SYSTEM_SENDER or msg_type.startswith("system."):
            return VerifiedMessage(
                message=message,
                verified=True,
                reason=None,
                sender_status=None,
                is_system=True,
                guardrail_hits=[],
            )

        if not sender:
            return VerifiedMessage(
                message=message,
                verified=False,
                reason="missing_sender",
                sender_status=None,
                is_system=False,
                guardrail_hits=[],
            )

        lookup = self._check_sender_status(sender)
        if not lookup["ok"]:
            return VerifiedMessage(
                message=message,
                verified=False,
                reason="sender_lookup_failed",
                sender_status=None,
                is_system=False,
                guardrail_hits=[],
            )
        sender_status = lookup["status"] or ""
        state = lookup["state"] or {}
        if state.get("revoked") or sender_status == "REVOKED":
            return VerifiedMessage(
                message=message,
                verified=False,
                reason="sender_revoked",
                sender_status="REVOKED",
                is_system=False,
                guardrail_hits=[],
            )
        if sender_status not in ACCEPTED_SENDER_STATUSES:
            return VerifiedMessage(
                message=message,
                verified=False,
                reason=f"sender_status_{sender_status.lower() or 'unknown'}",
                sender_status=sender_status or None,
                is_system=False,
                guardrail_hits=[],
            )

        hits: List[Dict[str, Any]] = []
        if scan_guardrails:
            hits = _payload_injection_hits(message.get("payload"))
            if hits:
                return VerifiedMessage(
                    message=message,
                    verified=False,
                    reason="guardrail_blocked",
                    sender_status=sender_status,
                    is_system=False,
                    guardrail_hits=hits,
                )

        return VerifiedMessage(
            message=message,
            verified=True,
            reason=None,
            sender_status=sender_status,
            is_system=False,
            guardrail_hits=[],
        )

    def process_inbox(
        self,
        *,
        agent_id: str,
        on_verified: Optional[Callable[[Dict[str, Any]], None]] = None,
        on_rejected: Optional[Callable[[VerifiedMessage], None]] = None,
        on_quarantine_alert: Optional[Callable[[Dict[str, Any]], None]] = None,
        on_status_change: Optional[Callable[[Dict[str, Any]], None]] = None,
        journal: Optional[MessageJournal] = None,
        apply_guardrails: bool = True,
        ack_after_process: bool = True,
        ack_rejected: bool = False,
        limit: int = 50,
    ) -> Dict[str, int]:
        """Fetch, verify, and dispatch inbox messages.

        Walks the inbox once, classifying each message and dispatching
        through the matching callback. System messages (``system.*``) are
        delivered to ``on_quarantine_alert`` / ``on_status_change``
        without going through sender verification. Non-system messages
        are run through :meth:`verify_message`; verified messages reach
        ``on_verified``, rejected messages reach ``on_rejected`` (and are
        not acked unless ``ack_rejected`` is set, so the recipient can
        re-pull them on a later run if desired).

        Each outcome is recorded in ``journal`` when one is supplied.

        Returns a summary dict::

            {
                "processed": int,
                "verified": int,
                "rejected": int,
                "guardrail_blocked": int,
                "quarantine_alerts": int,
                "status_events": int,
            }
        """
        summary = {
            "processed": 0,
            "verified": 0,
            "rejected": 0,
            "guardrail_blocked": 0,
            "quarantine_alerts": 0,
            "status_events": 0,
        }

        inbox_response = self.inbox(agent_id=agent_id, limit=limit)
        items = inbox_response.get("items") or []

        for msg in items:
            summary["processed"] += 1
            msg_id = str(msg.get("id") or "")
            from_ref = str(msg.get("from_agent_id") or "")
            msg_type = (msg.get("message_type") or "").lower()

            if msg_type == SYSTEM_QUARANTINE_TYPE:
                summary["quarantine_alerts"] += 1
                if on_quarantine_alert is not None:
                    try:
                        on_quarantine_alert(msg)
                    except Exception:
                        logger.exception("on_quarantine_alert callback failed")
                if journal is not None:
                    journal.record(
                        agent_id=agent_id,
                        message_id=msg_id,
                        from_agent_id=from_ref,
                        decision=DECISION_QUARANTINE_ALERT,
                        extra={"payload": msg.get("payload")},
                    )
                if ack_after_process:
                    self._safe_ack(message_id=msg_id, agent_id=agent_id)
                continue

            if msg_type == SYSTEM_STATUS_CHANGE_TYPE:
                summary["status_events"] += 1
                if on_status_change is not None:
                    try:
                        on_status_change(msg)
                    except Exception:
                        logger.exception("on_status_change callback failed")
                if journal is not None:
                    journal.record(
                        agent_id=agent_id,
                        message_id=msg_id,
                        from_agent_id=from_ref,
                        decision=DECISION_SYSTEM_EVENT,
                        extra={"payload": msg.get("payload")},
                    )
                if ack_after_process:
                    self._safe_ack(message_id=msg_id, agent_id=agent_id)
                continue

            verification = self.verify_message(msg, scan_guardrails=apply_guardrails)
            if verification.verified:
                summary["verified"] += 1
                if on_verified is not None:
                    try:
                        on_verified(msg)
                    except Exception:
                        logger.exception("on_verified callback failed")
                if journal is not None:
                    journal.record(
                        agent_id=agent_id,
                        message_id=msg_id,
                        from_agent_id=from_ref,
                        decision=DECISION_VERIFIED,
                        extra={"sender_status": verification.sender_status},
                    )
                if ack_after_process:
                    self._safe_ack(message_id=msg_id, agent_id=agent_id)
                continue

            # Rejected path
            if verification.guardrail_hits:
                summary["guardrail_blocked"] += 1
                decision = DECISION_GUARDRAIL_BLOCKED
            else:
                summary["rejected"] += 1
                decision = DECISION_REJECTED

            if on_rejected is not None:
                try:
                    on_rejected(verification)
                except Exception:
                    logger.exception("on_rejected callback failed")
            if journal is not None:
                journal.record(
                    agent_id=agent_id,
                    message_id=msg_id,
                    from_agent_id=from_ref,
                    decision=decision,
                    reason=verification.reason,
                    extra={
                        "sender_status": verification.sender_status,
                        "guardrail_hits": verification.guardrail_hits,
                    },
                )
            if ack_rejected:
                self._safe_ack(message_id=msg_id, agent_id=agent_id)

        return summary

    def _safe_ack(self, *, message_id: str, agent_id: str) -> None:
        try:
            self.ack_message(message_id=message_id, agent_id=agent_id)
        except Exception as exc:
            logger.warning("a2a ack failed for message_id=%s: %s", message_id, exc)
