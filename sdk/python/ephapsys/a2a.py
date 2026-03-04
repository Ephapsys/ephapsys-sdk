# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import hashlib
import hmac
import json
import os
import secrets
import time
from typing import Any, Dict, Optional

import requests


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
