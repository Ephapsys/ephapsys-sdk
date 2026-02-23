# SPDX-License-Identifier: Apache-2.0
import os
import time
from typing import Optional, Tuple, Dict

import requests

_TOKEN_CACHE: Dict[Tuple[str, str, str, str, str], Tuple[str, float]] = {}


def _truthy(name: str, default: str = "1") -> bool:
    return os.getenv(name, default).strip().lower() not in ("0", "false", "no", "")


def _exchange_bootstrap_token(
    *,
    base_url: str,
    org_id: str,
    bootstrap_token: str,
    device_id: Optional[str] = None,
    agent_instance_id: Optional[str] = None,
    verify_ssl: bool = True,
) -> str:
    key = (
        base_url.rstrip("/"),
        org_id,
        bootstrap_token,
        device_id or "",
        agent_instance_id or "",
    )
    now = time.time()
    cached = _TOKEN_CACHE.get(key)
    if cached and cached[1] > now + 15:
        return cached[0]

    url = f"{base_url.rstrip('/')}/auth/device/token"
    body = {
        "org_id": org_id,
        "bootstrap_token": bootstrap_token,
        "device_id": device_id,
        "agent_instance_id": agent_instance_id,
    }
    try:
        resp = requests.post(url, json=body, timeout=15, verify=verify_ssl)
    except requests.RequestException as e:
        raise RuntimeError(f"Failed to exchange bootstrap token at {url}: {e}") from e

    if resp.status_code != 200:
        detail = ""
        try:
            detail = (resp.json() or {}).get("detail", "")
        except Exception:
            pass
        raise RuntimeError(
            f"Bootstrap token exchange failed ({resp.status_code})"
            + (f": {detail}" if detail else "")
        )

    data = resp.json()
    token = data.get("access_token")
    if not token:
        raise RuntimeError("Bootstrap token exchange succeeded but access_token missing")

    ttl = int(data.get("expires_in") or 900)
    _TOKEN_CACHE[key] = (token, now + max(30, ttl))
    return token


def get_api_key(
    explicit: str = None,
    *,
    base_url: Optional[str] = None,
    org_id: Optional[str] = None,
    device_id: Optional[str] = None,
    agent_instance_id: Optional[str] = None,
    verify_ssl: Optional[bool] = None,
) -> str:
    """
    Resolution order:
      1) Explicit api_key argument
      2) AOC_BOOTSTRAP_TOKEN exchange -> short-lived device token
    """
    if explicit:
        return explicit

    bootstrap = os.getenv("AOC_BOOTSTRAP_TOKEN")
    if bootstrap:
        resolved_org = org_id or os.getenv("AOC_ORG_ID")
        if not resolved_org:
            raise RuntimeError("Missing AOC_ORG_ID for bootstrap token exchange")

        resolved_base = (
            base_url
            or os.getenv("AOC_BASE_URL")
            or os.getenv("AOC_API_URL")
            or "http://localhost:7001"
        )
        resolved_device = (
            device_id
            or os.getenv("EPHAPSYS_DEVICE_ID")
            or os.getenv("HOSTNAME")
            or "unknown-device"
        )
        resolved_instance = (
            agent_instance_id
            or os.getenv("EPHAPSYS_AGENT_ID")
            or os.getenv("AGENT_TEMPLATE_ID")
            or ""
        )
        resolved_verify = _truthy("AOC_VERIFY_SSL", "1") if verify_ssl is None else bool(verify_ssl)
        return _exchange_bootstrap_token(
            base_url=resolved_base,
            org_id=resolved_org,
            bootstrap_token=bootstrap,
            device_id=resolved_device,
            agent_instance_id=resolved_instance,
            verify_ssl=resolved_verify,
        )

    raise RuntimeError(
        "Missing credentials. Set AOC_ORG_ID + AOC_BOOTSTRAP_TOKEN."
    )
