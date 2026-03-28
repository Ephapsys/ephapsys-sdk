# SPDX-License-Identifier: Apache-2.0
import os
import time
import base64
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import requests
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec

_TOKEN_CACHE: Dict[Tuple[str, str, str, str], Tuple[str, float]] = {}


def _truthy(name: str, default: str = "1") -> bool:
    return os.getenv(name, default).strip().lower() not in ("0", "false", "no", "")


def _exchange_provisioning_token(
    *,
    base_url: str,
    org_id: str,
    provisioning_token: str,
    device_id: Optional[str] = None,
    agent_instance_id: Optional[str] = None,
    verify_ssl: bool = True,
) -> str:
    key = (
        base_url.rstrip("/"),
        org_id,
        provisioning_token,
        device_id or "",
    )
    now = time.time()
    cached = _TOKEN_CACHE.get(key)
    if cached and cached[1] > now + 15:
        return cached[0]

    url = f"{base_url.rstrip('/')}/auth/device/token"
    body = {
        "org_id": org_id,
        "provisioning_token": provisioning_token,
        "device_id": device_id,
        "agent_instance_id": agent_instance_id,
    }
    try:
        resp = requests.post(url, json=body, timeout=15, verify=verify_ssl)
    except requests.RequestException as e:
        raise RuntimeError(f"Failed to exchange provisioning token at {url}: {e}") from e

    if resp.status_code != 200:
        detail = ""
        try:
            detail = (resp.json() or {}).get("detail", "")
        except Exception:
            pass
        raise RuntimeError(
            f"Provisioning token exchange failed ({resp.status_code})"
            + (f": {detail}" if detail else "")
        )

    data = resp.json()
    token = data.get("access_token")
    if not token:
        raise RuntimeError("Provisioning token exchange succeeded but access_token missing")

    ttl = int(data.get("expires_in") or 900)
    _TOKEN_CACHE[key] = (token, now + max(30, ttl))
    return token


def _storage_root(storage_dir: Optional[str]) -> Path:
    return Path(storage_dir or os.getenv("EPHAPSYS_STORAGE_DIR", ".ephapsys_state"))


def _identity_key_paths(storage_dir: Optional[str]) -> Tuple[Path, Path]:
    root = _storage_root(storage_dir) / "kem"
    return root / "kem_priv.pem", root / "kem_pub.pem"


def _load_identity_private_key(storage_dir: Optional[str]) -> ec.EllipticCurvePrivateKey:
    priv_path, _ = _identity_key_paths(storage_dir)
    if not priv_path.exists():
        raise RuntimeError(f"Durable identity key not found at {priv_path}")
    pem = priv_path.read_bytes()
    key = serialization.load_pem_private_key(pem, password=None)
    if not isinstance(key, ec.EllipticCurvePrivateKey):
        raise RuntimeError("Durable identity key must be EC")
    return key


def _issue_identity_challenge(
    *,
    base_url: str,
    org_id: Optional[str],
    device_id: str,
    agent_instance_id: str,
    verify_ssl: bool,
) -> Dict[str, Any]:
    url = f"{base_url.rstrip('/')}/auth/device/challenge"
    body = {
        "org_id": org_id,
        "device_id": device_id,
        "agent_instance_id": agent_instance_id,
    }
    try:
        resp = requests.post(url, json=body, timeout=15, verify=verify_ssl)
    except requests.RequestException as e:
        raise RuntimeError(f"Failed to request device auth challenge at {url}: {e}") from e
    if resp.status_code != 200:
        detail = ""
        try:
            detail = (resp.json() or {}).get("detail", "")
        except Exception:
            pass
        raise RuntimeError(
            f"Device auth challenge failed ({resp.status_code})" + (f": {detail}" if detail else "")
        )
    return resp.json()


def _exchange_identity_token(
    *,
    base_url: str,
    org_id: Optional[str],
    device_id: str,
    agent_instance_id: str,
    storage_dir: Optional[str],
    verify_ssl: bool,
) -> str:
    key = (
        base_url.rstrip("/"),
        org_id or "",
        f"identity:{agent_instance_id}",
        device_id,
    )
    now = time.time()
    cached = _TOKEN_CACHE.get(key)
    if cached and cached[1] > now + 15:
        return cached[0]

    challenge = _issue_identity_challenge(
        base_url=base_url,
        org_id=org_id,
        device_id=device_id,
        agent_instance_id=agent_instance_id,
        verify_ssl=verify_ssl,
    )
    nonce_b64 = challenge.get("nonce_b64")
    if not nonce_b64:
        raise RuntimeError("Device auth challenge succeeded but nonce_b64 missing")

    message = f"ephapsys-device-auth-v1|{challenge.get('org_id') or org_id or ''}|{device_id}|{agent_instance_id}|{nonce_b64}".encode("utf-8")
    private_key = _load_identity_private_key(storage_dir)
    signature = private_key.sign(message, ec.ECDSA(hashes.SHA256()))

    url = f"{base_url.rstrip('/')}/auth/device/token"
    body = {
        "org_id": challenge.get("org_id") or org_id,
        "device_id": device_id,
        "agent_instance_id": agent_instance_id,
        "challenge_nonce_b64": nonce_b64,
        "challenge_signature_b64": base64.b64encode(signature).decode("ascii"),
    }
    try:
        resp = requests.post(url, json=body, timeout=15, verify=verify_ssl)
    except requests.RequestException as e:
        raise RuntimeError(f"Failed to exchange device identity token at {url}: {e}") from e
    if resp.status_code != 200:
        detail = ""
        try:
            detail = (resp.json() or {}).get("detail", "")
        except Exception:
            pass
        raise RuntimeError(
            f"Device identity token exchange failed ({resp.status_code})"
            + (f": {detail}" if detail else "")
        )

    data = resp.json()
    token = data.get("access_token")
    if not token:
        raise RuntimeError("Device identity token exchange succeeded but access_token missing")
    ttl = int(data.get("expires_in") or 900)
    _TOKEN_CACHE[key] = (token, now + max(30, ttl))
    return token


def check_helloworld_bootstrap(
    *,
    base_url: str,
    org_id: str,
    provisioning_token: str,
    agent_template_id: str,
    verify_ssl: bool = True,
) -> Dict[str, Any]:
    url = f"{base_url.rstrip('/')}/sdk/onboarding/helloworld/check"
    body = {
        "org_id": org_id,
        "provisioning_token": provisioning_token,
        "agent_template_id": agent_template_id,
    }
    try:
        resp = requests.post(url, json=body, timeout=15, verify=verify_ssl)
    except requests.RequestException as e:
        raise RuntimeError(f"Failed to run HelloWorld preflight at {url}: {e}") from e

    if resp.status_code != 200:
        detail = ""
        try:
            detail = (resp.json() or {}).get("detail", "")
        except Exception:
            detail = resp.text.strip()
        raise RuntimeError(
            f"HelloWorld preflight failed ({resp.status_code})"
            + (f": {detail}" if detail else "")
        )
    return resp.json()


def get_api_key(
    explicit: str = None,
    *,
    base_url: Optional[str] = None,
    org_id: Optional[str] = None,
    device_id: Optional[str] = None,
    agent_instance_id: Optional[str] = None,
    verify_ssl: Optional[bool] = None,
    storage_dir: Optional[str] = None,
) -> str:
    """
    Resolution order:
      1) Explicit api_key argument
      2) AOC_PROVISIONING_TOKEN exchange -> short-lived device token
    """
    if explicit:
        return explicit

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
    resolved_org = org_id or os.getenv("AOC_ORG_ID")

    identity_error: Optional[RuntimeError] = None
    identity_key_present = _identity_key_paths(storage_dir)[0].exists()
    if resolved_instance and identity_key_present:
        try:
            return _exchange_identity_token(
                base_url=resolved_base,
                org_id=resolved_org,
                device_id=resolved_device,
                agent_instance_id=resolved_instance,
                storage_dir=storage_dir,
                verify_ssl=resolved_verify,
            )
        except RuntimeError as exc:
            identity_error = exc

    provisioning_token = os.getenv("AOC_PROVISIONING_TOKEN")
    if provisioning_token:
        if not resolved_org:
            raise RuntimeError("Missing AOC_ORG_ID for provisioning token exchange")
        return _exchange_provisioning_token(
            base_url=resolved_base,
            org_id=resolved_org,
            provisioning_token=provisioning_token,
            device_id=resolved_device,
            agent_instance_id=resolved_instance,
            verify_ssl=resolved_verify,
        )

    if identity_error is not None:
        raise identity_error

    raise RuntimeError(
        "Missing credentials. Set AOC_ORG_ID + AOC_PROVISIONING_TOKEN."
    )
