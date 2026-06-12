# SPDX-License-Identifier: Apache-2.0
"""Per-agent peer authentication for decoupled transports (issue #126, #1).

Lets one agent authenticate that a request received over its *own* transport
(e.g. Graham's direct edge↔cloud HTTP) genuinely came from a specific peer
agent — beyond the status/kill-switch gate in ``a2a.A2AClient`` — by signing
with the agent's **EC identity key** (the same ``kem_priv.pem`` /
``ECDSA(SHA256)`` keypair used for device auth in ``auth.py``) and verifying
against the peer's public key.

Design reviewed before implementation (see platform #126). This module is the
reviewed crypto core: canonicalization, ECDSA sign/verify, and replay
protection (timestamp skew window + nonce cache).

**Trust-source boundary (read this):** verification checks that the request was
signed by the private key matching the ``sender_public_key`` you pass in. It
does *not* by itself prove that key belongs to ``x-ep-agent`` — that binding is
the caller's responsibility and must come from a trusted source: fetch the
peer's X.509 instance cert from the PKI (``GET /certificates?agent_id=``),
validate the chain to the Ephapsys Root CA and that it isn't revoked
(OCSP/CRL), then pass its public key here. ``public_key_from_cert_pem`` helps
extract the key; chain/revocation validation against your PKI is intentionally
left to the caller (and is the piece that benefits from a confirmed
cert↔identity-key linkage on the backend).
"""

from __future__ import annotations

import base64
import dataclasses
import json
import secrets
import time
from typing import Any, Dict, Optional, Set

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec

# Header names for the peer-auth envelope.
H_AGENT = "x-ep-agent"
H_TS = "x-ep-ts"
H_NONCE = "x-ep-nonce"
H_SIG = "x-ep-sig"

DEFAULT_MAX_SKEW_SECONDS = 300


@dataclasses.dataclass
class PeerVerifyResult:
    ok: bool
    reason: Optional[str]
    agent_id: Optional[str]


def _canonical(ts: int, nonce: str, method: str, path: str, agent_id: str, body: Any) -> bytes:
    """Deterministic byte string the signature covers.

    Order and serialization are fixed so signer and verifier produce identical
    bytes. Body is canonical JSON (sorted keys, no whitespace).
    """
    return "\n".join(
        [
            str(int(ts)),
            nonce,
            (method or "").upper(),
            path or "",
            agent_id or "",
            json.dumps(body if body is not None else {}, separators=(",", ":"), sort_keys=True, default=str),
        ]
    ).encode("utf-8")


def public_key_from_cert_pem(pem: bytes | str) -> ec.EllipticCurvePublicKey:
    """Extract the EC public key from an X.509 certificate PEM.

    The caller is responsible for having validated the cert's chain and
    revocation status against the PKI *before* trusting this key.
    """
    from cryptography import x509

    if isinstance(pem, str):
        pem = pem.encode("utf-8")
    cert = x509.load_pem_x509_certificate(pem)
    key = cert.public_key()
    if not isinstance(key, ec.EllipticCurvePublicKey):
        raise ValueError("certificate public key is not EC")
    return key


def public_key_from_pem(pem: bytes | str) -> ec.EllipticCurvePublicKey:
    """Load a bare EC public key PEM (e.g. the peer's ``kem_pub.pem``)."""
    if isinstance(pem, str):
        pem = pem.encode("utf-8")
    key = serialization.load_pem_public_key(pem)
    if not isinstance(key, ec.EllipticCurvePublicKey):
        raise ValueError("public key is not EC")
    return key


def sign_peer_request(
    *,
    agent_id: str,
    method: str,
    path: str,
    body: Any,
    storage_dir: Optional[str] = None,
    private_key: Optional[ec.EllipticCurvePrivateKey] = None,
    ts: Optional[int] = None,
    nonce: Optional[str] = None,
) -> Dict[str, str]:
    """Sign an outbound peer request with this agent's EC identity key.

    Returns the headers to attach to the request. ``private_key`` may be passed
    directly (tests / custom key handling); otherwise the durable identity key
    is loaded via ``auth._load_identity_private_key`` from ``storage_dir``.
    """
    if not agent_id:
        raise ValueError("agent_id is required")
    if private_key is None:
        from .auth import _load_identity_private_key
        private_key = _load_identity_private_key(storage_dir)
    ts = int(ts if ts is not None else time.time())
    nonce = nonce or secrets.token_hex(16)
    sig = private_key.sign(_canonical(ts, nonce, method, path, agent_id, body), ec.ECDSA(hashes.SHA256()))
    return {
        H_AGENT: agent_id,
        H_TS: str(ts),
        H_NONCE: nonce,
        H_SIG: base64.b64encode(sig).decode("ascii"),
    }


def verify_peer_request(
    *,
    headers: Dict[str, str],
    method: str,
    path: str,
    body: Any,
    sender_public_key: ec.EllipticCurvePublicKey,
    expected_agent_id: Optional[str] = None,
    max_skew_seconds: int = DEFAULT_MAX_SKEW_SECONDS,
    now: Optional[int] = None,
    seen_nonces: Optional[Set[str]] = None,
) -> PeerVerifyResult:
    """Verify a peer-signed request. Fail-closed on every check.

    Checks, in order: required headers present; ``expected_agent_id`` match (if
    given); timestamp within ``±max_skew_seconds``; nonce not already seen (when
    a ``seen_nonces`` set is supplied — replay protection); ECDSA signature
    valid against ``sender_public_key`` over the reconstructed canonical bytes.
    On success, records the nonce in ``seen_nonces``.

    See the module docstring for the trust-source boundary: ``sender_public_key``
    must be obtained and validated from the PKI by the caller.
    """
    # Header presence
    agent_id = headers.get(H_AGENT, "")
    ts_s = headers.get(H_TS, "")
    nonce = headers.get(H_NONCE, "")
    sig_b64 = headers.get(H_SIG, "")
    if not (agent_id and ts_s and nonce and sig_b64):
        return PeerVerifyResult(False, "missing_auth_headers", agent_id or None)

    if expected_agent_id is not None and agent_id != expected_agent_id:
        return PeerVerifyResult(False, "agent_id_mismatch", agent_id)

    # Timestamp freshness (anti-replay window)
    try:
        ts = int(ts_s)
    except (TypeError, ValueError):
        return PeerVerifyResult(False, "bad_timestamp", agent_id)
    now = int(now if now is not None else time.time())
    if abs(now - ts) > max_skew_seconds:
        return PeerVerifyResult(False, "timestamp_out_of_window", agent_id)

    # Nonce replay
    if seen_nonces is not None and nonce in seen_nonces:
        return PeerVerifyResult(False, "replayed_nonce", agent_id)

    # Signature
    try:
        sig = base64.b64decode(sig_b64)
    except Exception:
        return PeerVerifyResult(False, "bad_signature_encoding", agent_id)
    try:
        sender_public_key.verify(sig, _canonical(ts, nonce, method, path, agent_id, body), ec.ECDSA(hashes.SHA256()))
    except InvalidSignature:
        return PeerVerifyResult(False, "bad_signature", agent_id)
    except Exception:
        return PeerVerifyResult(False, "verify_error", agent_id)

    if seen_nonces is not None:
        seen_nonces.add(nonce)
    return PeerVerifyResult(True, None, agent_id)
