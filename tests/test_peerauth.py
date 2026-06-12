#!/usr/bin/env python3
"""Tests for per-agent peer authentication (issue #126, #1).

Offline — generates a real EC keypair, no AOC/torch. Covers happy path and
every fail-closed branch (tamper, expiry, replay, wrong key, agent mismatch,
missing headers).

Usage:
    PYTHONPATH=sdk/python python3 tests/test_peerauth.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "sdk", "python"))

from cryptography.hazmat.primitives.asymmetric import ec  # noqa: E402

from ephapsys import peerauth  # noqa: E402


def _keypair():
    priv = ec.generate_private_key(ec.SECP256R1())
    return priv, priv.public_key()


REQ = dict(agent_id="edge-1", method="POST", path="/infer", body={"input": "hello", "k": 1})


def test_sign_then_verify_ok():
    priv, pub = _keypair()
    h = peerauth.sign_peer_request(private_key=priv, **REQ)
    r = peerauth.verify_peer_request(headers=h, method="POST", path="/infer", body=REQ["body"], sender_public_key=pub, expected_agent_id="edge-1")
    assert r.ok and r.reason is None and r.agent_id == "edge-1", r


def test_tampered_body_fails():
    priv, pub = _keypair()
    h = peerauth.sign_peer_request(private_key=priv, **REQ)
    r = peerauth.verify_peer_request(headers=h, method="POST", path="/infer", body={"input": "HELLO", "k": 1}, sender_public_key=pub)
    assert not r.ok and r.reason == "bad_signature", r


def test_tampered_path_fails():
    priv, pub = _keypair()
    h = peerauth.sign_peer_request(private_key=priv, **REQ)
    r = peerauth.verify_peer_request(headers=h, method="POST", path="/other", body=REQ["body"], sender_public_key=pub)
    assert not r.ok and r.reason == "bad_signature", r


def test_wrong_key_fails():
    priv, _ = _keypair()
    _, other_pub = _keypair()
    h = peerauth.sign_peer_request(private_key=priv, **REQ)
    r = peerauth.verify_peer_request(headers=h, method="POST", path="/infer", body=REQ["body"], sender_public_key=other_pub)
    assert not r.ok and r.reason == "bad_signature", r


def test_expired_timestamp_fails():
    priv, pub = _keypair()
    h = peerauth.sign_peer_request(private_key=priv, ts=1000, **REQ)
    r = peerauth.verify_peer_request(headers=h, method="POST", path="/infer", body=REQ["body"], sender_public_key=pub, now=1000 + 10_000, max_skew_seconds=300)
    assert not r.ok and r.reason == "timestamp_out_of_window", r


def test_future_timestamp_fails():
    priv, pub = _keypair()
    h = peerauth.sign_peer_request(private_key=priv, ts=1_000_000, **REQ)
    r = peerauth.verify_peer_request(headers=h, method="POST", path="/infer", body=REQ["body"], sender_public_key=pub, now=1000, max_skew_seconds=300)
    assert not r.ok and r.reason == "timestamp_out_of_window", r


def test_replayed_nonce_fails():
    priv, pub = _keypair()
    seen = set()
    h = peerauth.sign_peer_request(private_key=priv, **REQ)
    first = peerauth.verify_peer_request(headers=h, method="POST", path="/infer", body=REQ["body"], sender_public_key=pub, seen_nonces=seen)
    assert first.ok, first
    replay = peerauth.verify_peer_request(headers=h, method="POST", path="/infer", body=REQ["body"], sender_public_key=pub, seen_nonces=seen)
    assert not replay.ok and replay.reason == "replayed_nonce", replay


def test_agent_id_mismatch_fails():
    priv, pub = _keypair()
    h = peerauth.sign_peer_request(private_key=priv, **REQ)
    r = peerauth.verify_peer_request(headers=h, method="POST", path="/infer", body=REQ["body"], sender_public_key=pub, expected_agent_id="someone-else")
    assert not r.ok and r.reason == "agent_id_mismatch", r


def test_missing_headers_fails():
    priv, pub = _keypair()
    r = peerauth.verify_peer_request(headers={}, method="POST", path="/infer", body=REQ["body"], sender_public_key=pub)
    assert not r.ok and r.reason == "missing_auth_headers", r


def test_public_key_from_cert_pem_roundtrip():
    # Build a minimal self-signed cert around an EC key and recover its pubkey.
    import datetime
    from cryptography import x509
    from cryptography.x509.oid import NameOID
    from cryptography.hazmat.primitives import hashes

    priv, pub = _keypair()
    subject = issuer = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "edge-1")])
    cert = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(issuer)
        .public_key(pub)
        .serial_number(x509.random_serial_number())
        .not_valid_before(datetime.datetime(2020, 1, 1))
        .not_valid_after(datetime.datetime(2030, 1, 1))
        .sign(priv, hashes.SHA256())
    )
    pem = cert.public_bytes(__import__("cryptography").hazmat.primitives.serialization.Encoding.PEM)
    recovered = peerauth.public_key_from_cert_pem(pem)
    # Sign with priv, verify with the cert-recovered pubkey -> proves linkage.
    h = peerauth.sign_peer_request(private_key=priv, **REQ)
    r = peerauth.verify_peer_request(headers=h, method="POST", path="/infer", body=REQ["body"], sender_public_key=recovered)
    assert r.ok, r


def _main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"FAIL {t.__name__}: {e!r}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_main())
