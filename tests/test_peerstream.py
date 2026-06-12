#!/usr/bin/env python3
"""Tests for peer-direct governed streaming (issue #126, #2).

Offline loopback: the client signs (real #1 auth), a LoopbackSession runs the
server core over those exact headers/body, and the client parses the SSE back.
Exercises the happy path + auth failure + revoked-server (governance) + bad
caller-authorization.

Usage:
    PYTHONPATH=sdk/python python3 tests/test_peerstream.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "sdk", "python"))

from urllib.parse import urlsplit  # noqa: E402

from cryptography.hazmat.primitives.asymmetric import ec  # noqa: E402

from ephapsys import peerstream  # noqa: E402


class FakeAgent:
    def __init__(self, chunks, revoked=False):
        self._chunks = chunks
        self._revoked = revoked

    def run_stream(self, prompt, model_kind="language"):
        if self._revoked:
            raise RuntimeError("Agent revoked; inference blocked")
        for c in self._chunks:
            yield c


class FakeResp:
    def __init__(self, frames):
        self._lines = []
        for f in frames:
            self._lines.extend(f.split("\n"))

    def raise_for_status(self):
        return None

    def iter_lines(self, decode_unicode=True):
        for line in self._lines:
            yield line


class LoopbackSession:
    """Runs the server core in-process over the client's signed request."""

    def __init__(self, agent, resolve_public_key, authorize_caller=None):
        self.agent = agent
        self.resolve = resolve_public_key
        self.authz = authorize_caller
        self.seen = set()

    def post(self, url, json=None, headers=None, stream=False, timeout=None):
        path = urlsplit(url).path or "/"
        frames = list(
            peerstream.serve_inference_stream(
                agent=self.agent,
                headers=headers or {},
                method="POST",
                path=path,
                body=json,
                resolve_public_key=self.resolve,
                authorize_caller=self.authz,
                seen_nonces=self.seen,
            )
        )
        return FakeResp(frames)


def _keypair():
    priv = ec.generate_private_key(ec.SECP256R1())
    return priv, priv.public_key()


URL = "http://cloud-peer.local/infer"


def test_end_to_end_stream_ok():
    priv, pub = _keypair()
    agent = FakeAgent(["Hel", "lo", " world"])
    sess = LoopbackSession(agent, resolve_public_key=lambda aid: pub)
    toks = list(peerstream.stream_peer_inference(url=URL, agent_id="edge-1", prompt="hi", private_key=priv, session=sess))
    assert "".join(toks) == "Hello world", toks


def test_auth_failure_wrong_key_raises():
    priv, _ = _keypair()
    _, other_pub = _keypair()
    agent = FakeAgent(["x"])
    sess = LoopbackSession(agent, resolve_public_key=lambda aid: other_pub)  # wrong key
    try:
        list(peerstream.stream_peer_inference(url=URL, agent_id="edge-1", prompt="hi", private_key=priv, session=sess))
    except RuntimeError as e:
        assert "auth:bad_signature" in str(e), e
    else:
        raise AssertionError("expected auth failure")


def test_unresolvable_sender_raises():
    priv, _ = _keypair()
    agent = FakeAgent(["x"])
    sess = LoopbackSession(agent, resolve_public_key=lambda aid: None)
    try:
        list(peerstream.stream_peer_inference(url=URL, agent_id="edge-1", prompt="hi", private_key=priv, session=sess))
    except RuntimeError as e:
        assert "unresolvable_sender" in str(e), e
    else:
        raise AssertionError("expected unresolvable sender error")


def test_revoked_server_agent_surfaces_as_error():
    priv, pub = _keypair()
    agent = FakeAgent(["x"], revoked=True)  # server agent's run_stream fails closed
    sess = LoopbackSession(agent, resolve_public_key=lambda aid: pub)
    try:
        list(peerstream.stream_peer_inference(url=URL, agent_id="edge-1", prompt="hi", private_key=priv, session=sess))
    except RuntimeError as e:
        assert "revoked" in str(e).lower(), e
    else:
        raise AssertionError("expected governance (revoked) error")


def test_caller_not_authorized_raises():
    priv, pub = _keypair()
    agent = FakeAgent(["x"])
    sess = LoopbackSession(agent, resolve_public_key=lambda aid: pub, authorize_caller=lambda aid: False)
    try:
        list(peerstream.stream_peer_inference(url=URL, agent_id="edge-1", prompt="hi", private_key=priv, session=sess))
    except RuntimeError as e:
        assert "caller_not_authorized" in str(e), e
    else:
        raise AssertionError("expected caller-authorization failure")


def test_parse_sse_tokens_done_and_error():
    ok = list(peerstream.parse_sse_tokens(['data: {"type":"token","token":"a"}', "", 'data: {"done":true}']))
    assert ok == ["a"], ok
    try:
        list(peerstream.parse_sse_tokens(['data: {"type":"error","error":"boom"}']))
    except RuntimeError as e:
        assert "boom" in str(e)
    else:
        raise AssertionError("expected error to raise")


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
