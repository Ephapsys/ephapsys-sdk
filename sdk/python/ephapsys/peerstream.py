# SPDX-License-Identifier: Apache-2.0
"""Peer-direct governed inference streaming (issue #126, #2 — path B).

Streams ``TrustedAgent.run_stream()`` tokens from one agent to another over the
consumer's *own* HTTP transport (no AOC inbox/poll), authenticated per request
by :mod:`ephapsys.peerauth` (#1) and gated by the kill-switch. This is the
SDK-native form of Graham's hybrid edge↔cloud streaming.

Three governance layers apply to every request, fail-closed:
  1. **Peer identity** — ``peerauth.verify_peer_request`` (signature + replay).
  2. **Caller authorization** — optional ``authorize_caller(agent_id)`` (e.g.
     ``A2AClient.is_peer_authorized``) so a revoked *caller* is refused.
  3. **Server governance** — ``run_stream`` runs the fail-closed preflight, so a
     revoked/disabled *serving* agent surfaces as an error event, not tokens.

The HTTP wiring is intentionally thin; the core (:func:`serve_inference_stream`,
:func:`parse_sse_tokens`) is transport-agnostic and unit-testable without
sockets.
"""

from __future__ import annotations

import json
from typing import Any, Callable, Dict, Iterator, Optional, Set

from . import peerauth


def format_sse(event: Dict[str, Any]) -> str:
    """Serialize one event as an SSE ``data:`` frame."""
    return f"data: {json.dumps(event, default=str)}\n\n"


def parse_sse_tokens(lines: Iterator[Any]) -> Iterator[str]:
    """Yield token strings from an SSE line iterator.

    Stops on a ``{"done": true}`` frame; raises ``RuntimeError`` on an
    ``{"type": "error"}`` frame (so a governance/auth failure on the server
    propagates to the caller rather than silently truncating).
    """
    for line in lines:
        if not line:
            continue
        if isinstance(line, bytes):
            line = line.decode("utf-8", "replace")
        if not line.startswith("data:"):
            continue
        try:
            event = json.loads(line[len("data:"):].strip())
        except json.JSONDecodeError:
            continue
        etype = event.get("type")
        if etype == "token":
            tok = event.get("token", "")
            if tok:
                yield tok
        elif etype == "error":
            raise RuntimeError(f"peer stream error: {event.get('error')}")
        elif event.get("done"):
            return


def serve_inference_stream(
    *,
    agent: Any,
    headers: Dict[str, str],
    method: str,
    path: str,
    body: Any,
    resolve_public_key: Callable[[str], Any],
    authorize_caller: Optional[Callable[[str], bool]] = None,
    max_skew_seconds: int = peerauth.DEFAULT_MAX_SKEW_SECONDS,
    seen_nonces: Optional[Set[str]] = None,
    model_kind: str = "language",
    journal: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Iterator[str]:
    """Server core: authenticate + authorize + stream ``run_stream`` as SSE.

    ``resolve_public_key(agent_id)`` must return the caller's *trusted* EC
    public key (resolved/validated from the PKI by you — see
    :mod:`ephapsys.peerauth` trust-source boundary), or ``None`` to reject.
    Yields SSE frames; the first failing check yields a single ``error`` frame
    and stops.
    """
    sender = headers.get(peerauth.H_AGENT, "")
    pub = None
    if sender:
        try:
            pub = resolve_public_key(sender)
        except Exception:
            pub = None
    if pub is None:
        yield format_sse({"type": "error", "error": "unresolvable_sender"})
        return

    res = peerauth.verify_peer_request(
        headers=headers,
        method=method,
        path=path,
        body=body,
        sender_public_key=pub,
        expected_agent_id=sender,
        max_skew_seconds=max_skew_seconds,
        seen_nonces=seen_nonces,
    )
    if not res.ok:
        yield format_sse({"type": "error", "error": f"auth:{res.reason}"})
        return

    if authorize_caller is not None:
        try:
            allowed = bool(authorize_caller(sender))
        except Exception:
            allowed = False
        if not allowed:
            yield format_sse({"type": "error", "error": "caller_not_authorized"})
            return

    prompt = body.get("input") if isinstance(body, dict) else None
    if prompt is None:
        yield format_sse({"type": "error", "error": "missing_input"})
        return

    produced = 0
    try:
        for chunk in agent.run_stream(prompt, model_kind=model_kind):
            if chunk:
                produced += 1
                yield format_sse({"type": "token", "token": chunk})
    except Exception as exc:  # noqa: BLE001 - server governance (revoked/disabled) is fail-closed
        yield format_sse({"type": "error", "error": str(exc)})
        return

    yield format_sse({"done": True})
    if journal is not None:
        try:
            journal({"event": "peer_inference_stream", "from_agent_id": sender, "tokens": produced})
        except Exception:
            pass


def stream_peer_inference(
    *,
    url: str,
    agent_id: str,
    prompt: str,
    storage_dir: Optional[str] = None,
    private_key: Any = None,
    session: Any = None,
    timeout: float = 120.0,
    model_kind: str = "language",
) -> Iterator[str]:
    """Client: call a peer's governed inference stream and yield tokens.

    Signs the request with this agent's identity key (#1) and consumes the SSE
    response. ``session`` defaults to ``requests`` (inject a stub in tests).
    """
    from urllib.parse import urlsplit

    body: Dict[str, Any] = {"input": prompt, "model_kind": model_kind}
    path = urlsplit(url).path or "/"
    headers = peerauth.sign_peer_request(
        agent_id=agent_id,
        method="POST",
        path=path,
        body=body,
        storage_dir=storage_dir,
        private_key=private_key,
    )
    if session is None:
        import requests
        session = requests
    resp = session.post(url, json=body, headers=headers, stream=True, timeout=timeout)
    resp.raise_for_status()
    yield from parse_sse_tokens(resp.iter_lines(decode_unicode=True))
