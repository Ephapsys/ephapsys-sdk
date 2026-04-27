from conftest import a2a


class _Resp:
    def __init__(self, *, ok=True, status_code=200, payload=None, text=""):
        self.ok = ok
        self.status_code = status_code
        self._payload = payload or {}
        self.text = text

    def json(self):
        return self._payload


def _build_responder(*, cluster_members, send_should_fail_for=None):
    """Return (fake_get, fake_post, send_calls)."""
    failing_recipients = set(send_should_fail_for or [])
    send_calls = []
    msg_counter = {"n": 0}

    def _fake_get(url, headers=None, params=None, timeout=None):
        if "/clusters/" in url and not url.endswith("/members"):
            return _Resp(payload={"cluster": {"agent_refs": list(cluster_members)}})
        return _Resp(ok=False, status_code=500, text=f"unexpected GET {url}")

    def _fake_post(url, headers=None, json=None, timeout=None):
        if url.endswith("/a2a/messages"):
            recipient = json["to_agent_id"]
            send_calls.append(json)
            if recipient in failing_recipients:
                return _Resp(ok=False, status_code=403, text=f"blocked: {recipient}")
            msg_counter["n"] += 1
            return _Resp(payload={
                "ok": True,
                "message": {"id": f"m{msg_counter['n']}", "to_agent_id": recipient},
            })
        return _Resp(ok=False, status_code=500, text=f"unexpected POST {url}")

    return _fake_get, _fake_post, send_calls


def test_broadcast_skips_self_and_returns_per_recipient_results(monkeypatch):
    fake_get, fake_post, send_calls = _build_responder(
        cluster_members=["alice", "bob", "carol"]
    )
    monkeypatch.setattr(a2a.requests, "get", fake_get)
    monkeypatch.setattr(a2a.requests, "post", fake_post)

    client = a2a.A2AClient(base_url="http://localhost:7001", token="tok")
    summary = client.broadcast(
        cluster_id="c1",
        from_agent_id="alice",
        payload={"op": "ping"},
        message_type="event",
    )

    assert summary["cluster_id"] == "c1"
    assert summary["sent"] == 2
    assert summary["failed"] == 0
    recipients = sorted(r["agent_id"] for r in summary["results"])
    assert recipients == ["bob", "carol"]
    assert all(r["ok"] for r in summary["results"])
    assert all(r["message_id"] for r in summary["results"])

    assert sorted(c["to_agent_id"] for c in send_calls) == ["bob", "carol"]
    for call in send_calls:
        assert call["from_agent_id"] == "alice"
        assert call["payload"] == {"op": "ping"}
        assert call["message_type"] == "event"


def test_broadcast_continues_on_partial_failure(monkeypatch):
    fake_get, fake_post, _ = _build_responder(
        cluster_members=["alice", "bob", "carol", "dave"],
        send_should_fail_for={"carol"},
    )
    monkeypatch.setattr(a2a.requests, "get", fake_get)
    monkeypatch.setattr(a2a.requests, "post", fake_post)

    client = a2a.A2AClient(base_url="http://localhost:7001", token="tok")
    summary = client.broadcast(
        cluster_id="c1",
        from_agent_id="alice",
        payload={"op": "ping"},
    )

    assert summary["sent"] == 2
    assert summary["failed"] == 1
    by_id = {r["agent_id"]: r for r in summary["results"]}
    assert by_id["bob"]["ok"] is True
    assert by_id["carol"]["ok"] is False
    assert "blocked" in by_id["carol"]["error"]
    assert by_id["dave"]["ok"] is True


def test_broadcast_skip_self_false_includes_sender(monkeypatch):
    fake_get, fake_post, send_calls = _build_responder(
        cluster_members=["alice", "bob"]
    )
    monkeypatch.setattr(a2a.requests, "get", fake_get)
    monkeypatch.setattr(a2a.requests, "post", fake_post)

    client = a2a.A2AClient(base_url="http://localhost:7001", token="tok")
    summary = client.broadcast(
        cluster_id="c1",
        from_agent_id="alice",
        payload={"op": "echo"},
        skip_self=False,
    )
    assert summary["sent"] == 2
    assert sorted(c["to_agent_id"] for c in send_calls) == ["alice", "bob"]


def test_broadcast_empty_cluster_is_noop(monkeypatch):
    fake_get, fake_post, send_calls = _build_responder(cluster_members=[])
    monkeypatch.setattr(a2a.requests, "get", fake_get)
    monkeypatch.setattr(a2a.requests, "post", fake_post)

    client = a2a.A2AClient(base_url="http://localhost:7001", token="tok")
    summary = client.broadcast(
        cluster_id="c1", from_agent_id="alice", payload={"x": 1}
    )
    assert summary == {"cluster_id": "c1", "sent": 0, "failed": 0, "results": []}
    assert send_calls == []
