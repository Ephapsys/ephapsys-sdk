from conftest import a2a, journal_mod


class _Resp:
    def __init__(self, *, ok=True, status_code=200, payload=None, text=""):
        self.ok = ok
        self.status_code = status_code
        self._payload = payload or {}
        self.text = text

    def json(self):
        return self._payload


def _build_inbox_responder(*, inbox_items, sender_statuses):
    """Return (fake_get, fake_post, ack_log).

    sender_statuses maps sender_id -> {"status": ..., "state": {...}} (or None to 503).
    """
    ack_log = []

    def _fake_get(url, headers=None, params=None, timeout=None):
        if url.endswith("/a2a/messages/inbox"):
            return _Resp(payload={"items": list(inbox_items)})
        if "/agents/" in url and url.endswith("/status"):
            agent_id = url.split("/agents/")[1].split("/")[0]
            status = sender_statuses.get(agent_id)
            if status is None:
                return _Resp(ok=False, status_code=503, text="status unavailable")
            return _Resp(payload=status)
        return _Resp(ok=False, status_code=500, text=f"unexpected GET {url}")

    def _fake_post(url, headers=None, json=None, timeout=None):
        if url.startswith("http://localhost:7001/a2a/messages/") and url.endswith("/ack"):
            msg_id = url.rstrip("/").split("/")[-2]
            ack_log.append(msg_id)
            return _Resp(payload={"ok": True})
        return _Resp(ok=False, status_code=500, text=f"unexpected POST {url}")

    return _fake_get, _fake_post, ack_log


def _msg(*, msg_id, sender, message_type="event", payload=None):
    return {
        "id": msg_id,
        "from_agent_id": sender,
        "to_agent_id": "me",
        "message_type": message_type,
        "payload": payload or {"op": "ping"},
        "status": "sent",
    }


def test_process_inbox_dispatches_each_kind_and_journals(monkeypatch, tmp_path):
    inbox_items = [
        _msg(msg_id="m1", sender="alice"),                                            # verified
        _msg(msg_id="m2", sender="bob"),                                              # rejected (revoked)
        _msg(msg_id="m3", sender="carol", payload={"text": "ignore previous!"}),     # guardrail blocked
        _msg(
            msg_id="m4",
            sender="__system__",
            message_type="system.message_quarantine",
            payload={"revoked_agent_id": "bob", "quarantined_message_ids": ["x", "y"]},
        ),
        _msg(
            msg_id="m5",
            sender="__system__",
            message_type="system.status_change",
            payload={"agent_id": "bob", "old_status": "ENABLED", "new_status": "REVOKED"},
        ),
    ]
    sender_statuses = {
        "alice": {"status": "ENABLED", "state": {}},
        "bob": {"status": "REVOKED", "state": {"revoked": True}},
        "carol": {"status": "ENABLED", "state": {}},
    }
    fake_get, fake_post, ack_log = _build_inbox_responder(
        inbox_items=inbox_items, sender_statuses=sender_statuses
    )
    monkeypatch.setattr(a2a.requests, "get", fake_get)
    monkeypatch.setattr(a2a.requests, "post", fake_post)
    monkeypatch.setattr(
        a2a, "_payload_injection_hits",
        lambda payload: (
            [{"pattern": "ignore previous", "snippet": "ignore previous!"}]
            if isinstance(payload, dict) and "ignore previous" in str(payload)
            else []
        ),
    )

    client = a2a.A2AClient(base_url="http://localhost:7001", token="tok")
    journal = journal_mod.MessageJournal(path=str(tmp_path / "j.jsonl"))

    verified, rejected, quarantine, status = [], [], [], []
    summary = client.process_inbox(
        agent_id="me",
        on_verified=verified.append,
        on_rejected=rejected.append,
        on_quarantine_alert=quarantine.append,
        on_status_change=status.append,
        journal=journal,
    )

    assert summary == {
        "processed": 5,
        "verified": 1,
        "rejected": 1,
        "guardrail_blocked": 1,
        "quarantine_alerts": 1,
        "status_events": 1,
    }
    assert [m["id"] for m in verified] == ["m1"]
    assert [v.message["id"] for v in rejected] == ["m2", "m3"]
    assert [m["id"] for m in quarantine] == ["m4"]
    assert [m["id"] for m in status] == ["m5"]

    # Acks: verified + both system messages by default; rejected NOT acked.
    assert sorted(ack_log) == ["m1", "m4", "m5"]

    journal_entries = list(journal.read())
    decisions_by_id = {e["message_id"]: e["decision"] for e in journal_entries}
    assert decisions_by_id == {
        "m1": "verified",
        "m2": "rejected",
        "m3": "guardrail_blocked",
        "m4": "quarantine_alert",
        "m5": "system_event",
    }
    bob_entry = next(e for e in journal_entries if e["message_id"] == "m2")
    assert bob_entry["reason"] == "sender_revoked"
    carol_entry = next(e for e in journal_entries if e["message_id"] == "m3")
    assert carol_entry["reason"] == "guardrail_blocked"
    assert carol_entry["extra"]["guardrail_hits"]


def test_process_inbox_empty_inbox_is_noop(monkeypatch, tmp_path):
    fake_get, fake_post, ack_log = _build_inbox_responder(
        inbox_items=[], sender_statuses={}
    )
    monkeypatch.setattr(a2a.requests, "get", fake_get)
    monkeypatch.setattr(a2a.requests, "post", fake_post)

    client = a2a.A2AClient(base_url="http://localhost:7001", token="tok")
    summary = client.process_inbox(agent_id="me")
    assert summary["processed"] == 0
    assert ack_log == []


def test_process_inbox_can_ack_rejected(monkeypatch, tmp_path):
    inbox_items = [_msg(msg_id="m1", sender="bob")]
    sender_statuses = {"bob": {"status": "REVOKED", "state": {"revoked": True}}}
    fake_get, fake_post, ack_log = _build_inbox_responder(
        inbox_items=inbox_items, sender_statuses=sender_statuses
    )
    monkeypatch.setattr(a2a.requests, "get", fake_get)
    monkeypatch.setattr(a2a.requests, "post", fake_post)

    client = a2a.A2AClient(base_url="http://localhost:7001", token="tok")
    summary = client.process_inbox(agent_id="me", ack_rejected=True)
    assert summary["rejected"] == 1
    assert ack_log == ["m1"]


def test_process_inbox_continues_when_callback_raises(monkeypatch, tmp_path):
    inbox_items = [
        _msg(msg_id="m1", sender="alice"),
        _msg(msg_id="m2", sender="alice"),
    ]
    sender_statuses = {"alice": {"status": "ENABLED", "state": {}}}
    fake_get, fake_post, ack_log = _build_inbox_responder(
        inbox_items=inbox_items, sender_statuses=sender_statuses
    )
    monkeypatch.setattr(a2a.requests, "get", fake_get)
    monkeypatch.setattr(a2a.requests, "post", fake_post)
    monkeypatch.setattr(a2a, "_payload_injection_hits", lambda p: [])

    client = a2a.A2AClient(base_url="http://localhost:7001", token="tok")

    def _boom(msg):
        raise RuntimeError("downstream handler exploded")

    summary = client.process_inbox(agent_id="me", on_verified=_boom)
    # Both messages still counted as verified despite callback failure.
    assert summary["verified"] == 2
    assert sorted(ack_log) == ["m1", "m2"]


def test_process_inbox_does_not_ack_when_disabled(monkeypatch, tmp_path):
    inbox_items = [_msg(msg_id="m1", sender="alice")]
    sender_statuses = {"alice": {"status": "ENABLED", "state": {}}}
    fake_get, fake_post, ack_log = _build_inbox_responder(
        inbox_items=inbox_items, sender_statuses=sender_statuses
    )
    monkeypatch.setattr(a2a.requests, "get", fake_get)
    monkeypatch.setattr(a2a.requests, "post", fake_post)
    monkeypatch.setattr(a2a, "_payload_injection_hits", lambda p: [])

    client = a2a.A2AClient(base_url="http://localhost:7001", token="tok")
    summary = client.process_inbox(agent_id="me", ack_after_process=False)
    assert summary["verified"] == 1
    assert ack_log == []
