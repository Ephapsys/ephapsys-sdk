from conftest import a2a


class _Resp:
    def __init__(self, *, ok=True, status_code=200, payload=None, text=""):
        self.ok = ok
        self.status_code = status_code
        self._payload = payload or {}
        self.text = text

    def json(self):
        return self._payload


def _make_client(monkeypatch, *, status_payload=None, status_code=200, status_ok=True):
    def _fake_get(url, headers=None, params=None, timeout=None):
        if "/agents/" in url and url.endswith("/status"):
            return _Resp(ok=status_ok, status_code=status_code, payload=status_payload or {})
        return _Resp(ok=False, status_code=500, text="unexpected url")

    monkeypatch.setattr(a2a.requests, "get", _fake_get)
    return a2a.A2AClient(base_url="http://localhost:7001", token="tok")


def _msg(*, sender="agent_x", message_type="event", payload=None, msg_id="m1"):
    return {
        "id": msg_id,
        "from_agent_id": sender,
        "to_agent_id": "me",
        "message_type": message_type,
        "payload": payload or {"op": "ping"},
        "status": "sent",
    }


def test_verify_passes_for_enabled_sender(monkeypatch):
    client = _make_client(monkeypatch, status_payload={"status": "ENABLED", "state": {}})
    result = client.verify_message(_msg(sender="alice"))
    assert result.verified is True
    assert result.reason is None
    assert result.sender_status == "ENABLED"
    assert result.is_system is False
    assert result.guardrail_hits == []


def test_verify_rejects_revoked_sender(monkeypatch):
    client = _make_client(monkeypatch, status_payload={"status": "REVOKED", "state": {"revoked": True}})
    result = client.verify_message(_msg(sender="alice"))
    assert result.verified is False
    assert result.reason == "sender_revoked"
    assert result.sender_status == "REVOKED"


def test_verify_rejects_disabled_sender(monkeypatch):
    client = _make_client(monkeypatch, status_payload={"status": "DISABLED", "state": {"enabled": False}})
    result = client.verify_message(_msg(sender="alice"))
    assert result.verified is False
    assert result.reason == "sender_status_disabled"
    assert result.sender_status == "DISABLED"


def test_verify_fails_closed_on_lookup_error(monkeypatch):
    client = _make_client(monkeypatch, status_ok=False, status_code=503, status_payload={})
    result = client.verify_message(_msg(sender="alice"))
    assert result.verified is False
    assert result.reason == "sender_lookup_failed"


def test_verify_skips_sender_check_for_system_messages(monkeypatch):
    # No HTTP call should be needed; install a get that fails to prove it.
    def _no_call(*args, **kwargs):
        raise AssertionError("status lookup should not happen for system messages")
    monkeypatch.setattr(a2a.requests, "get", _no_call)
    client = a2a.A2AClient(base_url="http://localhost:7001", token="tok")

    result = client.verify_message(
        _msg(sender="__system__", message_type="system.message_quarantine",
             payload={"revoked_agent_id": "x", "quarantined_message_ids": ["m1"]})
    )
    assert result.verified is True
    assert result.is_system is True
    assert result.reason is None


def test_verify_blocks_payload_with_prompt_injection(monkeypatch):
    client = _make_client(monkeypatch, status_payload={"status": "ENABLED", "state": {}})

    fake_hits = [{"pattern": "ignore previous instructions", "snippet": "ignore previous instructions and..."}]
    monkeypatch.setattr(a2a, "_payload_injection_hits", lambda payload: fake_hits)

    result = client.verify_message(_msg(payload={"text": "anything"}))
    assert result.verified is False
    assert result.reason == "guardrail_blocked"
    assert result.guardrail_hits == fake_hits
    assert result.sender_status == "ENABLED"


def test_verify_skips_guardrail_scan_when_disabled(monkeypatch):
    client = _make_client(monkeypatch, status_payload={"status": "ENABLED", "state": {}})

    def _explode(payload):
        raise AssertionError("scan should not run when scan_guardrails=False")
    monkeypatch.setattr(a2a, "_payload_injection_hits", _explode)

    result = client.verify_message(_msg(), scan_guardrails=False)
    assert result.verified is True
    assert result.guardrail_hits == []


def test_verify_rejects_missing_sender(monkeypatch):
    def _no_call(*args, **kwargs):
        raise AssertionError("status lookup should not happen for missing sender")
    monkeypatch.setattr(a2a.requests, "get", _no_call)
    client = a2a.A2AClient(base_url="http://localhost:7001", token="tok")

    result = client.verify_message(_msg(sender=""))
    assert result.verified is False
    assert result.reason == "missing_sender"
