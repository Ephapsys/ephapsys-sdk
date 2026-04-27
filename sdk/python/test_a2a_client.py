import hashlib
import hmac

from conftest import a2a


class _Resp:
    def __init__(self, *, ok=True, status_code=200, payload=None, text=""):
        self.ok = ok
        self.status_code = status_code
        self._payload = payload or {}
        self.text = text

    def json(self):
        return self._payload


def test_a2a_client_end_to_end_send_inbox_ack(monkeypatch):
    store = []

    def _fake_post(url, headers=None, json=None, timeout=None):
        if url.endswith("/a2a/messages"):
            msg_id = f"m{len(store) + 1}"
            msg = {
                "id": msg_id,
                "from_agent_id": json["from_agent_id"],
                "to_agent_id": json["to_agent_id"],
                "payload": json.get("payload", {}),
                "message_type": json.get("message_type", "event"),
                "status": "sent",
            }
            store.append(msg)
            return _Resp(payload={"ok": True, "message": msg})

        if "/a2a/messages/" in url and url.endswith("/ack"):
            parts = url.rstrip("/").split("/")
            msg_id = parts[-2]
            target = next((m for m in store if m["id"] == msg_id), None)
            if not target:
                return _Resp(ok=False, status_code=404, text="not found")
            if target["to_agent_id"] != json.get("agent_id"):
                return _Resp(ok=False, status_code=403, text="forbidden")
            target["status"] = "acked"
            target["acked_by"] = json.get("agent_id")
            return _Resp(payload={"ok": True, "message": target})

        return _Resp(ok=False, status_code=500, text="unexpected url")

    def _fake_get(url, headers=None, params=None, timeout=None):
        if url.endswith("/a2a/messages/inbox"):
            agent_id = params.get("agent_id")
            include_acked = bool(params.get("include_acked"))
            items = [m for m in store if m["to_agent_id"] == agent_id]
            if not include_acked:
                items = [m for m in items if m.get("status") != "acked"]
            return _Resp(payload={"items": items})
        return _Resp(ok=False, status_code=500, text="unexpected url")

    monkeypatch.setattr(a2a.requests, "post", _fake_post)
    monkeypatch.setattr(a2a.requests, "get", _fake_get)

    client = a2a.A2AClient(base_url="http://localhost:7001", token="tok")
    sent = client.send_message(
        from_agent_id="agent_sender",
        to_agent_id="agent_receiver",
        payload={"op": "ping"},
        message_type="event",
    )
    msg_id = sent["message"]["id"]
    assert sent["ok"] is True

    inbox = client.inbox(agent_id="agent_receiver")
    assert len(inbox["items"]) == 1
    assert inbox["items"][0]["id"] == msg_id
    assert inbox["items"][0]["status"] == "sent"

    acked = client.ack_message(message_id=msg_id, agent_id="agent_receiver")
    assert acked["ok"] is True
    assert acked["message"]["status"] == "acked"

    inbox_after = client.inbox(agent_id="agent_receiver")
    assert inbox_after["items"] == []


def test_a2a_client_error_surface(monkeypatch):
    def _fake_post(*args, **kwargs):
        return _Resp(ok=False, status_code=403, text="forbidden")

    monkeypatch.setattr(a2a.requests, "post", _fake_post)
    client = a2a.A2AClient(base_url="http://localhost:7001", token="tok")
    try:
        client.send_message(from_agent_id="a", to_agent_id="b", payload={})
        raise AssertionError("expected RuntimeError")
    except RuntimeError as exc:
        assert "A2A send failed" in str(exc)
        assert "403" in str(exc)


def test_a2a_client_from_env_prefers_a2a_token(monkeypatch):
    monkeypatch.setenv("AOC_BASE_URL", "http://localhost:7001")
    monkeypatch.setenv("AOC_A2A_TOKEN", "a2a_tok")
    monkeypatch.setenv("AOC_MODULATION_TOKEN", "mod_tok")
    c = a2a.A2AClient.from_env()
    assert c.token == "a2a_tok"


def test_a2a_client_signed_send_headers(monkeypatch):
    captured = {}

    def _fake_post(url, headers=None, json=None, timeout=None):
        captured["url"] = url
        captured["headers"] = headers or {}
        captured["json"] = json or {}
        return _Resp(payload={"ok": True, "message": {"id": "m1"}})

    monkeypatch.setattr(a2a.requests, "post", _fake_post)
    monkeypatch.setattr(a2a.time, "time", lambda: 1700000000)
    monkeypatch.setattr(a2a.secrets, "token_hex", lambda _n: "deadbeefdeadbeefdeadbeefdeadbeef")

    client = a2a.A2AClient(
        base_url="http://localhost:7001",
        token="tok",
        org_id="org_demo",
        sign_requests=True,
        hmac_secret="secret123",
    )
    client.send_message(
        from_agent_id="agent_sender",
        to_agent_id="agent_receiver",
        payload={"op": "ping"},
        message_type="event",
    )

    assert captured["url"].endswith("/a2a/messages")
    headers = captured["headers"]
    assert headers["x-a2a-ts"] == "1700000000"
    assert headers["x-a2a-nonce"] == "deadbeefdeadbeefdeadbeefdeadbeef"
    canonical_payload = (
        '{"correlation_id":"","from_agent_id":"agent_sender","message_type":"event",'
        '"payload":{"op":"ping"},"to_agent_id":"agent_receiver","ttl_seconds":0}'
    )
    canonical = "\n".join(
        [
            "1700000000",
            "deadbeefdeadbeefdeadbeefdeadbeef",
            "POST",
            "/a2a/messages",
            "org_demo",
            canonical_payload,
        ]
    )
    expected_sig = hmac.new(b"secret123", canonical.encode("utf-8"), hashlib.sha256).hexdigest()
    assert headers["x-a2a-sig"] == expected_sig
