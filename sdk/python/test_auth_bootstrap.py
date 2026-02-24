import importlib.util
from pathlib import Path


_AUTH_PATH = Path(__file__).parent / "ephapsys" / "auth.py"
_SPEC = importlib.util.spec_from_file_location("ephapsys_auth_module", _AUTH_PATH)
assert _SPEC and _SPEC.loader
auth = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(auth)


class _Resp:
    def __init__(self, status_code=200, payload=None):
        self.status_code = status_code
        self._payload = payload or {}

    def json(self):
        return self._payload


def test_get_api_key_prefers_explicit(monkeypatch):
    monkeypatch.setenv("AOC_PROVISIONING_TOKEN", "boot-token")
    assert auth.get_api_key("explicit-token") == "explicit-token"


def test_get_api_key_exchanges_provisioning_token(monkeypatch):
    monkeypatch.setenv("AOC_ORG_ID", "org_demo")
    monkeypatch.setenv("AOC_PROVISIONING_TOKEN", "boot-token")
    monkeypatch.setenv("AOC_BASE_URL", "http://localhost:7001")

    def _fake_post(url, json, timeout, verify):
        assert url == "http://localhost:7001/auth/device/token"
        assert json["org_id"] == "org_demo"
        assert json["provisioning_token"] == "boot-token"
        return _Resp(200, {"access_token": "jwt-token", "expires_in": 600})

    monkeypatch.setattr(auth.requests, "post", _fake_post)
    token = auth.get_api_key(None, base_url="http://localhost:7001", agent_instance_id="did:inst")
    assert token == "jwt-token"


def test_get_api_key_provisioning_token_requires_org(monkeypatch):
    monkeypatch.delenv("AOC_ORG_ID", raising=False)
    monkeypatch.setenv("AOC_PROVISIONING_TOKEN", "boot-token")
    try:
        auth.get_api_key(None, base_url="http://localhost:7001")
        raise AssertionError("expected RuntimeError")
    except RuntimeError as exc:
        assert "AOC_ORG_ID" in str(exc)
