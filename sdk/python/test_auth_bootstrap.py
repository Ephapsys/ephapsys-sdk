import importlib.util
from pathlib import Path
import base64

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec


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


def test_get_api_key_prefers_identity_challenge_when_key_present(monkeypatch, tmp_path):
    monkeypatch.delenv("AOC_PROVISIONING_TOKEN", raising=False)
    monkeypatch.setenv("AOC_ORG_ID", "org_demo")
    monkeypatch.setenv("AOC_BASE_URL", "http://localhost:7001")
    auth._TOKEN_CACHE.clear()

    storage = tmp_path / ".ephapsys_state"
    key_dir = storage / "kem"
    key_dir.mkdir(parents=True)
    priv = ec.generate_private_key(ec.SECP256R1())
    pub = priv.public_key()
    (key_dir / "kem_priv.pem").write_bytes(
        priv.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )
    )
    (key_dir / "kem_pub.pem").write_bytes(
        pub.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )
    )

    calls = {"challenge": 0, "token": 0}
    nonce_b64 = base64.b64encode(b"n" * 32).decode()

    def _fake_post(url, json, timeout, verify):
        if url == "http://localhost:7001/auth/device/challenge":
            calls["challenge"] += 1
            return _Resp(200, {"ok": True, "org_id": "org_demo", "nonce_b64": nonce_b64})
        if url == "http://localhost:7001/auth/device/token":
            calls["token"] += 1
            assert json["agent_instance_id"] == "did:ephapsys:inst"
            assert json["challenge_nonce_b64"] == nonce_b64
            message = f"ephapsys-device-auth-v1|org_demo|unknown-device|did:ephapsys:inst|{nonce_b64}".encode()
            signature = base64.b64decode(json["challenge_signature_b64"])
            pub.verify(signature, message, ec.ECDSA(hashes.SHA256()))
            return _Resp(200, {"access_token": "identity-jwt", "expires_in": 600})
        raise AssertionError(url)

    monkeypatch.setattr(auth.requests, "post", _fake_post)
    token = auth.get_api_key(
        None,
        base_url="http://localhost:7001",
        agent_instance_id="did:ephapsys:inst",
        storage_dir=str(storage),
    )
    assert token == "identity-jwt"
    assert calls == {"challenge": 1, "token": 1}


def test_get_api_key_provisioning_token_requires_org(monkeypatch):
    monkeypatch.delenv("AOC_ORG_ID", raising=False)
    monkeypatch.setenv("AOC_PROVISIONING_TOKEN", "boot-token")
    try:
        auth.get_api_key(None, base_url="http://localhost:7001")
        raise AssertionError("expected RuntimeError")
    except RuntimeError as exc:
        assert "AOC_ORG_ID" in str(exc)


def test_get_api_key_cache_not_split_by_agent_instance(monkeypatch):
    monkeypatch.setenv("AOC_ORG_ID", "org_demo")
    monkeypatch.setenv("AOC_PROVISIONING_TOKEN", "boot-token")
    monkeypatch.setenv("AOC_BASE_URL", "http://localhost:7001")
    auth._TOKEN_CACHE.clear()
    calls = {"count": 0}

    def _fake_post(url, json, timeout, verify):
        calls["count"] += 1
        return _Resp(200, {"access_token": "jwt-token", "expires_in": 600})

    monkeypatch.setattr(auth.requests, "post", _fake_post)
    t1 = auth.get_api_key(None, base_url="http://localhost:7001", agent_instance_id="agent_temp_1")
    t2 = auth.get_api_key(None, base_url="http://localhost:7001", agent_instance_id="did:ephapsys:abc")
    assert t1 == "jwt-token"
    assert t2 == "jwt-token"
    assert calls["count"] == 1


def test_check_helloworld_bootstrap(monkeypatch):
    def _fake_post(url, json, timeout, verify):
        assert url == "http://localhost:7001/sdk/onboarding/helloworld/check"
        assert json == {
            "org_id": "org_demo",
            "provisioning_token": "boot-token",
            "agent_template_id": "agent_temp_demo",
        }
        return _Resp(
            200,
            {
                "ok": True,
                "ready": True,
                "checks": [
                    {"code": "provisioning_token_valid", "ok": True},
                    {"code": "organization_active", "ok": True},
                ],
            },
        )

    monkeypatch.setattr(auth.requests, "post", _fake_post)
    out = auth.check_helloworld_bootstrap(
        base_url="http://localhost:7001",
        org_id="org_demo",
        provisioning_token="boot-token",
        agent_template_id="agent_temp_demo",
    )
    assert out["ready"] is True


def _identity_fails(monkeypatch, message="Device auth challenge failed (404): agent instance not found"):
    monkeypatch.setenv("AOC_ORG_ID", "org_demo")
    monkeypatch.setenv("AOC_BASE_URL", "http://localhost:7001")
    monkeypatch.setattr(auth, "_pkcs11_identity_enabled", lambda: True)
    auth._TOKEN_CACHE.clear()

    def _fail(**kw):
        raise RuntimeError(message)
    monkeypatch.setattr(auth, "_exchange_identity_token", _fail)


def test_both_exchanges_failing_reports_both(monkeypatch, caplog):
    """#190: the identity error used to be dropped when the provisioning fallback also failed."""
    _identity_fails(monkeypatch)
    monkeypatch.setenv("AOC_PROVISIONING_TOKEN", "boot-secret-value")
    monkeypatch.setattr(auth.requests, "post", lambda url, json, timeout, verify: _Resp(
        403, {"detail": "enrolled HSM instances require device-key proof"}))
    with caplog.at_level("WARNING", logger="ephapsys.auth"):
        try:
            auth.get_api_key(None, base_url="http://localhost:7001", agent_instance_id="did:inst")
        except RuntimeError as exc:
            msg = str(exc)
            assert msg.startswith("Device identity exchange failed: Device auth challenge failed (404)")
            assert "require device-key proof" in msg
            assert isinstance(exc.__cause__, RuntimeError) and "404" in str(exc.__cause__)
        else:
            raise AssertionError("expected RuntimeError")
    assert "agent instance not found" in caplog.text
    assert "boot-secret-value" not in msg and "boot-secret-value" not in caplog.text


def test_identity_fails_but_fallback_succeeds(monkeypatch, caplog):
    _identity_fails(monkeypatch)
    monkeypatch.setenv("AOC_PROVISIONING_TOKEN", "boot-token")
    monkeypatch.setattr(auth.requests, "post", lambda url, json, timeout, verify: _Resp(
        200, {"access_token": "jwt-fallback", "expires_in": 600}))
    with caplog.at_level("WARNING", logger="ephapsys.auth"):
        assert auth.get_api_key(None, base_url="http://localhost:7001", agent_instance_id="did:inst2") == "jwt-fallback"
    assert "Device identity exchange failed" in caplog.text


def test_identity_only_failure_is_raised_unchanged(monkeypatch):
    _identity_fails(monkeypatch, "Device auth challenge failed (409): agent instance not personalized")
    monkeypatch.delenv("AOC_PROVISIONING_TOKEN", raising=False)
    try:
        auth.get_api_key(None, base_url="http://localhost:7001", agent_instance_id="did:inst")
    except RuntimeError as exc:
        assert str(exc) == "Device auth challenge failed (409): agent instance not personalized"
    else:
        raise AssertionError("expected RuntimeError")


def test_provisioning_only_failure_is_raised_unchanged(monkeypatch):
    monkeypatch.setenv("AOC_ORG_ID", "org_demo")
    monkeypatch.setenv("AOC_PROVISIONING_TOKEN", "boot-token")
    monkeypatch.setattr(auth, "_pkcs11_identity_enabled", lambda: False)
    monkeypatch.setattr(auth, "_gcp_kms_identity_enabled", lambda: False)
    auth._TOKEN_CACHE.clear()
    monkeypatch.setattr(auth.requests, "post", lambda url, json, timeout, verify: _Resp(401, {"detail": "token expired"}))
    try:
        auth.get_api_key(None, base_url="http://localhost:7001", agent_instance_id="did:inst3",
                         storage_dir="/nonexistent-ephapsys-state")
    except RuntimeError as exc:
        assert "token expired" in str(exc) and "identity" not in str(exc)
    else:
        raise AssertionError("expected RuntimeError")


def test_identity_failure_with_fallback_missing_org(monkeypatch):
    _identity_fails(monkeypatch)
    monkeypatch.delenv("AOC_ORG_ID")
    monkeypatch.setenv("AOC_PROVISIONING_TOKEN", "boot-token")
    try:
        auth.get_api_key(None, base_url="http://localhost:7001", agent_instance_id="did:inst")
    except RuntimeError as exc:
        assert "Device identity exchange failed" in str(exc) and "Missing AOC_ORG_ID" in str(exc)
    else:
        raise AssertionError("expected RuntimeError")
