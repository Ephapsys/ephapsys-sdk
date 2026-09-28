"""PKCS#11 anchor tests against SoftHSM (CI backend only; SoftHSM is NOT hardware).

Skipped when python-pkcs11 or libsofthsm2 is unavailable. Set SOFTHSM2_LIB to override the module path.
"""
import base64
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import types

import pytest

pkcs11 = pytest.importorskip("pkcs11")

_CANDIDATES = [os.getenv("SOFTHSM2_LIB", ""), "/opt/homebrew/lib/softhsm/libsofthsm2.so",
               "/usr/local/lib/softhsm/libsofthsm2.so", "/usr/lib/softhsm/libsofthsm2.so",
               "/usr/lib/x86_64-linux-gnu/softhsm/libsofthsm2.so"]
SOFTHSM_LIB = next((p for p in _CANDIDATES if p and os.path.exists(p)), None)
if not SOFTHSM_LIB or not shutil.which("softhsm2-util"):
    pytest.skip("SoftHSM not installed", allow_module_level=True)

from cryptography.hazmat.primitives import hashes, serialization  # noqa: E402
from cryptography.hazmat.primitives.asymmetric import ec, padding  # noqa: E402
from pkcs11 import Attribute, KeyType, Mechanism  # noqa: E402
from pkcs11.util.ec import encode_named_curve_parameters  # noqa: E402

from ephapsys.crypto import pkcs11 as p11  # noqa: E402
from ephapsys.crypto import hpke  # noqa: E402
from ephapsys import storage  # noqa: E402

PIN = "1234"
TOKEN = "ephapsys-ci"


@pytest.fixture(scope="module")
def token_env(tmp_path_factory):
    root = tmp_path_factory.mktemp("softhsm")
    (root / "tokens").mkdir()
    conf = root / "softhsm2.conf"
    conf.write_text(f"directories.tokendir = {root / 'tokens'}\nobjectstore.backend = file\nlog.level = ERROR\n")
    os.environ["SOFTHSM2_CONF"] = str(conf)
    subprocess.run(["softhsm2-util", "--init-token", "--free", "--label", TOKEN, "--pin", PIN, "--so-pin", "5678"],
                   check=True, capture_output=True)
    lib = pkcs11.lib(SOFTHSM_LIB)
    tok = lib.get_token(token_label=TOKEN)
    with tok.open(user_pin=PIN, rw=True) as s:
        def ec_pair(curve, kid, label, priv):
            params = s.create_domain_parameters(KeyType.EC, {Attribute.EC_PARAMS: encode_named_curve_parameters(curve)}, local=True)
            tmpl = {Attribute.SENSITIVE: True, Attribute.EXTRACTABLE: False}
            tmpl.update(priv)
            return params.generate_keypair(store=True, id=kid, label=label, private_template=tmpl)
        SIGN_ONLY = {Attribute.SIGN: True, Attribute.DERIVE: False}
        KEM_ONLY = {Attribute.DERIVE: True, Attribute.SIGN: False}
        ec_pair("secp256r1", b"\x01", "sign", SIGN_ONLY)
        ec_pair("secp256r1", b"\x02", "kem", KEM_ONLY)
        s.generate_keypair(KeyType.RSA, 2048, store=True, id=b"\x03", label="rsa-sign",
                           private_template={Attribute.SENSITIVE: True, Attribute.EXTRACTABLE: False, Attribute.SIGN: True, Attribute.DERIVE: False})
        ec_pair("secp256r1", b"\x04", "kem-other", KEM_ONLY)
        ec_pair("secp256r1", b"\x05", "extractable", {**SIGN_ONLY, Attribute.SENSITIVE: False, Attribute.EXTRACTABLE: True})
        ec_pair("secp256r1", b"\x06", "dup", SIGN_ONLY)
        ec_pair("secp256r1", b"\x07", "dup", SIGN_ONLY)
        ec_pair("secp384r1", b"\x08", "kem-p384", KEM_ONLY)
        ec_pair("secp256r1", b"\x09", "no-sign-perm", {Attribute.SIGN: False, Attribute.DERIVE: False})
        ec_pair("secp256r1", b"\x0a", "kem-can-sign", {Attribute.DERIVE: True, Attribute.SIGN: True})
        ec_pair("secp256r1", b"\x0b", "sign-can-derive", {Attribute.SIGN: True, Attribute.DERIVE: True})
    return dict(PKCS11_MODULE=SOFTHSM_LIB, PKCS11_TOKEN_LABEL=TOKEN, PKCS11_PIN=PIN,
                PKCS11_SIGN_KEY_ID="01", PKCS11_KEM_KEY_ID="02")


def provider(env, **over):
    e = dict(env); e.update(over)
    e = {k: v for k, v in e.items() if v is not None}
    return p11.Pkcs11Provider(p11.Pkcs11Config.from_env(e))


def load_pub(pem):
    return serialization.load_pem_public_key(pem.encode())


# ---------------- evidence / binding ----------------
def test_ec_evidence_binding_verifies(token_env):
    prov = provider(token_env)
    nonce = os.urandom(32); nb64 = base64.b64encode(nonce).decode()
    ev = prov.personalization_evidence(nb64)
    assert ev["provider"] == "pkcs11" and ev["binding_version"] == 1 and ev["sig_alg"] == p11.SIG_ALG_EC
    msg = p11.BINDING_DOMAIN + nonce + hashlib.sha256(p11.spki_der(ev["kem_pub_pem"])).digest()
    assert msg == p11.binding_message(nonce, ev["kem_pub_pem"])
    load_pub(ev["pubkey_pem"]).verify(base64.b64decode(ev["sig_b64"]), msg, ec.ECDSA(hashes.SHA256()))
    assert ev["pubkey_spki_sha256"] == hashlib.sha256(p11.spki_der(ev["pubkey_pem"])).hexdigest()
    assert ev["token"]["label"] == TOKEN and ev["token"]["serial"]
    assert ev["sign_key_id_hex"] == "01" and ev["kem_key_id_hex"] == "02"
    assert ev["pubkey_pem"] != ev["kem_pub_pem"]                     # separate roles


def test_signature_does_not_verify_for_other_kem_or_nonce(token_env):
    prov = provider(token_env); nonce = os.urandom(16)
    ev = prov.personalization_evidence(base64.b64encode(nonce).decode())
    other_kem = provider(token_env, PKCS11_KEM_KEY_ID="04").public_key_pem("kem")
    pub = load_pub(ev["pubkey_pem"]); sig = base64.b64decode(ev["sig_b64"])
    from cryptography.exceptions import InvalidSignature
    with pytest.raises(InvalidSignature):
        pub.verify(sig, p11.binding_message(nonce, other_kem), ec.ECDSA(hashes.SHA256()))
    with pytest.raises(InvalidSignature):
        pub.verify(sig, p11.binding_message(os.urandom(16), ev["kem_pub_pem"]), ec.ECDSA(hashes.SHA256()))


def test_rsa_sign_key_evidence(token_env):
    prov = provider(token_env, PKCS11_SIGN_KEY_ID="03")
    nonce = os.urandom(32); ev = prov.personalization_evidence(base64.b64encode(nonce).decode())
    assert ev["sig_alg"] == p11.SIG_ALG_RSA
    load_pub(ev["pubkey_pem"]).verify(base64.b64decode(ev["sig_b64"]), p11.binding_message(nonce, ev["kem_pub_pem"]),
                                      padding.PKCS1v15(), hashes.SHA256())


def test_select_by_label(token_env):
    prov = provider(token_env, PKCS11_SIGN_KEY_ID=None, PKCS11_SIGN_KEY_LABEL="sign",
                    PKCS11_KEM_KEY_ID=None, PKCS11_KEM_KEY_LABEL="kem")
    assert prov.public_key_pem("sign") == provider(token_env).public_key_pem("sign")


# ---------------- ECDH / SIE ----------------
def test_ecdh_matches_software_and_sie_unwrap(token_env):
    prov = provider(token_env)
    kem_pub = load_pub(prov.public_key_pem("kem"))
    eph = ec.generate_private_key(ec.SECP256R1())
    eph_der = eph.public_key().public_bytes(serialization.Encoding.DER, serialization.PublicFormat.SubjectPublicKeyInfo)
    assert prov.ecdh(eph_der) == eph.exchange(ec.ECDH(), kem_pub)
    cek = os.urandom(32)
    wrapped = hpke.wrap(prov.public_key_pem("kem"), cek)
    assert hpke.unwrap(wrapped, ecdh_with_ephemeral=prov.ecdh) == cek


def test_ecdh_rejects_non_p256_peer(token_env):
    prov = provider(token_env)
    peer = ec.generate_private_key(ec.SECP384R1()).public_key().public_bytes(
        serialization.Encoding.DER, serialization.PublicFormat.SubjectPublicKeyInfo)
    with pytest.raises(p11.Pkcs11Error):
        prov.ecdh(peer)


# ---------------- fail-closed selection / policy ----------------
@pytest.mark.parametrize("over,needle", [
    (dict(PKCS11_SIGN_KEY_ID="7f"), "exactly one private sign key"),              # missing key
    (dict(PKCS11_SIGN_KEY_ID=None, PKCS11_SIGN_KEY_LABEL="dup"), "found 2"),         # multiple keys
    (dict(PKCS11_SIGN_KEY_ID="05"), "sensitive and non-extractable"),               # extractable key
    (dict(PKCS11_SIGN_KEY_ID="09"), "CKA_SIGN"),                                     # usage not permitted
])
def test_sign_key_rejections(token_env, over, needle):
    with pytest.raises(p11.Pkcs11Error, match=needle):
        provider(token_env, **over).sign(b"x")


def test_role_separation_enforced(token_env):
    with pytest.raises(p11.Pkcs11Error, match="must not have CKA_SIGN"):
        provider(token_env, PKCS11_KEM_KEY_ID="0a").public_key_pem("kem")
    with pytest.raises(p11.Pkcs11Error, match="must not have CKA_DERIVE"):
        provider(token_env, PKCS11_SIGN_KEY_ID="0b").sign(b"x")
    with pytest.raises(p11.Pkcs11Error, match="distinct"):
        provider(token_env, PKCS11_KEM_KEY_ID="01")          # identical selectors rejected at validation


def test_constructor_validates_config(token_env):
    bad = p11.Pkcs11Config(module=SOFTHSM_LIB, token_label=TOKEN, pin=PIN, sign_key_id=b"\x01")   # no kem selector
    with pytest.raises(p11.Pkcs11Error, match="kem key"):
        p11.Pkcs11Provider(bad)


def test_kem_must_be_p256(token_env):
    with pytest.raises(p11.Pkcs11Error, match="P-256"):
        provider(token_env, PKCS11_KEM_KEY_ID="08").public_key_pem("kem")


def test_ecdsa_raw_fallback_when_combined_mechanism_missing(token_env, monkeypatch):
    """Tokens such as SoftHSM 2.6 lack CKM_ECDSA_SHA256: sign SHA-256(msg) with raw CKM_ECDSA instead."""
    prov = provider(token_env)
    real = prov.mechanisms()
    monkeypatch.setattr(prov, "mechanisms", lambda: {m for m in real if m != Mechanism.ECDSA_SHA256})
    nonce = os.urandom(32)
    ev = prov.personalization_evidence(base64.b64encode(nonce).decode())
    assert ev["sig_alg"] == p11.SIG_ALG_EC
    load_pub(ev["pubkey_pem"]).verify(base64.b64decode(ev["sig_b64"]), p11.binding_message(nonce, ev["kem_pub_pem"]),
                                      ec.ECDSA(hashes.SHA256()))       # verifies exactly like ECDSA_SHA256 output


def test_missing_mechanism_fails_closed(token_env, monkeypatch):
    prov = provider(token_env)
    monkeypatch.setattr(prov, "mechanisms", lambda: {Mechanism.SHA256_RSA_PKCS})
    with pytest.raises(p11.Pkcs11Error, match="mechanism"):
        prov.sign(b"x")
    with pytest.raises(p11.Pkcs11Error, match="mechanism"):
        prov.ecdh(ec.generate_private_key(ec.SECP256R1()).public_key().public_bytes(
            serialization.Encoding.DER, serialization.PublicFormat.SubjectPublicKeyInfo))


@pytest.mark.parametrize("over,needle", [
    (dict(PKCS11_TOKEN_LABEL=None), "TOKEN_LABEL or PKCS11_TOKEN_SERIAL"),
    (dict(PKCS11_PIN=None), "PIN"),
    (dict(PKCS11_PIN_FILE="/x"), "only one"),
    (dict(PKCS11_SIGN_KEY_ID="zz"), "hex"),
    (dict(PKCS11_TOKEN_LABEL="nope"), "exactly one token"),
])
def test_config_rejections(token_env, over, needle):
    with pytest.raises(p11.Pkcs11Error, match=needle):
        provider(token_env, **over)


def test_wrong_pin_fails(token_env):
    with pytest.raises(Exception):
        provider(token_env, PKCS11_PIN="0000").sign(b"x")


def test_pin_file(token_env, tmp_path):
    f = tmp_path / "pin"; f.write_text(PIN + "\n")
    assert provider(token_env, PKCS11_PIN=None, PKCS11_PIN_FILE=str(f)).sign(b"x")[0] == p11.SIG_ALG_EC


# ---------------- at-rest cache (storage key provider) ----------------
@pytest.fixture
def pkcs11_storage(token_env, monkeypatch):
    for k, v in token_env.items(): monkeypatch.setenv(k, v)
    monkeypatch.delenv("EPHAPSYS_KEY_PROVIDER", raising=False)
    storage.set_pkcs11_provider(provider(token_env))
    yield
    storage.set_pkcs11_provider(None)


def test_cache_roundtrip_and_format(pkcs11_storage, tmp_path):
    assert storage.key_provider() == "pkcs11"
    d = str(tmp_path); data = os.urandom(1000)
    storage.write_encrypted(d, "ecm", data)
    assert storage.read_encrypted(d, "ecm") == data
    rec = json.load(open(os.path.join(d, "sealed_dek.pkcs11.json")))
    assert rec["provider"] == "pkcs11" and "wrapped_dek" in rec and not os.path.exists(os.path.join(d, "sealed_dek.b64"))
    meta = json.load(open(os.path.join(d, "cache", "ecm.meta.json")))
    assert meta["provider"] == "pkcs11" and meta["v"] == 2 and meta["name"] == "ecm"
    assert data not in open(os.path.join(d, "cache", "ecm.enc"), "rb").read()


def test_cache_tamper_detection(pkcs11_storage, tmp_path):
    from cryptography.exceptions import InvalidTag
    d = str(tmp_path); storage.write_encrypted(d, "ecm", b"secret-lambda")
    enc = os.path.join(d, "cache", "ecm.enc"); b = bytearray(open(enc, "rb").read()); b[0] ^= 1; open(enc, "wb").write(bytes(b))
    with pytest.raises(InvalidTag):
        storage.read_encrypted(d, "ecm")
    # ciphertext moved under another name: AAD binds the name
    storage.write_encrypted(d, "a", b"alpha")
    for suf in (".enc",):
        shutil.copy(os.path.join(d, "cache", "a" + suf), os.path.join(d, "cache", "b" + suf))
    meta = json.load(open(os.path.join(d, "cache", "a.meta.json"))); meta["name"] = "b"
    json.dump(meta, open(os.path.join(d, "cache", "b.meta.json"), "w"))
    with pytest.raises(InvalidTag):
        storage.read_encrypted(d, "b")
    meta["provider"] = "tpm"; json.dump(meta, open(os.path.join(d, "cache", "b.meta.json"), "w"))
    with pytest.raises(storage.KeyProviderError):
        storage.read_encrypted(d, "b")


def test_cache_refuses_other_kem_key(pkcs11_storage, token_env, tmp_path):
    d = str(tmp_path); storage.write_encrypted(d, "ecm", b"x")
    storage.set_pkcs11_provider(provider(token_env, PKCS11_KEM_KEY_ID="04"))
    with pytest.raises(storage.KeyProviderError, match="different token KEM key"):
        storage.read_encrypted(d, "ecm")


def test_cache_refuses_provider_switch(pkcs11_storage, tmp_path, monkeypatch):
    d = str(tmp_path); storage.write_encrypted(d, "ecm", b"x")
    monkeypatch.setenv("EPHAPSYS_KEY_PROVIDER", "tpm")
    with pytest.raises(storage.KeyProviderError, match="different key provider"):
        storage.read_encrypted(d, "ecm")
    monkeypatch.setenv("EPHAPSYS_KEY_PROVIDER", "pkcs11")
    open(os.path.join(d, "sealed_dek.b64"), "w").write("x")
    with pytest.raises(storage.KeyProviderError, match="different key provider"):
        storage.read_encrypted(d, "ecm")


def test_invalid_provider_name(monkeypatch):
    monkeypatch.setenv("EPHAPSYS_KEY_PROVIDER", "software")
    with pytest.raises(storage.KeyProviderError):
        storage.key_provider()


def test_tpm_path_unchanged_format(monkeypatch, tmp_path):
    """Default provider is still TPM with the original v1 format (insecure test seal)."""
    monkeypatch.delenv("PKCS11_MODULE", raising=False); monkeypatch.delenv("EPHAPSYS_KEY_PROVIDER", raising=False)
    monkeypatch.setattr(storage.tpm, "ALLOW_INSECURE", True)
    d = str(tmp_path); storage.write_encrypted(d, "ecm", b"v1")
    meta = json.load(open(os.path.join(d, "cache", "ecm.meta.json")))
    assert set(meta) == {"alg", "nonce_b64", "len"} and storage.read_encrypted(d, "ecm") == b"v1"


# ---------------- device auth (auth.py) ----------------
def test_device_auth_signs_with_sign_key_not_kem(token_env, monkeypatch):
    from ephapsys import auth
    for k, v in token_env.items(): monkeypatch.setenv(k, v)
    msg = b"ephapsys-device-auth-v1|org|dev|agent|nonce"
    sig = auth._sign_identity_message(msg, None)
    prov = provider(token_env)
    load_pub(prov.public_key_pem("sign")).verify(sig, msg, ec.ECDSA(hashes.SHA256()))
    from cryptography.exceptions import InvalidSignature
    with pytest.raises(InvalidSignature):
        load_pub(prov.public_key_pem("kem")).verify(sig, msg, ec.ECDSA(hashes.SHA256()))


def test_resolve_device_id(monkeypatch):
    from ephapsys import auth
    monkeypatch.delenv("EPHAPSYS_DEVICE_ID", raising=False); monkeypatch.delenv("HOSTNAME", raising=False)
    assert auth.resolve_device_id() == "unknown-device"
    with pytest.raises(RuntimeError):
        auth.resolve_device_id(strict=True)
    monkeypatch.setenv("EPHAPSYS_DEVICE_ID", "device-0001")
    assert auth.resolve_device_id(strict=True) == "device-0001"


# ---------------- agent hsm anchor path ----------------
def _stub_agent(agent_mod, prov):
    stub = types.SimpleNamespace()
    TA = agent_mod.TrustedAgent
    stub._pkcs11_enabled = types.MethodType(TA._pkcs11_enabled, stub)
    stub._pkcs11_provider = lambda: prov
    return stub


def test_agent_hsm_uses_native_pkcs11(token_env, monkeypatch):
    import ephapsys.agent as agent_mod
    for k, v in token_env.items(): monkeypatch.setenv(k, v)
    for k in ("HSM_HELPER", "HSM_KMS_KEY", "HSM_EVIDENCE_PATH"): monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("EPHAPSYS_DEVICE_ID", "device-0001")
    stub = _stub_agent(agent_mod, provider(token_env))
    nb64 = base64.b64encode(os.urandom(32)).decode()
    ev = agent_mod.TrustedAgent._collect_hsm_evidence(stub, nb64)
    assert ev["provider"] == "pkcs11" and ev["device_id"] == "device-0001" and ev["nonce_b64"] == nb64
    assert agent_mod.TrustedAgent._ensure_auth_pub_pem(stub) == ev["kem_pub_pem"]


@pytest.mark.parametrize("other", ["HSM_HELPER", "HSM_KMS_KEY", "HSM_EVIDENCE_PATH"])
def test_agent_hsm_rejects_ambiguous_provider(token_env, monkeypatch, other):
    import ephapsys.agent as agent_mod
    for k, v in token_env.items(): monkeypatch.setenv(k, v)
    monkeypatch.setenv("EPHAPSYS_DEVICE_ID", "device-0001"); monkeypatch.setenv(other, "x")
    stub = _stub_agent(agent_mod, provider(token_env))
    with pytest.raises(RuntimeError, match="Ambiguous HSM configuration"):
        agent_mod.TrustedAgent._collect_hsm_evidence(stub, base64.b64encode(b"n").decode())


def test_agent_hsm_requires_stable_device_id(token_env, monkeypatch):
    import ephapsys.agent as agent_mod
    for k, v in token_env.items(): monkeypatch.setenv(k, v)
    for k in ("HSM_HELPER", "HSM_KMS_KEY", "HSM_EVIDENCE_PATH", "EPHAPSYS_DEVICE_ID", "HOSTNAME"): monkeypatch.delenv(k, raising=False)
    stub = _stub_agent(agent_mod, provider(token_env))
    with pytest.raises(RuntimeError, match="stable device id"):
        agent_mod.TrustedAgent._collect_hsm_evidence(stub, base64.b64encode(b"n").decode())


# ---------------- CSR bound to the durable device key (no throwaway key) ----------------
def _csr(pem):
    from cryptography import x509
    return x509.load_pem_x509_csr(pem.encode())


def _same_key(pub_a, pub_b):
    f = lambda k: k.public_bytes(serialization.Encoding.DER, serialization.PublicFormat.SubjectPublicKeyInfo)
    return f(pub_a) == f(pub_b)


@pytest.mark.parametrize("sign_id", ["01", "03"])
def test_pkcs11_csr_signed_by_token_sign_key(token_env, sign_id):
    from cryptography.x509.oid import NameOID
    prov = provider(token_env, PKCS11_SIGN_KEY_ID=sign_id)
    csr = _csr(prov.build_csr("agent:abc"))
    assert csr.is_signature_valid
    assert _same_key(csr.public_key(), load_pub(prov.public_key_pem("sign")))
    assert not _same_key(csr.public_key(), load_pub(prov.public_key_pem("kem")))
    assert csr.subject.get_attributes_for_oid(NameOID.COMMON_NAME)[0].value == "agent:abc"


def _csr_stub(agent_mod, storage_dir, prov=None):
    import pathlib
    TA = agent_mod.TrustedAgent
    stub = types.SimpleNamespace(storage_dir=pathlib.Path(storage_dir))
    for name in ("_pkcs11_enabled", "_kem_key_paths", "_ensure_auth_pub_pem", "_load_kem_priv", "_generate_csr"):
        setattr(stub, name, types.MethodType(getattr(TA, name), stub))
    if prov is not None:
        stub._pkcs11_provider = lambda: prov
    return stub


def test_agent_csr_uses_token_when_pkcs11(token_env, monkeypatch, tmp_path):
    import ephapsys.agent as agent_mod
    for k, v in token_env.items(): monkeypatch.setenv(k, v)
    prov = provider(token_env)
    csr = _csr(agent_mod.TrustedAgent._generate_csr(_csr_stub(agent_mod, tmp_path, prov), "abc"))
    assert csr.is_signature_valid and _same_key(csr.public_key(), load_pub(prov.public_key_pem("sign")))
    assert not (tmp_path / "kem" / "kem_priv.pem").exists()            # no software key created


def test_agent_csr_uses_durable_key_without_pkcs11(monkeypatch, tmp_path):
    import ephapsys.agent as agent_mod
    monkeypatch.delenv("PKCS11_MODULE", raising=False)
    stub = _csr_stub(agent_mod, tmp_path)
    a = _csr(agent_mod.TrustedAgent._generate_csr(stub, "abc"))
    b = _csr(agent_mod.TrustedAgent._generate_csr(stub, "abc"))
    durable = load_pub((tmp_path / "kem" / "kem_pub.pem").read_text())
    assert a.is_signature_valid and _same_key(a.public_key(), durable) and _same_key(b.public_key(), durable)


# ---------------- strict SIE (native PKCS#11 path) ----------------
class _Resp:
    def __init__(self, code, content=b""):
        self.status_code, self.content = code, content


def _sie(agent_mod, state_dir, prov, base="https://aoc.example.com", strict=True):
    sie = object.__new__(agent_mod.SIEManager)          # bypass network credential resolution
    sie.strict, sie.base_url, sie.agent_id, sie.state_dir = strict, base, "inst-1", str(state_dir)
    sie.verify_ssl, sie.api_key = True, "device-token"
    sie.privkey_pem = sie.privkey_loader = None
    sie.tpm_ecdh = prov.ecdh if prov else None
    sie._status = lambda: {"ok": True, "enabled": True, "revoked": False}
    return sie


def _cipher_entry(prov, ecm=b"lambda-bytes"):
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    cek, nonce = os.urandom(32), os.urandom(12)
    ct = nonce + AESGCM(cek).encrypt(nonce, ecm, None)
    entry = {"id": "m1", "cipher_ecm_uri": "/agents/inst-1/sie/m1", "cipher_ecm_digest": "sha256:" + hashlib.sha256(ct).hexdigest(),
             "sie_wrapped_cek_b64": hpke.wrap(prov.public_key_pem("kem"), cek)}
    return entry, ct


def test_strict_sie_roundtrip_same_origin_auth(pkcs11_storage, token_env, tmp_path, monkeypatch):
    import ephapsys.agent as agent_mod, requests
    prov = provider(token_env); entry, ct = _cipher_entry(prov)
    calls = []
    def fake_get(url, headers=None, timeout=None, verify=None, allow_redirects=None):
        calls.append((url, dict(headers or {}), verify, allow_redirects)); return _Resp(200, ct)
    monkeypatch.setattr(requests, "get", fake_get)
    sie = _sie(agent_mod, tmp_path, prov)
    assert sie.ensure_ecm_cached_and_get_bytes(entry) == b"lambda-bytes"
    url, headers, verify, redirects = calls[0]
    assert url == "https://aoc.example.com/agents/inst-1/sie/m1" and headers["Authorization"] == "Bearer device-token"
    assert verify is True and redirects is False
    assert sie.ensure_ecm_cached_and_get_bytes(entry) == b"lambda-bytes" and len(calls) == 1      # served from the encrypted cache
    cached = [f for f in os.listdir(os.path.join(tmp_path, "cache")) if f.endswith(".enc")]
    assert cached and all(entry["cipher_ecm_digest"][7:23] in f for f in cached)                  # digest-bound cache name


def test_strict_sie_url_rules(token_env, tmp_path, monkeypatch):
    import ephapsys.agent as agent_mod, requests
    seen = []
    monkeypatch.setattr(requests, "get", lambda url, headers=None, **k: seen.append(dict(headers or {})) or _Resp(200, b"x"))
    sie = _sie(agent_mod, tmp_path, provider(token_env))
    sie._fetch_cipher("https://storage.example.net/obj?sig=1")                                   # foreign origin: no token
    assert "Authorization" not in seen[-1]
    for bad in ("http://aoc.example.com/x", "file:///etc/passwd", "ftp://aoc.example.com/x"):
        with pytest.raises(agent_mod.SecureInferenceError):
            sie._fetch_cipher(bad)
    monkeypatch.setattr(requests, "get", lambda *a, **k: _Resp(302))
    with pytest.raises(agent_mod.SecureInferenceError, match="302"):
        sie._fetch_cipher("/agents/inst-1/sie/m1")
    dev = _sie(agent_mod, tmp_path, provider(token_env), base="http://localhost:7001")            # dev AOC over http, same origin
    monkeypatch.setattr(requests, "get", lambda url, headers=None, **k: seen.append(dict(headers or {})) or _Resp(200, b"x"))
    dev._fetch_cipher("/agents/inst-1/sie/m1"); assert seen[-1]["Authorization"] == "Bearer device-token"


def test_strict_sie_requires_digest_ecdh_and_matching_ciphertext(pkcs11_storage, token_env, tmp_path, monkeypatch):
    import ephapsys.agent as agent_mod, requests
    prov = provider(token_env); entry, ct = _cipher_entry(prov)
    monkeypatch.setattr(requests, "get", lambda *a, **k: _Resp(200, ct[:-1] + bytes([ct[-1] ^ 1])))
    with pytest.raises(agent_mod.SecureInferenceError, match="digest mismatch"):
        _sie(agent_mod, tmp_path, prov).ensure_ecm_cached_and_get_bytes(entry)
    with pytest.raises(agent_mod.SecureInferenceError, match="cipher_ecm_digest is required"):
        _sie(agent_mod, tmp_path, prov).ensure_ecm_cached_and_get_bytes({**entry, "cipher_ecm_digest": ""})
    with pytest.raises(agent_mod.SecureInferenceError, match="token ECDH"):
        _sie(agent_mod, tmp_path, None).ensure_ecm_cached_and_get_bytes(entry)
    other = hpke.wrap(provider(token_env, PKCS11_KEM_KEY_ID="04").public_key_pem("kem"), os.urandom(32))
    monkeypatch.setattr(requests, "get", lambda *a, **k: _Resp(200, ct))
    with pytest.raises(agent_mod.SecureInferenceError, match="unwrap failed"):                   # CEK wrapped to another key
        _sie(agent_mod, tmp_path, prov).ensure_ecm_cached_and_get_bytes({**entry, "sie_wrapped_cek_b64": other})


@pytest.mark.parametrize("mutate,needle", [
    (lambda e: e.pop("cipher_ecm_uri"), "lacks SIE"),
    (lambda e: e.update(sie_wrapped_cek_b64=None), "lacks SIE"),
    (lambda e: e.update(cipher_ecm_digest=""), "cipher_ecm_digest"),
    (lambda e: e.update(ecm_uri="https://x/ecm.pt"), "plaintext ecm_uri"),
    (lambda e: e.update(artifact_urls={"ecm.pt": {"url": "https://x/ecm.pt"}}), "plaintext ECM artifacts"),
    (lambda e: e.update(artifact_urls={"weights": {"url": "https://x/files/lambda_ecm.pt?sig=1"}}), "plaintext ECM artifacts"),
])
def test_native_manifest_rejections(mutate, needle):
    import ephapsys.agent as agent_mod
    entry = {"cipher_ecm_uri": "/agents/i/sie/m", "sie_wrapped_cek_b64": "w", "cipher_ecm_digest": "sha256:ab",
             "artifact_urls": {"model.safetensors": {"url": "https://x/model.safetensors"}}}
    stub = types.SimpleNamespace(_is_ecm_artifact=agent_mod.TrustedAgent._is_ecm_artifact)
    agent_mod.TrustedAgent._check_native_sie_entry(stub, "m", entry)                              # valid entry passes
    mutate(entry)
    with pytest.raises(agent_mod.SecureInferenceError, match=needle):
        agent_mod.TrustedAgent._check_native_sie_entry(stub, "m", entry)


def test_ecm_apply_fails_closed_when_required():
    import ephapsys.agent as agent_mod
    stub = types.SimpleNamespace(_resolve_ecm_target=lambda model, t: None)
    runtime = {"kind": "language", "_ecm_required": True}
    with pytest.raises(agent_mod.SecureInferenceError, match="ECM target"):
        agent_mod.TrustedAgent._apply_ecm_if_available(stub, object(), runtime)
    agent_mod.TrustedAgent._apply_ecm_if_available(stub, object(), {"kind": "language"})          # legacy path: warn only
