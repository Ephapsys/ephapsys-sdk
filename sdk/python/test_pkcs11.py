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


# ---------------------------------------------------------------------------
# hsm provision (pkcs11_provision): each SoftHSM test initialises the free slot under its own label
# ---------------------------------------------------------------------------
from ephapsys.crypto import pkcs11_provision as prov_mod  # noqa: E402


def _pcfg(label, pin_file, sign="ephapsys-sign", kem="ephapsys-kem", **over):
    return p11.Pkcs11Config(module=SOFTHSM_LIB, token_label=label, pin_file=str(pin_file),
                            sign_key_label=sign, kem_key_label=kem, **over)


def _secret(path, text, mode=0o600):
    path.write_text(text)
    path.chmod(mode)
    return path


def _labels():
    return sorted((s.get_token().label or "").strip() for s in pkcs11.lib(SOFTHSM_LIB).get_slots(token_present=True))


def spki_of(prov, role):
    return p11.spki_sha256_hex(prov.public_key_pem(role))


def test_provision_fresh_then_idempotent(token_env, tmp_path):
    import grp
    import stat
    pin_file = tmp_path / "etc" / "pkcs11.pin"
    group = grp.getgrgid(os.getgid()).gr_name
    first = prov_mod.provision(_pcfg("prov-fresh", pin_file), pin_group=group)
    assert first["status"] == "provisioned"
    acts = first["actions"]
    assert acts[0] == "created PIN file" and acts[1].startswith("initialised token 'prov-fresh' in slot ")
    assert acts[2:] == ["set the user PIN", "discarded the SO PIN",
                        "created sign key 'ephapsys-sign'", "created kem key 'ephapsys-kem'"]
    assert stat.S_IMODE(pin_file.stat().st_mode) == 0o440 and pin_file.stat().st_gid == os.getgid()
    assert not pin_file.read_text().endswith("\n") and len(pin_file.read_text()) == 32
    assert not os.path.exists(str(pin_file) + prov_mod.SO_PENDING_SUFFIX)
    assert not [n for n in os.listdir(pin_file.parent) if n.startswith(prov_mod.TMP_PREFIX)]
    again = prov_mod.provision(_pcfg("prov-fresh", pin_file), pin_group=group)
    assert again["status"] == "already_provisioned" and again["actions"] == []
    assert (again["sign"], again["kem"]) == (first["sign"], first["kem"])
    rt = p11.Pkcs11Provider(_pcfg("prov-fresh", pin_file))                                  # runtime accepts the result
    assert spki_of(rt, "sign") == first["sign"]["spki_sha256"]


def test_provision_default_pin_file_is_owner_only(token_env, tmp_path):
    import stat
    pin_file = tmp_path / "pin"
    prov_mod.provision(_pcfg("prov-0400", pin_file))
    assert stat.S_IMODE(pin_file.stat().st_mode) == 0o400


def test_provision_keeps_existing_token_and_keys(token_env, tmp_path):
    pin_file = _secret(tmp_path / "pin", PIN)
    before = provider(token_env)
    report = prov_mod.provision(_pcfg(TOKEN, pin_file, sign="sign", kem="kem"))
    assert report["status"] == "already_provisioned"
    assert report["sign"]["spki_sha256"] == spki_of(before, "sign")
    assert report["kem"]["spki_sha256"] == spki_of(before, "kem")


def test_provision_refuses_labelled_token_without_pin_file(token_env, tmp_path):
    pin_file = tmp_path / "missing.pin"
    with pytest.raises(p11.Pkcs11Error, match="manual repair"):
        prov_mod.provision(_pcfg(TOKEN, pin_file, sign="sign", kem="kem"))
    assert not pin_file.exists()


def test_provision_wrong_pin_stops_then_corrected_file_works(token_env, tmp_path):
    pin_file = tmp_path / "pin"
    prov_mod.provision(_pcfg("prov-wrongpin", pin_file))
    good = pin_file.read_text()
    pin_file.chmod(0o600)
    pin_file.write_text("0" * 32)
    with pytest.raises(p11.Pkcs11Error, match="incorrect PIN"):
        prov_mod.provision(_pcfg("prov-wrongpin", pin_file))
    pin_file.write_text(good)
    assert prov_mod.provision(_pcfg("prov-wrongpin", pin_file))["status"] == "already_provisioned"


def test_provision_reuses_existing_pin_file_for_new_token(token_env, tmp_path):
    pin_file = _secret(tmp_path / "pin", "c0ffee" * 4)
    report = prov_mod.provision(_pcfg("prov-reuse", pin_file))
    assert report["actions"][0] == "reused existing PIN file"
    assert pin_file.read_text() == "c0ffee" * 4
    with pkcs11.lib(SOFTHSM_LIB).get_token(token_label="prov-reuse").open(user_pin="c0ffee" * 4):
        pass


def _pending(pin_file, so_pin, label, module=SOFTHSM_LIB):
    return _secret(pin_file.parent / (pin_file.name + prov_mod.SO_PENDING_SUFFIX),
                   json.dumps({"v": 1, "module": module, "label": label, "slot_id": 0, "so_pin": so_pin}), 0o400)


def _init_free_slot(label, so_pin):
    lib = pkcs11.lib(SOFTHSM_LIB)
    free = [s for s in lib.get_slots(token_present=True) if not (s.get_token().flags & pkcs11.TokenFlag.TOKEN_INITIALIZED)][0]
    prov_mod._c_init_token(SOFTHSM_LIB, free.slot_id, so_pin, label)


def test_provision_resumes_after_interrupted_init(token_env, tmp_path):
    """Token initialised but the user PIN never set: the recovery record finishes the job, then is discarded."""
    pin_file = _secret(tmp_path / "pin", "ab" * 16)
    _pending(pin_file, "cd" * 16, "prov-resume")
    _init_free_slot("prov-resume", "cd" * 16)
    report = prov_mod.provision(_pcfg("prov-resume", pin_file))
    assert "set the user PIN" in report["actions"] and "discarded the SO PIN" in report["actions"]
    assert not os.path.exists(str(pin_file) + prov_mod.SO_PENDING_SUFFIX)


def test_provision_refuses_recovery_record_for_another_label(token_env, tmp_path):
    pin_file = _secret(tmp_path / "pin", "ab" * 16)
    _pending(pin_file, "cd" * 16, "some-other-token")
    _init_free_slot("prov-otherrec", "cd" * 16)
    with pytest.raises(p11.Pkcs11Error, match="unfinished run"):
        prov_mod.provision(_pcfg("prov-otherrec", pin_file))


def test_provision_refuses_half_key_pair(token_env, tmp_path):
    pin_file = tmp_path / "pin"
    prov_mod.provision(_pcfg("prov-half", pin_file))
    with pkcs11.lib(SOFTHSM_LIB).get_token(token_label="prov-half").open(rw=True, user_pin=pin_file.read_text()) as s:
        s.get_key(object_class=pkcs11.ObjectClass.PUBLIC_KEY, label="ephapsys-kem").destroy()
    with pytest.raises(p11.Pkcs11Error, match="incomplete"):
        prov_mod.provision(_pcfg("prov-half", pin_file))


def test_provision_refuses_existing_key_with_bad_policy(token_env, tmp_path):
    pin_file = _secret(tmp_path / "pin", PIN)
    with pytest.raises(p11.Pkcs11Error, match="non-extractable"):
        prov_mod.provision(_pcfg(TOKEN, pin_file, sign="extractable", kem="kem"))


# ---- refusals before any write (Codex R227 findings 1, 2 and 4) ----------------------------------------
def test_provision_rejects_serial_before_touching_anything(token_env, tmp_path):
    before = _labels()
    with pytest.raises(p11.Pkcs11Error, match="PKCS11_TOKEN_SERIAL"):
        prov_mod.provision(_pcfg("prov-serial", tmp_path / "pin", token_serial="nope"))
    assert _labels() == before and not (tmp_path / "pin").exists()


def test_provision_rejects_long_label_before_writing(token_env, tmp_path):
    with pytest.raises(p11.Pkcs11Error, match="32 bytes"):
        prov_mod.provision(_pcfg("x" * 33, tmp_path / "pin"))
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("content", ["", "123", "a" * 65, "has space"])
def test_provision_rejects_bad_existing_pin_before_init(token_env, tmp_path, content):
    before = _labels()
    pin_file = _secret(tmp_path / "pin", content)
    with pytest.raises(p11.Pkcs11Error, match="printable ASCII"):
        prov_mod.provision(_pcfg("prov-badpin", pin_file))
    assert _labels() == before                                                               # no slot consumed


@pytest.mark.parametrize("mode", [0o666, 0o644, 0o640])
def test_provision_rejects_pin_file_readable_by_others(token_env, tmp_path, mode):
    pin_file = _secret(tmp_path / "pin", PIN, mode)
    with pytest.raises(p11.Pkcs11Error, match="group or others"):
        prov_mod.provision(_pcfg(TOKEN, pin_file, sign="sign", kem="kem"))


def test_provision_rejects_pin_file_without_requested_group_access(token_env, tmp_path):
    import grp
    pin_file = _secret(tmp_path / "pin", PIN, 0o400)
    with pytest.raises(p11.Pkcs11Error, match="group-readable"):
        prov_mod.provision(_pcfg(TOKEN, pin_file, sign="sign", kem="kem"),
                           pin_group=grp.getgrgid(os.getgid()).gr_name)


def test_provision_rejects_symlinked_pin_file(token_env, tmp_path):
    real = _secret(tmp_path / "real", PIN)
    (tmp_path / "pin").symlink_to(real)
    with pytest.raises(p11.Pkcs11Error, match="symlinks"):
        prov_mod.provision(_pcfg(TOKEN, tmp_path / "pin", sign="sign", kem="kem"))


def test_provision_never_follows_recovery_symlink(token_env, tmp_path):
    victim = tmp_path / "victim"
    victim.write_text("keep me")
    pin_file = _secret(tmp_path / "pin", PIN)
    (tmp_path / ("pin" + prov_mod.SO_PENDING_SUFFIX)).symlink_to(victim)
    with pytest.raises(p11.Pkcs11Error, match="symlinks"):
        prov_mod.provision(_pcfg(TOKEN, pin_file, sign="sign", kem="kem"))
    assert victim.read_text() == "keep me"


def test_provision_rejects_writable_directory(token_env, tmp_path):
    d = tmp_path / "open"
    d.mkdir()
    d.chmod(0o777)
    with pytest.raises(p11.Pkcs11Error, match="writable by group or others"):
        prov_mod.provision(_pcfg(TOKEN, d / "pin", sign="sign", kem="kem"))


def test_provision_unknown_group_is_a_refusal(token_env, tmp_path):
    with pytest.raises(p11.Pkcs11Error, match="unknown group"):
        prov_mod.provision(_pcfg(TOKEN, tmp_path / "pin"), pin_group="no-such-group-ephapsys")


def test_provision_removes_orphaned_temp_files(token_env, tmp_path):
    pin_file = _secret(tmp_path / "pin", PIN)
    orphan = _secret(tmp_path / (prov_mod.TMP_PREFIX + "abc"), "leftover")
    prov_mod.provision(_pcfg(TOKEN, pin_file, sign="sign", kem="kem"))
    assert not orphan.exists()


# ---- serialisation (R227 finding 3, R228) -----------------------------------------------------------------
def _hold(path):
    import fcntl
    fd = os.open(str(path), os.O_RDWR | os.O_CREAT, 0o600)
    fcntl.flock(fd, fcntl.LOCK_EX)
    return fd


def test_provision_refuses_another_pin_path_in_the_same_directory(token_env, tmp_path):
    """Run A holds the directory lock; run B with a different PIN file in that directory must not proceed
    (so it cannot remove A's temporary files)."""
    _secret(tmp_path / "pinB", PIN)
    live_tmp = _secret(tmp_path / (prov_mod.TMP_PREFIX + "runA"), "in flight")
    fd = _hold(tmp_path / prov_mod.DIR_LOCK_NAME)
    try:
        with pytest.raises(p11.Pkcs11Error, match="another provisioning run"):
            prov_mod.provision(_pcfg(TOKEN, tmp_path / "pinB", sign="sign", kem="kem"))
    finally:
        os.close(fd)
    assert live_tmp.exists()


def test_provision_refuses_a_second_run_on_the_same_module(token_env, tmp_path):
    (tmp_path / "a").mkdir(mode=0o700)
    pin_file = _secret(tmp_path / "a" / "pin", PIN)
    fd = _hold(prov_mod._module_lock_path(SOFTHSM_LIB))
    try:
        with pytest.raises(p11.Pkcs11Error, match="another provisioning run"):
            prov_mod.provision(_pcfg(TOKEN, pin_file, sign="sign", kem="kem"))
    finally:
        os.close(fd)


# ---- recovery record is validated before any delete or replace (R228) ----------------------------------
def test_provision_keeps_other_tokens_recovery_record_when_label_absent(token_env, tmp_path):
    pin_file = _secret(tmp_path / "pin", "ab" * 16)
    rec = _pending(pin_file, "cd" * 16, "interrupted-token-A")
    before_bytes, before_labels = rec.read_bytes(), _labels()
    with pytest.raises(p11.Pkcs11Error, match="unfinished run for token 'interrupted-token-A'"):
        prov_mod.provision(_pcfg("prov-label-B", pin_file))
    assert rec.read_bytes() == before_bytes and _labels() == before_labels


def test_provision_keeps_other_tokens_recovery_record_when_label_provisioned(token_env, tmp_path):
    pin_file = _secret(tmp_path / "pin", PIN)
    rec = _pending(pin_file, "cd" * 16, "interrupted-token-A")
    before = rec.read_bytes()
    with pytest.raises(p11.Pkcs11Error, match="unfinished run"):
        prov_mod.provision(_pcfg(TOKEN, pin_file, sign="sign", kem="kem"))
    assert rec.read_bytes() == before


def test_provision_keeps_recovery_record_for_another_module(token_env, tmp_path):
    pin_file = _secret(tmp_path / "pin", PIN)
    rec = _pending(pin_file, "cd" * 16, TOKEN, module="/usr/lib/other-module.so")
    with pytest.raises(p11.Pkcs11Error, match="unfinished run"):
        prov_mod.provision(_pcfg(TOKEN, pin_file, sign="sign", kem="kem"))
    assert rec.exists()


@pytest.mark.parametrize("body", ["not json", "[]", json.dumps({"v": 2, "module": "m", "label": "l", "slot_id": 0, "so_pin": "x"}),
                                  json.dumps({"v": 1, "module": "m", "label": "l", "slot_id": "0", "so_pin": "x"}),
                                  json.dumps({"v": 1, "module": "m", "label": "l", "slot_id": 0, "so_pin": ""})])
def test_provision_keeps_malformed_recovery_record(token_env, tmp_path, body):
    pin_file = _secret(tmp_path / "pin", PIN)
    rec = _secret(tmp_path / ("pin" + prov_mod.SO_PENDING_SUFFIX), body, 0o400)
    before_labels = _labels()
    with pytest.raises(p11.Pkcs11Error, match="malformed"):
        prov_mod.provision(_pcfg("prov-malformed", pin_file))
    assert rec.read_text() == body and _labels() == before_labels


# ---- slot choice and PIN flags, on a fake module (no SoftHSM state involved) -----------------------------
class _FakeToken:
    def __init__(self, slot_id, label="", flags=0, open_exc=None):
        self.label, self.flags, self.opens, self._exc = label, flags, [], open_exc
        mechs = {pkcs11.Mechanism.EC_KEY_PAIR_GEN, pkcs11.Mechanism.ECDH1_DERIVE, pkcs11.Mechanism.ECDSA_SHA256}
        self.slot = types.SimpleNamespace(slot_id=slot_id, get_token=lambda: self, get_mechanisms=lambda: mechs)

    def open(self, **kw):
        self.opens.append(kw)
        raise self._exc or AssertionError("unexpected login")


def _fake_lib(*tokens):
    return types.SimpleNamespace(get_slots=lambda token_present=True: [t.slot for t in tokens])


@pytest.fixture
def no_init(monkeypatch):
    calls = []
    monkeypatch.setattr(prov_mod, "_c_init_token", lambda *a: calls.append(a))
    return calls


def test_provision_never_initialises_an_initialised_token(tmp_path, no_init):
    TF = pkcs11.TokenFlag
    vendor = _FakeToken(0, "ManufacturingKey", TF.TOKEN_INITIALIZED | TF.USER_PIN_INITIALIZED)
    with pytest.raises(p11.Pkcs11Error, match="refusing to initialise"):
        prov_mod.provision(_pcfg("ephapsys", tmp_path / "pin"), lib_factory=lambda m: _fake_lib(vendor))
    assert no_init == [] and not (tmp_path / "pin").exists()


def test_provision_refuses_ambiguous_free_slots_unless_chosen(tmp_path, no_init):
    lib = _fake_lib(_FakeToken(2), _FakeToken(0), _FakeToken(1))
    with pytest.raises(p11.Pkcs11Error, match="choose one with --init-slot"):
        prov_mod.provision(_pcfg("ephapsys", tmp_path / "pin"), lib_factory=lambda m: lib)
    assert no_init == [] and not (tmp_path / "pin").exists()
    for choice, expected in (("first", 0), (2, 2)):
        with pytest.raises(p11.Pkcs11Error, match="not found after initialisation"):           # the fake never relabels
            prov_mod.provision(_pcfg("ephapsys", tmp_path / "pin"), init_slot=choice, lib_factory=lambda m: lib)
        assert no_init[-1][1] == expected
    with pytest.raises(p11.Pkcs11Error, match="slot 7 has no uninitialised token"):
        prov_mod.provision(_pcfg("ephapsys", tmp_path / "pin"), init_slot=7, lib_factory=lambda m: lib)


@pytest.mark.parametrize("flag", ["USER_PIN_LOCKED", "USER_PIN_FINAL_TRY"])
def test_provision_does_not_try_a_locked_or_final_try_pin(tmp_path, flag):
    TF = pkcs11.TokenFlag
    tok = _FakeToken(0, "ephapsys", TF.TOKEN_INITIALIZED | TF.USER_PIN_INITIALIZED | getattr(TF, flag))
    with pytest.raises(p11.Pkcs11Error, match="locked or on its final try"):
        prov_mod.provision(_pcfg("ephapsys", _secret(tmp_path / "pin", PIN)), lib_factory=lambda m: _fake_lib(tok))
    assert tok.opens == []


def test_provision_wrong_pin_is_tried_exactly_once(tmp_path):
    TF = pkcs11.TokenFlag
    tok = _FakeToken(0, "ephapsys", TF.TOKEN_INITIALIZED | TF.USER_PIN_INITIALIZED, open_exc=pkcs11.exceptions.PinIncorrect())
    with pytest.raises(p11.Pkcs11Error, match="incorrect PIN"):
        prov_mod.provision(_pcfg("ephapsys", _secret(tmp_path / "pin", PIN)), lib_factory=lambda m: _fake_lib(tok))
    assert len(tok.opens) == 1


@pytest.mark.parametrize("flag", ["SO_PIN_LOCKED", "SO_PIN_FINAL_TRY"])
def test_provision_recovery_does_not_try_a_locked_so_pin(tmp_path, flag):
    TF = pkcs11.TokenFlag
    tok = _FakeToken(0, "ephapsys", TF.TOKEN_INITIALIZED | getattr(TF, flag))
    pin_file = _secret(tmp_path / "pin", PIN)
    _pending(pin_file, "cd" * 16, "ephapsys")
    with pytest.raises(p11.Pkcs11Error, match="SO PIN .* final try"):
        prov_mod.provision(_pcfg("ephapsys", pin_file), lib_factory=lambda m: _fake_lib(tok))
    assert tok.opens == []


@pytest.mark.parametrize("over,needle", [
    (dict(pin_file=None, pin="1234"), "PKCS11_PIN_FILE"),
    (dict(sign_key_label=None, sign_key_id=b"\x01"), "by label"),
])
def test_provision_config_requirements(tmp_path, over, needle):
    import dataclasses
    cfg = dataclasses.replace(_pcfg("x", tmp_path / "pin"), **over)
    with pytest.raises(p11.Pkcs11Error, match=needle):
        prov_mod.provision(cfg)


def test_provision_rejects_bad_init_slot(tmp_path):
    with pytest.raises(p11.Pkcs11Error, match="init_slot"):
        prov_mod.provision(_pcfg("x", tmp_path / "pin"), init_slot=-1)
