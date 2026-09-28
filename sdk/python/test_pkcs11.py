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
        ec_pair("secp256r1", b"\x01", "sign", {Attribute.SIGN: True})
        ec_pair("secp256r1", b"\x02", "kem", {Attribute.DERIVE: True})
        s.generate_keypair(KeyType.RSA, 2048, store=True, id=b"\x03", label="rsa-sign",
                           private_template={Attribute.SENSITIVE: True, Attribute.EXTRACTABLE: False, Attribute.SIGN: True})
        ec_pair("secp256r1", b"\x04", "kem-other", {Attribute.DERIVE: True})
        ec_pair("secp256r1", b"\x05", "extractable", {Attribute.SIGN: True, Attribute.SENSITIVE: False, Attribute.EXTRACTABLE: True})
        ec_pair("secp256r1", b"\x06", "dup", {Attribute.SIGN: True})
        ec_pair("secp256r1", b"\x07", "dup", {Attribute.SIGN: True})
        ec_pair("secp384r1", b"\x08", "kem-p384", {Attribute.DERIVE: True})
        ec_pair("secp256r1", b"\x09", "no-sign-perm", {Attribute.SIGN: False})
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


def test_kem_must_be_p256(token_env):
    with pytest.raises(p11.Pkcs11Error, match="P-256"):
        provider(token_env, PKCS11_KEM_KEY_ID="08").public_key_pem("kem")


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
