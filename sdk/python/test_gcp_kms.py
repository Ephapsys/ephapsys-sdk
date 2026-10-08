"""Cloud KMS provider tests against an in-memory KMS double (no network, no credentials)."""
import base64
import hashlib
import json
import os
from types import SimpleNamespace

import pytest
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec, rsa
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from ephapsys import storage
from ephapsys.crypto import gcp_kms as g

RING = "projects/p-test-1/locations/us-central1/keyRings/r/cryptoKeys/"
SIGN = RING + "sign/cryptoKeyVersions/1"
DECRYPT = RING + "decrypt/cryptoKeyVersions/1"


def _pem(key):
    return key.public_key().public_bytes(serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo).decode()


class FakeKms:
    """Behaves like KeyManagementServiceClient for the calls the provider makes, including CRC32C fields."""

    def __init__(self, protection="HSM"):
        self.keys = {SIGN: ec.generate_private_key(ec.SECP256R1()),
                     DECRYPT: rsa.generate_private_key(public_exponent=65537, key_size=3072)}
        self.protection = protection
        self.corrupt = None

    def list_crypto_key_versions(self, request):
        assert request["filter"] == "state=ENABLED"
        found = [n for n in self.keys if n.startswith(request["parent"] + "/cryptoKeyVersions/")]
        return [SimpleNamespace(name=n, state=SimpleNamespace(name="ENABLED")) for n in found]

    def get_public_key(self, request):
        key = self.keys[request["name"]]
        pem = _pem(key)
        alg = g.SIGN_ALGORITHM if isinstance(key, ec.EllipticCurvePrivateKey) else g.DECRYPT_ALGORITHM
        return SimpleNamespace(name=request["name"], pem=pem, pem_crc32c=g.crc32c(pem.encode()),
                               protection_level=SimpleNamespace(name=self.protection), algorithm=SimpleNamespace(name=alg))

    def asymmetric_sign(self, request):
        digest = request["digest"]["sha256"]
        assert request["digest_crc32c"] == g.crc32c(digest)
        from cryptography.hazmat.primitives.asymmetric.utils import Prehashed
        sig = self.keys[request["name"]].sign(digest, ec.ECDSA(Prehashed(hashes.SHA256())))
        if self.corrupt == "sign":
            sig = sig[:-1] + bytes([sig[-1] ^ 1])
        return SimpleNamespace(signature=sig, signature_crc32c=g.crc32c(sig), verified_digest_crc32c=True,
                               name=request["name"], protection_level=SimpleNamespace(name=self.protection))

    def asymmetric_decrypt(self, request):
        assert request["ciphertext_crc32c"] == g.crc32c(request["ciphertext"])
        pt = self.keys[request["name"]].decrypt(request["ciphertext"], g.oaep())
        crc = g.crc32c(pt) ^ (1 if self.corrupt == "decrypt" else 0)
        return SimpleNamespace(plaintext=pt, plaintext_crc32c=crc, verified_ciphertext_crc32c=True,
                               protection_level=SimpleNamespace(name=self.protection))

    def get_crypto_key_version(self, request):
        chains = SimpleNamespace(cavium_certs=["C1", "C2"], google_card_certs=["G1"], google_partition_certs=["G2"])
        return SimpleNamespace(attestation=SimpleNamespace(format=SimpleNamespace(name="CAVIUM_V2_COMPRESSED"),
                                                           content=b"att:" + request["name"].encode(), cert_chains=chains))


def provider(kms=None, token=None):
    kms = kms or FakeKms()
    claims = base64.urlsafe_b64encode(json.dumps({"sub": "4242"}).encode()).decode().rstrip("=")
    tok = token or f"h.{claims}.s"
    return g.GcpKmsProvider(SIGN, DECRYPT, client=kms, http_get=lambda url, headers: tok), kms


def test_crc32c_reference_vector():
    assert g.crc32c(b"123456789") == 0xE3069283


def test_sign_is_hsm_backed_integrity_checked_and_verified():
    prov, kms = provider()
    sig = prov.sign(b"hello")
    kms.keys[SIGN].public_key().verify(sig, b"hello", ec.ECDSA(hashes.SHA256()))
    kms.corrupt = "sign"
    with pytest.raises(g.GcpKmsError):
        prov.sign(b"hello")


def test_software_protection_level_is_refused():
    prov, kms = provider(FakeKms(protection="SOFTWARE"))
    with pytest.raises(g.GcpKmsError, match="Cloud HSM"):
        prov.public_key_pem("sign")


def test_decrypt_integrity():
    prov, kms = provider()
    ct = kms.keys[DECRYPT].public_key().encrypt(b"k" * 32, g.oaep())
    assert prov.decrypt(ct) == b"k" * 32
    kms.corrupt = "decrypt"
    with pytest.raises(g.GcpKmsError, match="CRC32C"):
        prov.decrypt(ct)


def test_key_names_resolve_to_newest_enabled_version_and_must_differ():
    kms = FakeKms()
    prov = g.GcpKmsProvider(RING + "sign", RING + "decrypt", client=kms, http_get=lambda *a: "t")
    assert prov.sign_key == SIGN and prov.decrypt_key == DECRYPT
    kms.keys[RING + "sign/cryptoKeyVersions/10"] = ec.generate_private_key(ec.SECP256R1())
    kms.keys[RING + "sign/cryptoKeyVersions/9"] = ec.generate_private_key(ec.SECP256R1())
    assert g.GcpKmsProvider(RING + "sign", DECRYPT, client=kms).sign_key == RING + "sign/cryptoKeyVersions/10"
    with pytest.raises(g.GcpKmsError, match="no enabled version"):
        g.GcpKmsProvider(RING + "missing", DECRYPT, client=kms)
    with pytest.raises(g.GcpKmsError):
        g.GcpKmsProvider(SIGN, SIGN, client=kms)


def test_evidence_matches_the_wire_contract():
    prov, kms = provider()
    tok = prov.workload_token("https://aoc.test")
    ev = prov.evidence(nonce_b64="bm9uY2U=", org_id="org", template_id="tid", device_id="cloud-1", token=tok)
    t = ev["transcript"]
    assert set(t) == set(g.TRANSCRIPT_FIELDS) and t["domain"] == "ephapsys-hsm-personalize-v1"
    assert t["workload_sub"] == "4242" and t["sign_key"] == SIGN and t["old_generation"] == "none"
    msg = json.dumps(t, sort_keys=True, separators=(",", ":")).encode()
    kms.keys[SIGN].public_key().verify(base64.b64decode(ev["sig_b64"]), msg, ec.ECDSA(hashes.SHA256()))
    assert ev["attestations"]["sign"]["cavium_certs_pem"] == "C1C2"
    assert t["decrypt_spki_sha256"] == g.spki_sha256(_pem(kms.keys[DECRYPT]))


def test_rotation_evidence_is_cosigned_by_the_old_key():
    prov, kms = provider()
    old = RING + "sign/cryptoKeyVersions/0"
    kms.keys[old] = ec.generate_private_key(ec.SECP256R1())
    ev = prov.evidence(nonce_b64="bg==", org_id="o", template_id="t", device_id="d", token=prov.workload_token("a"),
                       mode="rotate", old_generation="g1", old_sign_key=old, old_sign_spki="x" * 64)
    msg = g.transcript_bytes(ev["transcript"])
    kms.keys[old].public_key().verify(base64.b64decode(ev["rotation_sig_b64"]), msg, ec.ECDSA(hashes.SHA256()))


def test_transcript_rejects_bad_values():
    prov, _ = provider()
    with pytest.raises(g.GcpKmsError):
        prov.evidence(nonce_b64="has space", org_id="o", template_id="t", device_id="d", token=prov.workload_token("a"))


def _envelope(kms, cek, aad, recipient=DECRYPT, **over):
    env = {"v": 2, "alg": "RSA-OAEP-3072-SHA256", "mgf1": "SHA256", "label": "", "recipient_key": recipient,
           "recipient_spki_sha256": aad["recipient_spki_sha256"], "generation": aad["generation"], "aad": aad,
           "wrapped_cek_b64": base64.b64encode(kms.keys[DECRYPT].public_key().encrypt(cek, g.oaep())).decode()}
    env.update(over)
    return base64.urlsafe_b64encode(json.dumps(env, sort_keys=True, separators=(",", ":")).encode()).decode()


def test_envelope_v2_roundtrip_and_binding():
    prov, kms = provider()
    spki = g.spki_sha256(_pem(kms.keys[DECRYPT]))
    aad = {"v": 2, "instance_id": "i", "model_id": "m1", "model_public_id": "model_inst_x", "generation": "g",
           "recipient_spki_sha256": spki}
    cek = os.urandom(32)
    nonce = os.urandom(12)
    blob = AESGCM(cek).encrypt(nonce, b"ecm", json.dumps(aad, sort_keys=True, separators=(",", ":")).encode())
    opened = prov.unwrap_envelope(_envelope(kms, cek, aad), expect_model_ids=["model_inst_x"])
    assert AESGCM(opened["cek"]).decrypt(nonce, blob, opened["aad"]) == b"ecm"
    with pytest.raises(g.GcpKmsError, match="another model"):
        prov.unwrap_envelope(_envelope(kms, cek, aad), expect_model_ids=["other"])
    with pytest.raises(g.GcpKmsError, match="another key"):
        prov.unwrap_envelope(_envelope(kms, cek, aad, recipient=RING + "x/cryptoKeyVersions/1"))
    with pytest.raises(g.GcpKmsError, match="generation"):
        prov.unwrap_envelope(_envelope(kms, cek, aad, generation="other"))
    with pytest.raises(g.GcpKmsError, match="unsupported"):
        prov.unwrap_envelope(_envelope(kms, cek, aad, alg="RSA-OAEP-3072-SHA1"))


def test_workload_token_from_metadata_or_file(tmp_path, monkeypatch):
    calls = []
    prov = g.GcpKmsProvider(SIGN, DECRYPT, client=FakeKms(), http_get=lambda url, h: calls.append((url, h)) or "tok")
    assert prov.workload_token("https://api.example.com") == "tok"
    assert "audience=https%3A%2F%2Fapi.example.com&format=full" in calls[0][0]
    assert calls[0][1] == {"Metadata-Flavor": "Google"}
    f = tmp_path / "t"
    f.write_text("filetok\n")
    monkeypatch.setenv("EPHAPSYS_WORKLOAD_TOKEN_FILE", str(f))
    assert prov.workload_token("x") == "filetok"
    monkeypatch.delenv("EPHAPSYS_WORKLOAD_TOKEN_FILE")
    monkeypatch.setenv("EPHAPSYS_WORKLOAD_AUDIENCE", "https://aud")
    assert g.GcpKmsProvider.audience("https://api.example.com/v1") == "https://aud"
    monkeypatch.delenv("EPHAPSYS_WORKLOAD_AUDIENCE")
    assert g.GcpKmsProvider.audience("https://api.example.com/v1") == "https://api.example.com"


# ---------------------------------------------------------------------------------------------- storage

@pytest.fixture
def kms_storage(monkeypatch, tmp_path):
    prov, kms = provider()
    monkeypatch.setenv("HSM_KMS_KEY", SIGN)
    monkeypatch.setenv("HSM_KMS_DECRYPT_KEY", DECRYPT)
    monkeypatch.delenv("PKCS11_MODULE", raising=False)
    monkeypatch.delenv("EPHAPSYS_KEY_PROVIDER", raising=False)
    monkeypatch.setattr(g.GcpKmsProvider, "shared", classmethod(lambda cls: prov))
    return prov, kms, str(tmp_path)


def test_storage_roundtrip_with_cloud_kms(kms_storage):
    prov, kms, state = kms_storage
    assert storage.key_provider() == "gcp-kms"
    storage.write_encrypted(state, "ecm.bin", b"lambda")
    assert storage.read_encrypted(state, "ecm.bin") == b"lambda"
    rec = json.load(open(os.path.join(state, "sealed_dek.gcpkms.json")))
    assert rec["decrypt_key"] == DECRYPT and "wrapped_dek" in rec


def test_storage_discards_cache_after_decrypt_key_rotation(kms_storage):
    prov, kms, state = kms_storage
    storage.write_encrypted(state, "ecm.bin", b"lambda")
    kms.keys[DECRYPT] = rsa.generate_private_key(public_exponent=65537, key_size=3072)
    prov._pem.clear()
    with pytest.raises(FileNotFoundError):
        storage.read_encrypted(state, "ecm.bin")
    storage.write_encrypted(state, "ecm.bin", b"lambda2")
    assert storage.read_encrypted(state, "ecm.bin") == b"lambda2"


def test_storage_refuses_ambiguous_or_switched_providers(kms_storage, monkeypatch):
    prov, kms, state = kms_storage
    monkeypatch.setenv("PKCS11_MODULE", "/lib/x.so")
    with pytest.raises(storage.KeyProviderError, match="both"):
        storage.key_provider()
    monkeypatch.delenv("PKCS11_MODULE")
    storage.write_encrypted(state, "a", b"x")
    monkeypatch.setenv("EPHAPSYS_KEY_PROVIDER", "pkcs11")
    with pytest.raises(storage.KeyProviderError, match="different key provider"):
        storage.write_encrypted(state, "a", b"x")
