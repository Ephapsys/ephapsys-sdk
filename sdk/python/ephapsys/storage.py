# SPDX-License-Identifier: Apache-2.0
# sdk/storage.py
"""
Encrypted-at-rest cache for sensitive artifacts (e.g. the ECM / Λ).

The 32-byte DEK is protected by a key provider:
  - "tpm"    (default): PCR-bound TPM seal/unseal (crypto.tpm). Unchanged format.
  - "pkcs11": ECIES (hpke v1) to the token's non-extractable EC P-256 KEM key; unwrap via
              CKM_ECDH1_DERIVE on the token. Cache entries carry authenticated metadata (AES-GCM AAD).
  - "gcp-kms": RSA-OAEP-SHA256 to the Cloud HSM decrypt key (HSM_KMS_DECRYPT_KEY); unwrap via Cloud KMS
              AsymmetricDecrypt. Same authenticated cache metadata. After a key rotation the old cache is
              discarded (it is only a cache; the old key version may already be destroyed).

Selection: EPHAPSYS_KEY_PROVIDER=tpm|pkcs11|gcp-kms if set, else "gcp-kms" when HSM_KMS_KEY and
HSM_KMS_DECRYPT_KEY are set, else "pkcs11" when PKCS11_MODULE is set, else "tpm".
A DEK sealed by one provider is never silently reused or replaced by another (fail closed).
The DEK and decrypted plaintext still live in ordinary process memory while in use.
"""
from __future__ import annotations
import base64, json, os, hashlib
from typing import Tuple
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from .crypto import tpm

PROVIDERS = ("tpm", "pkcs11", "gcp-kms")
_PKCS11_DEK_FILE = "sealed_dek.pkcs11.json"
_KMS_DEK_FILE = "sealed_dek.gcpkms.json"
_TPM_DEK_FILE = "sealed_dek.b64"
_DEK_FILES = {"tpm": _TPM_DEK_FILE, "pkcs11": _PKCS11_DEK_FILE, "gcp-kms": _KMS_DEK_FILE}
_PKCS11_CACHE_VERSION = 2

_provider_override = None  # tests may inject a Pkcs11Provider-like object


class KeyProviderError(RuntimeError):
    pass


def set_pkcs11_provider(provider) -> None:
    """Inject a provider instance (tests / advanced integrations). None resets to env construction."""
    global _provider_override
    _provider_override = provider


def key_provider() -> str:
    explicit = (os.getenv("EPHAPSYS_KEY_PROVIDER") or "").strip().lower()
    if explicit:
        if explicit not in PROVIDERS:
            raise KeyProviderError(f"EPHAPSYS_KEY_PROVIDER must be one of {PROVIDERS} (got {explicit!r})")
        return explicit
    from .crypto.gcp_kms import GcpKmsProvider
    kms, p11 = GcpKmsProvider.configured(), bool((os.getenv("PKCS11_MODULE") or "").strip())
    if kms and p11:
        raise KeyProviderError("both Cloud KMS and PKCS11_MODULE are configured; set EPHAPSYS_KEY_PROVIDER")
    return "gcp-kms" if kms else "pkcs11" if p11 else "tpm"


def _kms():
    from .crypto.gcp_kms import GcpKmsProvider
    return GcpKmsProvider.shared()


def _pkcs11():
    if _provider_override is not None:
        return _provider_override
    from .crypto.pkcs11 import Pkcs11Provider
    return Pkcs11Provider.from_env()


def _paths(state_dir: str, name: str) -> Tuple[str, str]:
    cache = os.path.join(state_dir, "cache")
    os.makedirs(cache, exist_ok=True)
    return os.path.join(cache, name + ".enc"), os.path.join(cache, name + ".meta.json")


def _refuse_other_provider(state_dir: str, provider: str) -> None:
    for other, fname in _DEK_FILES.items():
        if other != provider and os.path.exists(os.path.join(state_dir, fname)):
            raise KeyProviderError(
                f"state_dir has a DEK sealed by a different key provider ({fname}); refusing to switch providers "
                f"silently. Clear the cache or set EPHAPSYS_KEY_PROVIDER to the original provider."
            )


# ---------------- TPM (unchanged format) ----------------
def _ensure_tpm_dek(state_dir: str) -> str:
    sec_path = os.path.join(state_dir, _TPM_DEK_FILE)
    if os.path.exists(sec_path):
        return open(sec_path, "r").read().strip()
    dek = os.urandom(32)
    sealed = tpm.seal(dek)
    with open(sec_path, "w") as f:
        f.write(sealed)
    return sealed


# ---------------- PKCS#11 ----------------
def _ensure_pkcs11_dek(state_dir: str) -> dict:
    from .crypto import hpke
    from .crypto.pkcs11 import spki_sha256_hex
    path = os.path.join(state_dir, _PKCS11_DEK_FILE)
    prov = _pkcs11()
    kem_pem = prov.public_key_pem("kem")
    kem_hash = spki_sha256_hex(kem_pem)
    if os.path.exists(path):
        rec = json.load(open(path, "r"))
        if rec.get("v") != 1 or rec.get("provider") != "pkcs11":
            raise KeyProviderError("unsupported sealed DEK record")
        if rec.get("kem_spki_sha256") != kem_hash:
            raise KeyProviderError("sealed DEK was wrapped to a different token KEM key; refusing to use it")
        return rec
    rec = {"v": 1, "provider": "pkcs11", "kem_spki_sha256": kem_hash, "kem_key_id_hex": prov.key_id_hex("kem"),
           "wrapped_dek": hpke.wrap(kem_pem, os.urandom(32))}
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(rec, f)
    os.replace(tmp, path)
    return rec


def _pkcs11_unwrap(rec: dict) -> bytes:
    from .crypto import hpke
    prov = _pkcs11()
    dek = hpke.unwrap(rec["wrapped_dek"], ecdh_with_ephemeral=prov.ecdh)
    if len(dek) != 32:
        raise KeyProviderError("invalid DEK length")
    return dek


def _pkcs11_aad(name: str, kem_hash: str) -> bytes:
    return json.dumps({"v": _PKCS11_CACHE_VERSION, "provider": "pkcs11", "kem_spki_sha256": kem_hash, "name": name},
                      sort_keys=True, separators=(",", ":")).encode()


# ---------------- Cloud KMS ----------------
def _ensure_kms_dek(state_dir: str) -> dict:
    from .crypto.gcp_kms import spki_sha256
    path = os.path.join(state_dir, _KMS_DEK_FILE)
    prov = _kms()
    kem_hash = spki_sha256(prov.public_key_pem("decrypt"))
    if os.path.exists(path):
        rec = json.load(open(path, "r"))
        if rec.get("v") != 1 or rec.get("provider") != "gcp-kms":
            raise KeyProviderError("unsupported sealed DEK record")
        if rec.get("kem_spki_sha256") == kem_hash:
            return rec
        _discard_cache(state_dir)                    # decrypt key rotated: the old cache is unreadable by design
    rec = {"v": 1, "provider": "gcp-kms", "kem_spki_sha256": kem_hash, "decrypt_key": prov.decrypt_key,
           "wrapped_dek": prov.wrap_dek(os.urandom(32))}
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(rec, f)
    os.replace(tmp, path)
    return rec


def _discard_cache(state_dir: str) -> None:
    cache = os.path.join(state_dir, "cache")
    if os.path.isdir(cache):
        for name in os.listdir(cache):
            if name.endswith(".enc") or name.endswith(".meta.json"):
                os.remove(os.path.join(cache, name))
    path = os.path.join(state_dir, _KMS_DEK_FILE)
    if os.path.exists(path):
        os.remove(path)


def _kms_unwrap(rec: dict) -> bytes:
    dek = _kms().decrypt(base64.b64decode(rec["wrapped_dek"]))
    if len(dek) != 32:
        raise KeyProviderError("invalid DEK length")
    return dek


def _sealed(state_dir: str, provider: str) -> Tuple[dict, bytes]:
    """(record, DEK) for the token-backed providers."""
    if provider == "gcp-kms":
        rec = _ensure_kms_dek(state_dir)
        return rec, _kms_unwrap(rec)
    rec = _ensure_pkcs11_dek(state_dir)
    return rec, _pkcs11_unwrap(rec)


def _aad(provider: str, name: str, kem_hash: str) -> bytes:
    if provider == "pkcs11":
        return _pkcs11_aad(name, kem_hash)
    return json.dumps({"v": _PKCS11_CACHE_VERSION, "provider": provider, "kem_spki_sha256": kem_hash, "name": name},
                      sort_keys=True, separators=(",", ":")).encode()


# ---------------- Public API ----------------
def ensure_sealed_dek(state_dir: str) -> str:
    """Create or load the sealed DEK for the active provider. Returns an opaque serialized record."""
    provider = key_provider()
    _refuse_other_provider(state_dir, provider)
    if provider == "tpm":
        return _ensure_tpm_dek(state_dir)
    if provider == "gcp-kms":
        return json.dumps(_ensure_kms_dek(state_dir), sort_keys=True)
    return json.dumps(_ensure_pkcs11_dek(state_dir), sort_keys=True)


def _unsealed_dek(state_dir: str) -> bytes:
    provider = key_provider()
    _refuse_other_provider(state_dir, provider)
    if provider == "tpm":
        return tpm.unseal(_ensure_tpm_dek(state_dir))
    return _sealed(state_dir, provider)[1]


def sha256_hex(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def write_encrypted(state_dir: str, name: str, plaintext: bytes) -> Tuple[str, str]:
    """
    Encrypts bytes to cache/<name>.enc with sidecar .meta.json.
    Returns (enc_path, meta_path).
    """
    provider = key_provider()
    _refuse_other_provider(state_dir, provider)
    nonce = os.urandom(12)
    if provider == "tpm":
        dek = tpm.unseal(_ensure_tpm_dek(state_dir))
        ct = AESGCM(dek).encrypt(nonce, plaintext, None)
        meta = {"alg": "AES-256-GCM", "nonce_b64": base64.b64encode(nonce).decode(), "len": len(plaintext)}
    else:
        rec, dek = _sealed(state_dir, provider)
        ct = AESGCM(dek).encrypt(nonce, plaintext, _aad(provider, name, rec["kem_spki_sha256"]))
        meta = {"alg": "AES-256-GCM", "v": _PKCS11_CACHE_VERSION, "provider": provider,
                "kem_spki_sha256": rec["kem_spki_sha256"], "name": name,
                "nonce_b64": base64.b64encode(nonce).decode(), "len": len(plaintext)}

    enc_path, meta_path = _paths(state_dir, name)
    with open(enc_path, "wb") as f:
        f.write(ct)
    with open(meta_path, "w") as f:
        json.dump(meta, f)
    return enc_path, meta_path


def read_encrypted(state_dir: str, name: str) -> bytes:
    """Reads cache/<name>.enc + .meta.json, returns plaintext bytes (in memory)."""
    provider = key_provider()
    _refuse_other_provider(state_dir, provider)
    sealed = _sealed(state_dir, provider) if provider != "tpm" else None   # may discard a pre-rotation cache
    enc_path, meta_path = _paths(state_dir, name)
    meta = json.load(open(meta_path, "r"))
    nonce = base64.b64decode(meta["nonce_b64"])
    ct = open(enc_path, "rb").read()
    if provider == "tpm":
        if meta.get("provider") not in (None, "tpm"):
            raise KeyProviderError("cache entry was written by a different key provider")
        dek = tpm.unseal(_ensure_tpm_dek(state_dir))
        return AESGCM(dek).decrypt(nonce, ct, None)
    rec, dek = sealed
    if meta.get("provider") != provider or meta.get("v") != _PKCS11_CACHE_VERSION or meta.get("name") != name \
            or meta.get("kem_spki_sha256") != rec["kem_spki_sha256"]:
        raise KeyProviderError(f"cache metadata does not match the active {provider} key provider")
    return AESGCM(dek).decrypt(nonce, ct, _aad(provider, name, rec["kem_spki_sha256"]))


def has_encrypted(state_dir: str, name: str) -> bool:
    enc_path = os.path.join(state_dir, "cache", name + ".enc")
    meta_path = os.path.join(state_dir, "cache", name + ".meta.json")
    return os.path.exists(enc_path) and os.path.exists(meta_path)
