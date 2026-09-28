# SPDX-License-Identifier: Apache-2.0
# sdk/storage.py
"""
Encrypted-at-rest cache for sensitive artifacts (e.g. the ECM / Λ).

The 32-byte DEK is protected by a key provider:
  - "tpm"    (default): PCR-bound TPM seal/unseal (crypto.tpm). Unchanged format.
  - "pkcs11": ECIES (hpke v1) to the token's non-extractable EC P-256 KEM key; unwrap via
              CKM_ECDH1_DERIVE on the token. Cache entries carry authenticated metadata (AES-GCM AAD).

Selection: EPHAPSYS_KEY_PROVIDER=tpm|pkcs11 if set, else "pkcs11" when PKCS11_MODULE is set, else "tpm".
A DEK sealed by one provider is never silently reused or replaced by another (fail closed).
The DEK and decrypted plaintext still live in ordinary process memory while in use.
"""
from __future__ import annotations
import base64, json, os, hashlib
from typing import Tuple
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from .crypto import tpm

PROVIDERS = ("tpm", "pkcs11")
_PKCS11_DEK_FILE = "sealed_dek.pkcs11.json"
_TPM_DEK_FILE = "sealed_dek.b64"
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
    return "pkcs11" if (os.getenv("PKCS11_MODULE") or "").strip() else "tpm"


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
    other = _PKCS11_DEK_FILE if provider == "tpm" else _TPM_DEK_FILE
    if os.path.exists(os.path.join(state_dir, other)):
        raise KeyProviderError(
            f"state_dir has a DEK sealed by a different key provider ({other}); refusing to switch providers "
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


# ---------------- Public API ----------------
def ensure_sealed_dek(state_dir: str) -> str:
    """Create or load the sealed DEK for the active provider. Returns an opaque serialized record."""
    provider = key_provider()
    _refuse_other_provider(state_dir, provider)
    if provider == "tpm":
        return _ensure_tpm_dek(state_dir)
    return json.dumps(_ensure_pkcs11_dek(state_dir), sort_keys=True)


def _unsealed_dek(state_dir: str) -> bytes:
    provider = key_provider()
    _refuse_other_provider(state_dir, provider)
    if provider == "tpm":
        return tpm.unseal(_ensure_tpm_dek(state_dir))
    return _pkcs11_unwrap(_ensure_pkcs11_dek(state_dir))


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
        rec = _ensure_pkcs11_dek(state_dir)
        dek = _pkcs11_unwrap(rec)
        ct = AESGCM(dek).encrypt(nonce, plaintext, _pkcs11_aad(name, rec["kem_spki_sha256"]))
        meta = {"alg": "AES-256-GCM", "v": _PKCS11_CACHE_VERSION, "provider": "pkcs11",
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
    enc_path, meta_path = _paths(state_dir, name)
    meta = json.load(open(meta_path, "r"))
    nonce = base64.b64decode(meta["nonce_b64"])
    ct = open(enc_path, "rb").read()
    if provider == "tpm":
        if meta.get("provider") not in (None, "tpm"):
            raise KeyProviderError("cache entry was written by a different key provider")
        dek = tpm.unseal(_ensure_tpm_dek(state_dir))
        return AESGCM(dek).decrypt(nonce, ct, None)
    rec = _ensure_pkcs11_dek(state_dir)
    if meta.get("provider") != "pkcs11" or meta.get("v") != _PKCS11_CACHE_VERSION or meta.get("name") != name \
            or meta.get("kem_spki_sha256") != rec["kem_spki_sha256"]:
        raise KeyProviderError("cache metadata does not match the active pkcs11 key provider")
    dek = _pkcs11_unwrap(rec)
    return AESGCM(dek).decrypt(nonce, ct, _pkcs11_aad(name, rec["kem_spki_sha256"]))


def has_encrypted(state_dir: str, name: str) -> bool:
    enc_path = os.path.join(state_dir, "cache", name + ".enc")
    meta_path = os.path.join(state_dir, "cache", name + ".meta.json")
    return os.path.exists(enc_path) and os.path.exists(meta_path)
