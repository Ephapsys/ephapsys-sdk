# SPDX-License-Identifier: Apache-2.0
# sdk/storage.py
from __future__ import annotations
import base64, json, os, hashlib
from typing import Optional, Tuple
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from .crypto import tpm


def _paths(state_dir: str, name: str) -> Tuple[str, str]:
    cache = os.path.join(state_dir, "cache")
    os.makedirs(cache, exist_ok=True)
    return os.path.join(cache, name + ".enc"), os.path.join(cache, name + ".meta.json")


def ensure_sealed_dek(state_dir: str) -> str:
    """Create or load sealed DEK; returns sealed blob (base64 or 'INSECURE::<b64>')."""
    sec_path = os.path.join(state_dir, "sealed_dek.b64")
    if os.path.exists(sec_path):
        return open(sec_path, "r").read().strip()
    dek = os.urandom(32)
    sealed = tpm.seal(dek)
    with open(sec_path, "w") as f:
        f.write(sealed)
    return sealed


def _unsealed_dek(state_dir: str) -> bytes:
    sealed = ensure_sealed_dek(state_dir)
    return tpm.unseal(sealed)


def sha256_hex(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def write_encrypted(state_dir: str, name: str, plaintext: bytes) -> Tuple[str, str]:
    """
    Encrypts bytes to cache/<name>.enc with sidecar .meta.json.
    Returns (enc_path, meta_path).
    """
    dek = _unsealed_dek(state_dir)
    aes = AESGCM(dek)
    nonce = os.urandom(12)
    ct = aes.encrypt(nonce, plaintext, None)

    enc_path, meta_path = _paths(state_dir, name)
    with open(enc_path, "wb") as f:
        f.write(ct)
    with open(meta_path, "w") as f:
        json.dump({"alg": "AES-256-GCM", "nonce_b64": base64.b64encode(nonce).decode(), "len": len(plaintext)}, f)
    return enc_path, meta_path


def read_encrypted(state_dir: str, name: str) -> bytes:
    """Reads cache/<name>.enc + .meta.json, returns plaintext bytes (in memory)."""
    enc_path, meta_path = _paths(state_dir, name)
    meta = json.load(open(meta_path, "r"))
    nonce = base64.b64decode(meta["nonce_b64"])
    ct = open(enc_path, "rb").read()
    dek = _unsealed_dek(state_dir)
    return AESGCM(dek).decrypt(nonce, ct, None)

def has_encrypted(state_dir: str, name: str) -> bool:
    enc_path = os.path.join(state_dir, "cache", name + ".enc")
    meta_path = os.path.join(state_dir, "cache", name + ".meta.json")
    return os.path.exists(enc_path) and os.path.exists(meta_path)
