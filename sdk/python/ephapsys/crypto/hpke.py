# SPDX-License-Identifier: Apache-2.0
# sdk/crypto/hpke.py
from __future__ import annotations
import base64, json, os
from typing import Optional, Callable, Union

from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.serialization import (
    load_pem_public_key, load_pem_private_key,
    Encoding, PublicFormat, PrivateFormat, NoEncryption
)
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM


HPKE_INFO = b"Ephapsys-HPKE-CEK-v1"
CURVE = ec.SECP256R1  # matches EcdsaSecp256r1VerificationKey2019


def _derive_key(shared_secret: bytes) -> bytes:
    return HKDF(algorithm=hashes.SHA256(), length=32, salt=None, info=HPKE_INFO).derive(shared_secret)


def wrap(pubkey_pem: str, cek: bytes) -> str:
    """
    Encrypt (wrap) a CEK for the holder of pubkey_pem (P-256).
    Returns a base64 URL-safe JSON blob containing epk + nonce + ct.
    """
    if not isinstance(cek, (bytes, bytearray)) or len(cek) != 32:
        raise ValueError("CEK must be 32 bytes")

    pub = load_pem_public_key(pubkey_pem.encode())
    if not isinstance(pub, ec.EllipticCurvePublicKey):
        raise ValueError("Public key must be EC (P-256)")

    # Ephemeral key
    epk = ec.generate_private_key(CURVE())
    shared = epk.exchange(ec.ECDH(), pub)
    k = _derive_key(shared)

    aes = AESGCM(k)
    nonce = os.urandom(12)
    ct = aes.encrypt(nonce, cek, None)

    epk_der = epk.public_key().public_bytes(Encoding.DER, PublicFormat.SubjectPublicKeyInfo)
    blob = {
        "v": 1,
        "alg": "HPKE-P256-AESGCM",
        "epk_der_b64": base64.b64encode(epk_der).decode(),
        "nonce_b64": base64.b64encode(nonce).decode(),
        "ct_b64": base64.b64encode(ct).decode()
    }
    return base64.urlsafe_b64encode(json.dumps(blob, separators=(",", ":")).encode()).decode()


def unwrap(
    wrapped_cek_b64: str,
    *,
    # one of the following three must be provided:
    privkey_pem: Optional[str] = None,
    privkey_loader: Optional[Callable[[], str]] = None,
    ecdh_with_ephemeral: Optional[Callable[[bytes], bytes]] = None,
) -> bytes:
    """
    Decrypt (unwrap) a CEK.

    Options:
    - privkey_pem: software private key in PEM (dev/test).
    - privkey_loader(): returns PEM (lazy loaded).
    - ecdh_with_ephemeral(epk_der)->shared_secret: for TPM/TEE where private key never leaves device.
    """
    blob = json.loads(base64.urlsafe_b64decode(wrapped_cek_b64).decode())
    if int(blob.get("v", 0)) != 1:
        raise ValueError("Unsupported SIE wrap version")
    if blob.get("alg") != "HPKE-P256-AESGCM":
        raise ValueError("Unsupported SIE wrap alg")

    epk_der = base64.b64decode(blob["epk_der_b64"])
    nonce = base64.b64decode(blob["nonce_b64"])
    ct = base64.b64decode(blob["ct_b64"])

    if ecdh_with_ephemeral:
        shared = ecdh_with_ephemeral(epk_der)
    else:
        pem = privkey_pem or (privkey_loader() if privkey_loader else None)
        if not pem:
            raise ValueError("No private key available for SIE unwrap")
        prv = load_pem_private_key(pem.encode(), password=None)
        if not isinstance(prv, ec.EllipticCurvePrivateKey):
            raise ValueError("Private key must be EC (P-256)")
        epk_pub = load_pem_public_key(_spki_der_to_pem(epk_der).encode())
        shared = prv.exchange(ec.ECDH(), epk_pub)

    k = _derive_key(shared)
    cek = AESGCM(k).decrypt(nonce, ct, None)
    if len(cek) != 32:
        raise ValueError("Invalid CEK size")
    return cek


def _spki_der_to_pem(der: bytes) -> str:
    # cheap DER->PEM for SubjectPublicKeyInfo
    b64 = base64.encodebytes(der).decode().replace("\n", "")
    lines = [b64[i:i+64] for i in range(0, len(b64), 64)]
    return "-----BEGIN PUBLIC KEY-----\n" + "\n".join(lines) + "\n-----END PUBLIC KEY-----\n"
