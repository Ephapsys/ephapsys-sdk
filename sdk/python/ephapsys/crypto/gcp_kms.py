# SPDX-License-Identifier: Apache-2.0
"""
Google Cloud KMS / Cloud HSM provider for cloud workloads.

Two Cloud HSM keys per workload, both with protection level HSM:
  - sign:    EC_SIGN_P256_SHA256           (HSM_KMS_KEY)          signs the personalization transcript and
                                                                   device-auth challenges.
  - decrypt: RSA_DECRYPT_OAEP_3072_SHA256  (HSM_KMS_DECRYPT_KEY)  receives secure-ECM content keys
                                                                   (RSA-OAEP, SHA-256, MGF1-SHA-256, empty label)
                                                                   and protects the at-rest cache DEK.
Key names may be exact CryptoKeyVersion names, or CryptoKey names (the newest ENABLED version is used).

The workload identity is a Google-signed ID token for the service account the workload runs as (on GKE: Workload
Identity), fetched from the metadata server with the AOC as audience (EPHAPSYS_WORKLOAD_AUDIENCE, default: the AOC
base URL origin).

Every KMS response is integrity-checked (CRC32C of request and response) and must report protection level HSM.
Long-term private keys never leave Cloud HSM; a content key returned by AsymmetricDecrypt exists in process memory.

Requires the optional dependency `google-cloud-kms` (pip install 'ephapsys[hsm]').
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
from typing import Any, Callable, Dict, Optional
from urllib.parse import quote, urlsplit

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec, padding, rsa

PROVIDER = "gcp-kms"
EVIDENCE_VERSION = 1
SIGN_ALGORITHM = "EC_SIGN_P256_SHA256"
DECRYPT_ALGORITHM = "RSA_DECRYPT_OAEP_3072_SHA256"
TRANSCRIPT_FIELDS = ("domain", "mode", "nonce_b64", "org_id", "template_id", "device_id", "provider", "sign_key",
                     "sign_spki_sha256", "decrypt_key", "decrypt_spki_sha256", "workload_sub", "old_generation",
                     "old_sign_spki_sha256")
METADATA_IDENTITY = ("http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/identity"
                     "?audience={aud}&format=full")


class GcpKmsError(RuntimeError):
    """Any Cloud KMS configuration, integrity or policy failure (always fail closed)."""


class GcpKmsKeyUnavailable(GcpKmsError):
    """A specific key version can no longer be used: disabled, scheduled for destruction, destroyed, missing, or
    no longer usable by this identity. Only this condition may trigger the recovery fallback."""


def _unavailable_errors():
    try:
        from google.api_core import exceptions as gexc
        return (gexc.NotFound, gexc.Forbidden, gexc.FailedPrecondition)
    except ImportError:                              # client doubles in tests
        return ()


# --------------------------------------------------------------------------------------------- helpers
_CRC_TABLE = []
for _i in range(256):
    _c = _i
    for _ in range(8):
        _c = (_c >> 1) ^ 0x82F63B78 if _c & 1 else _c >> 1
    _CRC_TABLE.append(_c)


def crc32c(data: bytes) -> int:
    crc = 0xFFFFFFFF
    for byte in data:
        crc = _CRC_TABLE[(crc ^ byte) & 0xFF] ^ (crc >> 8)
    return crc ^ 0xFFFFFFFF


def _int(v) -> Optional[int]:
    """KMS wraps CRC32C in Int64Value; accept the wrapper or a plain int."""
    if v is None:
        return None
    return int(getattr(v, "value", v))


def spki_sha256(pem: str) -> str:
    key = serialization.load_pem_public_key(pem.encode())
    return hashlib.sha256(key.public_bytes(serialization.Encoding.DER,
                                           serialization.PublicFormat.SubjectPublicKeyInfo)).hexdigest()


def oaep() -> padding.OAEP:
    return padding.OAEP(mgf=padding.MGF1(algorithm=hashes.SHA256()), algorithm=hashes.SHA256(), label=None)


def transcript_bytes(fields: Dict[str, str]) -> bytes:
    """Canonical transcript: exactly TRANSCRIPT_FIELDS, printable-ASCII values without spaces, sorted keys, no
    whitespace. Must match the AOC verifier byte for byte."""
    if set(fields) != set(TRANSCRIPT_FIELDS):
        raise GcpKmsError("transcript fields do not match the v1 schema")
    for k, v in fields.items():
        if not isinstance(v, str) or not v or len(v) > 512 or any(not (0x21 <= ord(c) <= 0x7E) for c in v):
            raise GcpKmsError(f"transcript field {k} is not a valid value")
    return json.dumps(fields, sort_keys=True, separators=(",", ":")).encode("ascii")


def _protection_ok(resp, what: str) -> None:
    level = getattr(resp, "protection_level", None)
    name = getattr(level, "name", level)
    if name not in ("HSM", 3):
        raise GcpKmsError(f"{what}: key is not protected by Cloud HSM (protection level {name})")


# --------------------------------------------------------------------------------------------- provider
class GcpKmsProvider:
    def __init__(self, sign_key: str, decrypt_key: str, *, client: Any = None, endpoint: Optional[str] = None,
                 credentials_path: Optional[str] = None,
                 http_get: Optional[Callable[[str, Dict[str, str]], str]] = None):
        if not sign_key or not decrypt_key:
            raise GcpKmsError("both HSM_KMS_KEY (sign) and HSM_KMS_DECRYPT_KEY (decrypt) are required")
        self._client = client or self._make_client(endpoint, credentials_path)
        self._http_get = http_get or self._metadata_get
        self.sign_key = self._resolve(sign_key)
        self.decrypt_key = self._resolve(decrypt_key)
        if self.sign_key == self.decrypt_key:
            raise GcpKmsError("signing and decrypt keys must be distinct")
        self._pem: Dict[str, str] = {}

    @staticmethod
    def configured(env: Optional[Dict[str, str]] = None) -> bool:
        env = os.environ if env is None else env
        return bool((env.get("HSM_KMS_KEY") or "").strip() and (env.get("HSM_KMS_DECRYPT_KEY") or "").strip())

    @classmethod
    def from_env(cls, env: Optional[Dict[str, str]] = None, **kw) -> "GcpKmsProvider":
        env = os.environ if env is None else env
        return cls(env.get("HSM_KMS_KEY", "").strip(), env.get("HSM_KMS_DECRYPT_KEY", "").strip(),
                   endpoint=env.get("HSM_KMS_ENDPOINT") or None,
                   credentials_path=env.get("HSM_KMS_CREDENTIALS") or None, **kw)

    _shared: Dict[tuple, "GcpKmsProvider"] = {}

    @classmethod
    def shared(cls) -> "GcpKmsProvider":
        """One provider (and KMS client) per configuration for the process."""
        key = tuple(os.getenv(k, "") for k in ("HSM_KMS_KEY", "HSM_KMS_DECRYPT_KEY", "HSM_KMS_ENDPOINT",
                                                 "HSM_KMS_CREDENTIALS"))
        if key not in cls._shared:
            cls._shared[key] = cls.from_env()
        return cls._shared[key]

    @staticmethod
    def _make_client(endpoint, credentials_path):
        try:
            from google.cloud import kms_v1
        except ImportError as exc:
            raise GcpKmsError("google-cloud-kms is required: pip install 'ephapsys[hsm]'") from exc
        kwargs: Dict[str, Any] = {}
        if endpoint:
            kwargs["client_options"] = {"api_endpoint": endpoint}
        if credentials_path:
            from google.oauth2 import service_account
            path = os.path.expanduser(credentials_path)
            if not os.path.exists(path):
                raise GcpKmsError(f"HSM_KMS_CREDENTIALS not found: {path}")
            kwargs["credentials"] = service_account.Credentials.from_service_account_file(path)
        return kms_v1.KeyManagementServiceClient(**kwargs)

    def _resolve(self, name: str) -> str:
        """An exact version name is used as is. A CryptoKey name resolves to its newest ENABLED version (asymmetric
        keys have no primary version), so adding a version and restarting the workload rotates automatically."""
        if "/cryptoKeyVersions/" in name:
            return name
        versions = self._client.list_crypto_key_versions(request={"parent": name, "filter": "state=ENABLED"})
        names = [v.name for v in versions if getattr(getattr(v, "state", None), "name", "ENABLED") == "ENABLED"]
        if not names:
            raise GcpKmsError(f"KMS key {name} has no enabled version")
        return max(names, key=lambda n: int(n.rsplit("/", 1)[1]))

    # -- keys ---------------------------------------------------------------------------------------
    def key_name(self, role: str) -> str:
        return {"sign": self.sign_key, "decrypt": self.decrypt_key}[role]

    def public_key_pem(self, role: str, name: Optional[str] = None) -> str:
        name = name or self.key_name(role)
        if name in self._pem:
            return self._pem[name]
        resp = self._client.get_public_key(request={"name": name})
        if resp.name and resp.name != name:
            raise GcpKmsError("GetPublicKey answered for another key")
        if _int(resp.pem_crc32c) is not None and _int(resp.pem_crc32c) != crc32c(resp.pem.encode()):
            raise GcpKmsError("GetPublicKey response corrupted in transit (CRC32C)")
        _protection_ok(resp, "GetPublicKey")
        algorithm = getattr(resp.algorithm, "name", resp.algorithm)
        want = SIGN_ALGORITHM if role == "sign" else DECRYPT_ALGORITHM
        if algorithm != want:
            raise GcpKmsError(f"{role} key must be {want} (got {algorithm})")
        self._pem[name] = resp.pem
        return resp.pem

    def sign(self, message: bytes, name: Optional[str] = None) -> bytes:
        """ECDSA P-256 over SHA-256(message), computed by Cloud HSM. Verified locally before it is returned."""
        name = name or self.sign_key
        digest = hashlib.sha256(message).digest()
        resp = self._client.asymmetric_sign(request={"name": name, "digest": {"sha256": digest},
                                                     "digest_crc32c": crc32c(digest)})
        if not resp.verified_digest_crc32c or resp.name != name:
            raise GcpKmsError("AsymmetricSign request corrupted in transit")
        if _int(resp.signature_crc32c) != crc32c(resp.signature):
            raise GcpKmsError("AsymmetricSign response corrupted in transit (CRC32C)")
        _protection_ok(resp, "AsymmetricSign")
        pub = serialization.load_pem_public_key(self.public_key_pem("sign", name).encode())
        try:
            pub.verify(resp.signature, message, ec.ECDSA(hashes.SHA256()))
        except Exception as exc:
            raise GcpKmsError("Cloud KMS signature does not verify") from exc
        return resp.signature

    def decrypt(self, ciphertext: bytes) -> bytes:
        """RSA-OAEP-SHA256 decrypt by Cloud HSM with request/response integrity checks."""
        resp = self._client.asymmetric_decrypt(request={"name": self.decrypt_key, "ciphertext": ciphertext,
                                                        "ciphertext_crc32c": crc32c(ciphertext)})
        if not resp.verified_ciphertext_crc32c:
            raise GcpKmsError("AsymmetricDecrypt request corrupted in transit")
        if _int(resp.plaintext_crc32c) != crc32c(resp.plaintext):
            raise GcpKmsError("AsymmetricDecrypt response corrupted in transit (CRC32C)")
        _protection_ok(resp, "AsymmetricDecrypt")
        return resp.plaintext

    def attestation(self, role: str) -> Dict[str, str]:
        version = self._client.get_crypto_key_version(request={"name": self.key_name(role)})
        att = version.attestation
        if not att or not att.content:
            raise GcpKmsError(f"{role} key has no Cloud HSM attestation (protection level must be HSM)")
        chains = att.cert_chains
        return {"format": getattr(att.format, "name", str(att.format)),
                "content_b64": base64.b64encode(att.content).decode(),
                "cavium_certs_pem": "".join(chains.cavium_certs),
                "google_card_certs_pem": "".join(chains.google_card_certs),
                "google_partition_certs_pem": "".join(chains.google_partition_certs)}

    # -- identity -----------------------------------------------------------------------------------
    @staticmethod
    def audience(api_base: str) -> str:
        aud = (os.getenv("EPHAPSYS_WORKLOAD_AUDIENCE") or "").strip()
        if aud:
            return aud
        parts = urlsplit(api_base)
        return f"{parts.scheme}://{parts.netloc}"

    @staticmethod
    def _metadata_get(url: str, headers: Dict[str, str]) -> str:
        import requests
        resp = requests.get(url, headers=headers, timeout=10)
        if resp.status_code != 200:
            raise GcpKmsError(f"workload identity token unavailable from the metadata server ({resp.status_code})")
        return resp.text.strip()

    def workload_token(self, audience: str) -> str:
        path = (os.getenv("EPHAPSYS_WORKLOAD_TOKEN_FILE") or "").strip()
        if path:
            with open(path, "r", encoding="utf-8") as f:
                return f.read().strip()
        return self._http_get(METADATA_IDENTITY.format(aud=quote(audience, safe="")), {"Metadata-Flavor": "Google"})

    @staticmethod
    def token_sub(token: str) -> str:
        """The (unverified) subject of an ID token, used only to fill the transcript; the AOC verifies the token."""
        try:
            payload = token.split(".")[1]
            claims = json.loads(base64.urlsafe_b64decode(payload + "=" * (-len(payload) % 4)))
            return str(claims["sub"])
        except Exception as exc:
            raise GcpKmsError("workload identity token is malformed") from exc

    def require_usable(self, name: str) -> None:
        """Raise GcpKmsKeyUnavailable when a key version is not ENABLED or cannot be read by this identity."""
        try:
            version = self._client.get_crypto_key_version(request={"name": name})
        except _unavailable_errors() as exc:
            raise GcpKmsKeyUnavailable(f"{name}: {exc.__class__.__name__}") from exc
        state = getattr(getattr(version, "state", None), "name", getattr(version, "state", None))
        if state != "ENABLED":
            raise GcpKmsKeyUnavailable(f"{name} is {state}")

    def sign_with_version(self, message: bytes, name: str) -> bytes:
        """Sign with a specific (older) version; its unavailability is reported as GcpKmsKeyUnavailable."""
        self.require_usable(name)
        try:
            self.public_key_pem("sign", name)
            return self.sign(message, name=name)
        except _unavailable_errors() as exc:
            raise GcpKmsKeyUnavailable(f"{name}: {exc.__class__.__name__}") from exc

    # -- personalization ------------------------------------------------------------------------------
    def evidence(self, *, nonce_b64: str, org_id: str, template_id: str, device_id: str, token: str,
                 mode: str = "personalize", old_generation: str = "none", old_sign_key: Optional[str] = None,
                 old_sign_spki: str = "none", attest: bool = True) -> Dict[str, Any]:
        sign_pem, decrypt_pem = self.public_key_pem("sign"), self.public_key_pem("decrypt")
        fields = {"domain": f"ephapsys-hsm-{mode}-v1", "mode": mode, "nonce_b64": nonce_b64, "org_id": org_id,
                  "template_id": template_id, "device_id": device_id, "provider": PROVIDER,
                  "sign_key": self.sign_key, "sign_spki_sha256": spki_sha256(sign_pem),
                  "decrypt_key": self.decrypt_key, "decrypt_spki_sha256": spki_sha256(decrypt_pem),
                  "workload_sub": self.token_sub(token), "old_generation": old_generation,
                  "old_sign_spki_sha256": old_sign_spki}
        message = transcript_bytes(fields)
        ev: Dict[str, Any] = {"provider": PROVIDER, "version": EVIDENCE_VERSION, "device_id": device_id,
                              "workload_token": token, "transcript": fields, "sign_pub_pem": sign_pem,
                              "decrypt_pub_pem": decrypt_pem,
                              "sig_b64": base64.b64encode(self.sign(message)).decode()}
        if attest:
            ev["attestations"] = {"sign": self.attestation("sign"), "decrypt": self.attestation("decrypt")}
        if old_sign_key:
            ev["rotation_sig_b64"] = base64.b64encode(self.sign_with_version(message, old_sign_key)).decode()
        return ev

    def answer(self, challenge_ct_b64: str) -> str:
        return base64.b64encode(self.decrypt(base64.b64decode(challenge_ct_b64))).decode()

    # -- delivery --------------------------------------------------------------------------------------
    def unwrap_envelope(self, wrapped_b64: str, *, expect_model_ids=()) -> Dict[str, Any]:
        """Open an SIE envelope v2. Returns {"cek": bytes, "aad": bytes}."""
        try:
            env = json.loads(base64.urlsafe_b64decode(wrapped_b64.encode()))
        except Exception as exc:
            raise GcpKmsError("SIE envelope is malformed") from exc
        if (env.get("v") != 2 or env.get("alg") != "RSA-OAEP-3072-SHA256" or env.get("mgf1") != "SHA256"
                or env.get("label") != ""):
            raise GcpKmsError("unsupported SIE envelope")
        aad = env.get("aad") or {}
        if env.get("recipient_key") != self.decrypt_key or env.get("recipient_spki_sha256") != spki_sha256(
                self.public_key_pem("decrypt")) or aad.get("recipient_spki_sha256") != env.get("recipient_spki_sha256"):
            raise GcpKmsError("SIE envelope is addressed to another key")
        if aad.get("generation") != env.get("generation"):
            raise GcpKmsError("SIE envelope generation mismatch")
        ids = {str(x) for x in expect_model_ids if x}
        if ids and not ({str(aad.get("model_id")), str(aad.get("model_public_id"))} & ids):
            raise GcpKmsError("SIE envelope belongs to another model")
        cek = self.decrypt(base64.b64decode(env["wrapped_cek_b64"]))
        if len(cek) != 32:
            raise GcpKmsError("invalid content key length")
        return {"cek": cek, "aad": json.dumps(aad, sort_keys=True, separators=(",", ":")).encode()}

    def wrap_dek(self, dek: bytes) -> str:
        pub = serialization.load_pem_public_key(self.public_key_pem("decrypt").encode())
        if not isinstance(pub, rsa.RSAPublicKey):
            raise GcpKmsError("decrypt key must be RSA")
        return base64.b64encode(pub.encrypt(dek, oaep())).decode()
