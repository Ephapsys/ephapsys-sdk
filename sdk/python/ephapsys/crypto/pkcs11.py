# SPDX-License-Identifier: Apache-2.0
"""
Generic PKCS#11 token provider for the Ephapsys SDK.

One code path for any PKCS#11 token (TEE-backed tokens, HSMs, USB tokens, SoftHSM for CI).
Only configuration is device-specific: the module path, token and key identifiers, and the PIN.

Two key roles, separate token objects, both required to be sensitive and non-extractable:
  - sign: EC P-256 or RSA; signs personalization evidence with an internal-hash mechanism
          (CKM_ECDSA_SHA256 / CKM_SHA256_RSA_PKCS).
  - kem:  EC P-256; CKM_ECDH1_DERIVE for SIE CEK unwrap and for the at-rest cache DEK
          (ECIES in the existing hpke v1 format). Long-term private keys never leave the token;
          only the per-message derived secret reaches host memory.

Keys are selected by stable identifiers (token label/serial + CKA_ID and/or CKA_LABEL), never by
slot index. Every lookup requires exactly one match, and required mechanisms are checked before use:
anything missing or ambiguous fails closed.

Security scope: "token-backed key custody". Whether the token is hardware-backed is a property of
the deployed provider and its provisioning, not something this module can prove. No measured-boot
attestation is provided.

Requires the optional dependency `python-pkcs11` (pip install 'ephapsys[pkcs11]').
"""
from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec, rsa

BINDING_VERSION = 1
BINDING_DOMAIN = b"ephapsys-hsm-bind-v1\x00"
SIG_ALG_EC = "ECDSA_P256_SHA256"
SIG_ALG_RSA = "RSA_PKCS1V15_SHA256"
MIN_RSA_BITS = 2048


class Pkcs11Error(RuntimeError):
    """Any PKCS#11 configuration, lookup, policy or operation failure (always fail closed)."""


# ---------------------------------------------------------------------------
# Pure helpers (shared contract with the AOC verifier)
# ---------------------------------------------------------------------------
def spki_der(pub_pem: str) -> bytes:
    """Canonical SubjectPublicKeyInfo DER of a PEM public key."""
    key = serialization.load_pem_public_key(pub_pem.encode())
    return key.public_bytes(serialization.Encoding.DER, serialization.PublicFormat.SubjectPublicKeyInfo)


def spki_sha256_hex(pub_pem: str) -> str:
    return hashlib.sha256(spki_der(pub_pem)).hexdigest()


def binding_message(nonce: bytes, kem_pub_pem: str) -> bytes:
    """Binding v1: domain || nonce || SHA256(SPKI DER of the KEM key). The token hashes it with SHA-256."""
    if not nonce:
        raise Pkcs11Error("empty nonce")
    return BINDING_DOMAIN + nonce + hashlib.sha256(spki_der(kem_pub_pem)).digest()


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
def _hex_bytes(value: Optional[str], name: str) -> Optional[bytes]:
    if value is None or value == "":
        return None
    v = value.strip().lower()
    if v.startswith("0x"):
        v = v[2:]
    try:
        return bytes.fromhex(v)
    except ValueError as exc:
        raise Pkcs11Error(f"{name} must be hex (got {value!r})") from exc


@dataclass(frozen=True)
class Pkcs11Config:
    module: str
    token_label: Optional[str] = None
    token_serial: Optional[str] = None
    pin: Optional[str] = None
    pin_file: Optional[str] = None
    sign_key_id: Optional[bytes] = None
    sign_key_label: Optional[str] = None
    kem_key_id: Optional[bytes] = None
    kem_key_label: Optional[str] = None

    @staticmethod
    def configured(env: Optional[Dict[str, str]] = None) -> bool:
        env = os.environ if env is None else env
        return bool((env.get("PKCS11_MODULE") or "").strip())

    @classmethod
    def from_env(cls, env: Optional[Dict[str, str]] = None) -> "Pkcs11Config":
        env = os.environ if env is None else env
        module = (env.get("PKCS11_MODULE") or "").strip()
        if not module:
            raise Pkcs11Error("PKCS11_MODULE is not set")
        g = lambda k: (env.get(k) or "").strip() or None
        cfg = cls(
            module=module,
            token_label=g("PKCS11_TOKEN_LABEL"),
            token_serial=g("PKCS11_TOKEN_SERIAL"),
            pin=env.get("PKCS11_PIN") or None,
            pin_file=g("PKCS11_PIN_FILE"),
            sign_key_id=_hex_bytes(g("PKCS11_SIGN_KEY_ID"), "PKCS11_SIGN_KEY_ID"),
            sign_key_label=g("PKCS11_SIGN_KEY_LABEL"),
            kem_key_id=_hex_bytes(g("PKCS11_KEM_KEY_ID"), "PKCS11_KEM_KEY_ID"),
            kem_key_label=g("PKCS11_KEM_KEY_LABEL"),
        )
        cfg.validate()
        return cfg

    def validate(self) -> None:
        if not (self.token_label or self.token_serial):
            raise Pkcs11Error("set PKCS11_TOKEN_LABEL or PKCS11_TOKEN_SERIAL (slot index alone is not accepted)")
        if self.pin and self.pin_file:
            raise Pkcs11Error("set only one of PKCS11_PIN and PKCS11_PIN_FILE")
        if not (self.pin or self.pin_file):
            raise Pkcs11Error("set PKCS11_PIN_FILE (preferred) or PKCS11_PIN")

    def resolve_pin(self) -> str:
        if self.pin_file:
            try:
                with open(os.path.expanduser(self.pin_file), "r", encoding="utf-8") as f:
                    pin = f.read().strip()
            except OSError as exc:
                raise Pkcs11Error(f"cannot read PKCS11_PIN_FILE: {exc}") from exc
            if not pin:
                raise Pkcs11Error("PKCS11_PIN_FILE is empty")
            return pin
        return self.pin or ""

    def key_selector(self, role: str) -> Tuple[Optional[bytes], Optional[str]]:
        if role == "sign":
            sel = (self.sign_key_id, self.sign_key_label)
        elif role == "kem":
            sel = (self.kem_key_id, self.kem_key_label)
        else:
            raise Pkcs11Error(f"unknown key role {role!r}")
        if sel == (None, None):
            env = "PKCS11_SIGN_KEY_ID/LABEL" if role == "sign" else "PKCS11_KEM_KEY_ID/LABEL"
            raise Pkcs11Error(f"no identifier configured for the {role} key (set {env})")
        return sel


# ---------------------------------------------------------------------------
# Provider
# ---------------------------------------------------------------------------
def _import_pkcs11():
    try:
        import pkcs11  # type: ignore
        return pkcs11
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise Pkcs11Error("python-pkcs11 is required for PKCS#11 anchors: pip install 'ephapsys[pkcs11]'") from exc


def _norm_serial(v: Any) -> str:
    if isinstance(v, (bytes, bytearray)):
        v = bytes(v).decode("ascii", "ignore")
    return str(v or "").strip()


class Pkcs11Provider:
    """Thin, fail-closed wrapper over one PKCS#11 token. Each operation opens its own session."""

    def __init__(self, config: Pkcs11Config, *, lib_factory: Optional[Callable[[str], Any]] = None):
        self.config = config
        self._p = _import_pkcs11()
        factory = lib_factory or self._p.lib
        try:
            self._lib = factory(config.module)
        except Exception as exc:
            raise Pkcs11Error(f"cannot load PKCS#11 module {config.module!r}: {exc}") from exc
        self._token = self._find_token()

    @classmethod
    def from_env(cls, env: Optional[Dict[str, str]] = None) -> "Pkcs11Provider":
        return cls(Pkcs11Config.from_env(env))

    # ---- token / session -------------------------------------------------
    def _find_token(self):
        want_label, want_serial = self.config.token_label, self.config.token_serial
        matches = []
        for slot in self._lib.get_slots(token_present=True):
            tok = slot.get_token()
            if want_label and (tok.label or "").strip() != want_label:
                continue
            if want_serial and _norm_serial(tok.serial) != want_serial:
                continue
            matches.append(tok)
        if len(matches) != 1:
            raise Pkcs11Error(f"expected exactly one token matching label={want_label!r} serial={want_serial!r}, found {len(matches)}")
        return matches[0]

    def token_info(self) -> Dict[str, str]:
        t = self._token
        return {"label": (t.label or "").strip(), "serial": _norm_serial(t.serial),
                "manufacturer": (t.manufacturer_id or "").strip(), "model": (t.model or "").strip()}

    def _session(self):
        return self._token.open(user_pin=self.config.resolve_pin())

    def mechanisms(self):
        return set(self._token.slot.get_mechanisms())

    def _require_mechanism(self, mech) -> None:
        if mech not in self.mechanisms():
            raise Pkcs11Error(f"token does not support required mechanism {mech!r}")

    # ---- key lookup ------------------------------------------------------
    def _find_one(self, session, object_class, role: str):
        P = self._p; A = P.Attribute
        key_id, label = self.config.key_selector(role)
        attrs = {A.CLASS: object_class}
        if key_id is not None:
            attrs[A.ID] = key_id
        if label is not None:
            attrs[A.LABEL] = label
        found = list(session.get_objects(attrs))
        if len(found) != 1:
            kind = "private" if object_class == P.ObjectClass.PRIVATE_KEY else "public"
            raise Pkcs11Error(f"expected exactly one {kind} {role} key (id={key_id.hex() if key_id else None}, label={label!r}), found {len(found)}")
        return found[0]

    def _check_private_policy(self, priv, role: str) -> None:
        A = self._p.Attribute
        try:
            sensitive, extractable = priv[A.SENSITIVE], priv[A.EXTRACTABLE]
        except Exception as exc:
            raise Pkcs11Error(f"cannot read SENSITIVE/EXTRACTABLE on the {role} key: {exc}") from exc
        if not sensitive or extractable:
            raise Pkcs11Error(f"{role} private key must be sensitive and non-extractable (sensitive={sensitive}, extractable={extractable})")
        usage = A.SIGN if role == "sign" else A.DERIVE
        try:
            allowed = priv[usage]
        except Exception:
            allowed = False
        if not allowed:
            raise Pkcs11Error(f"{role} private key lacks the {'CKA_SIGN' if role == 'sign' else 'CKA_DERIVE'} permission")

    def _public_key(self, session, role: str):
        """Return a `cryptography` public key for the role's public object."""
        P = self._p
        pub = self._find_one(session, P.ObjectClass.PUBLIC_KEY, role)
        kt = pub[P.Attribute.KEY_TYPE]
        if kt == P.KeyType.EC:
            from pkcs11.util.ec import encode_ec_public_key  # type: ignore
            key = serialization.load_der_public_key(encode_ec_public_key(pub))
        elif kt == P.KeyType.RSA:
            from pkcs11.util.rsa import encode_rsa_public_key  # type: ignore
            key = serialization.load_der_public_key(encode_rsa_public_key(pub))
        else:
            raise Pkcs11Error(f"unsupported {role} key type {kt!r}")
        return key

    def _checked_keys(self, session, role: str):
        P = self._p
        priv = self._find_one(session, P.ObjectClass.PRIVATE_KEY, role)
        self._check_private_policy(priv, role)
        pub = self._public_key(session, role)
        if role == "kem" and not (isinstance(pub, ec.EllipticCurvePublicKey) and isinstance(pub.curve, ec.SECP256R1)):
            raise Pkcs11Error("kem key must be EC P-256 (hpke v1)")
        if role == "sign":
            if isinstance(pub, ec.EllipticCurvePublicKey) and not isinstance(pub.curve, ec.SECP256R1):
                raise Pkcs11Error("EC sign key must be P-256")
            if isinstance(pub, rsa.RSAPublicKey) and pub.key_size < MIN_RSA_BITS:
                raise Pkcs11Error(f"RSA sign key must be >= {MIN_RSA_BITS} bits")
        return priv, pub

    def public_key_pem(self, role: str) -> str:
        with self._session() as s:
            _, pub = self._checked_keys(s, role)
        return pub.public_bytes(serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo).decode()

    def key_id_hex(self, role: str) -> Optional[str]:
        key_id, _ = self.config.key_selector(role)
        return key_id.hex() if key_id is not None else None

    # ---- operations ------------------------------------------------------
    def sign(self, message: bytes) -> Tuple[str, bytes]:
        """Sign with an internal-hash mechanism. Returns (sig_alg, signature) with ECDSA as DER."""
        P = self._p
        with self._session() as s:
            priv, pub = self._checked_keys(s, "sign")
            if isinstance(pub, ec.EllipticCurvePublicKey):
                self._require_mechanism(P.Mechanism.ECDSA_SHA256)
                from pkcs11.util.ec import encode_ecdsa_signature  # type: ignore
                raw = priv.sign(message, mechanism=P.Mechanism.ECDSA_SHA256)
                return SIG_ALG_EC, encode_ecdsa_signature(raw)
            self._require_mechanism(P.Mechanism.SHA256_RSA_PKCS)
            return SIG_ALG_RSA, priv.sign(message, mechanism=P.Mechanism.SHA256_RSA_PKCS)

    def ecdh(self, peer_spki_der: bytes) -> bytes:
        """ECDH (CKM_ECDH1_DERIVE, null KDF) with the kem key; returns the raw shared secret x-coordinate."""
        P = self._p; A = P.Attribute
        peer = serialization.load_der_public_key(peer_spki_der)
        if not (isinstance(peer, ec.EllipticCurvePublicKey) and isinstance(peer.curve, ec.SECP256R1)):
            raise Pkcs11Error("peer key must be EC P-256")
        point = peer.public_bytes(serialization.Encoding.X962, serialization.PublicFormat.UncompressedPoint)
        with self._session() as s:
            priv, _ = self._checked_keys(s, "kem")
            self._require_mechanism(P.Mechanism.ECDH1_DERIVE)
            try:
                secret = priv.derive_key(P.KeyType.GENERIC_SECRET, 256, mechanism_param=(P.KDF.NULL, None, point),
                                         template={A.SENSITIVE: False, A.EXTRACTABLE: True, A.TOKEN: False})
                value = bytes(secret[A.VALUE])
            except Exception as exc:
                raise Pkcs11Error(f"ECDH derive failed: {exc}") from exc
        if len(value) != 32:
            raise Pkcs11Error(f"unexpected ECDH secret length {len(value)}")
        return value

    def build_csr(self, common_name: str) -> str:
        """PKCS#10 CSR for the token SIGN key, signed on the token (no software key involved)."""
        try:
            from asn1crypto import csr as acsr, keys as akeys, pem as apem, x509 as ax509  # type: ignore
        except ImportError as exc:  # pragma: no cover - asn1crypto ships with python-pkcs11
            raise Pkcs11Error("asn1crypto is required to build a PKCS#11 CSR") from exc
        pub_pem = self.public_key_pem("sign")
        info = acsr.CertificationRequestInfo({
            "version": "v1",
            "subject": ax509.Name.build({"common_name": common_name}),
            "subject_pk_info": akeys.PublicKeyInfo.load(spki_der(pub_pem)),
            "attributes": [],
        })
        sig_alg, sig = self.sign(info.dump())
        algo = {"algorithm": "sha256_ecdsa"} if sig_alg == SIG_ALG_EC else {"algorithm": "sha256_rsa", "parameters": None}
        req = acsr.CertificationRequest({"certification_request_info": info, "signature_algorithm": algo, "signature": sig})
        return apem.armor("CERTIFICATE REQUEST", req.dump()).decode()

    # ---- evidence --------------------------------------------------------
    def personalization_evidence(self, nonce_b64: str) -> Dict[str, Any]:
        """Binding-v1 evidence: the sign key signs domain || nonce || SHA256(SPKI(kem_pub))."""
        import base64
        try:
            nonce = base64.b64decode(nonce_b64, validate=True)
        except Exception as exc:
            raise Pkcs11Error(f"invalid challenge nonce: {exc}") from exc
        kem_pem = self.public_key_pem("kem")
        sign_pem = self.public_key_pem("sign")
        sig_alg, sig = self.sign(binding_message(nonce, kem_pem))
        return {
            "provider": "pkcs11",
            "binding_version": BINDING_VERSION,
            "nonce_b64": nonce_b64,
            "sig_alg": sig_alg,
            "sig_b64": base64.b64encode(sig).decode(),
            "pubkey_pem": sign_pem,
            "pubkey_spki_sha256": spki_sha256_hex(sign_pem),
            "kem_pub_pem": kem_pem,
            "token": self.token_info(),
            "sign_key_id_hex": self.key_id_hex("sign"),
            "kem_key_id_hex": self.key_id_hex("kem"),
        }
