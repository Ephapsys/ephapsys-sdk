# SPDX-License-Identifier: Apache-2.0
"""
Idempotent, non-destructive provisioning of a PKCS#11 token for the Ephapsys `hsm` anchor.

Brings a device from "token never set up" to "token, PIN file and both keys present", and is safe to run
again on a device that is already provisioned (it then changes nothing):

  1. Token. Use the token whose label is PKCS11_TOKEN_LABEL. If none exists, initialise the first token
     the module reports as NOT initialised (CKF_TOKEN_INITIALIZED clear). A token that is already
     initialised is never re-initialised, so other tokens on the module and their keys cannot be wiped.
  2. PINs. A random user PIN is written to PKCS11_PIN_FILE, created exclusively (an existing file is never
     overwritten). A random SO PIN is generated and discarded once the user PIN is set. Until then it is kept
     in a root-only "<pin file>.so-pending" file, so an interrupted run can finish instead of leaving a token
     nobody can administer.
  3. Keys. Create the sign key (EC P-256, CKA_SIGN only) and the KEM key (EC P-256, CKA_DERIVE only),
     sensitive and non-extractable, selected by PKCS11_SIGN_KEY_LABEL / PKCS11_KEM_KEY_LABEL, only if they
     are absent. Existing keys are never replaced; they are checked with the same policy as at runtime.

Anything inconsistent (a labelled token without a PIN file, a locked or nearly locked PIN, half a key pair,
missing mechanisms) is refused with an error rather than repaired. A wrong PIN is tried once per run, and
not at all when the token reports the PIN locked or on its final try.
"""
from __future__ import annotations

import ctypes
import os
import secrets
import tempfile
from typing import Any, Callable, Dict, List, Optional

from .pkcs11 import Pkcs11Config, Pkcs11Error, Pkcs11Provider, _import_pkcs11, spki_sha256_hex

SO_PENDING_SUFFIX = ".so-pending"


# ---------------------------------------------------------------------------
# Files
# ---------------------------------------------------------------------------
def _fsync_dir(path: str) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _create_exclusive(path: str, content: str, mode: int, gid: Optional[int] = None) -> None:
    """Write `content` to `path` atomically; raise FileExistsError if `path` already exists."""
    d = os.path.dirname(os.path.abspath(path))
    os.makedirs(d, mode=0o755, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=d, prefix=".ephapsys-")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(content)
            f.flush()
            os.fsync(f.fileno())
            if gid is not None:
                os.fchown(f.fileno(), -1, gid)
            os.fchmod(f.fileno(), mode)
        os.link(tmp, path)                               # fails if path exists: never overwrites
    finally:
        os.unlink(tmp)
    _fsync_dir(d)


def _read_secret(path: str, what: str) -> str:
    try:
        with open(path, "r", encoding="utf-8") as f:
            value = f.read().strip()
    except OSError as exc:
        raise Pkcs11Error(f"cannot read {what} {path}: {exc}") from exc
    if not value:
        raise Pkcs11Error(f"{what} {path} is empty")
    return value


def _discard(path: str) -> None:
    """Overwrite then remove a small secret file (best effort against casual recovery)."""
    try:
        size = os.path.getsize(path)
        with open(path, "r+b") as f:
            f.write(b"\0" * size)
            f.flush()
            os.fsync(f.fileno())
    except OSError:
        pass
    os.unlink(path)
    _fsync_dir(os.path.dirname(os.path.abspath(path)))


# ---------------------------------------------------------------------------
# C_InitToken (python-pkcs11 does not expose it)
# ---------------------------------------------------------------------------
class _CkVersion(ctypes.Structure):
    _fields_ = [("major", ctypes.c_ubyte), ("minor", ctypes.c_ubyte)]


class _CkFunctionListHead(ctypes.Structure):
    # The first ten entries of CK_FUNCTION_LIST, in the order fixed by the PKCS#11 standard.
    _fields_ = [("version", _CkVersion)] + [(n, ctypes.c_void_p) for n in (
        "C_Initialize", "C_Finalize", "C_GetInfo", "C_GetFunctionList", "C_GetSlotList", "C_GetSlotInfo",
        "C_GetTokenInfo", "C_GetMechanismList", "C_GetMechanismInfo", "C_InitToken")]


def _c_init_token(module: str, slot_id: int, so_pin: str, label: str) -> None:
    """C_InitToken on an already C_Initialize'd module (python-pkcs11 loaded it in this process)."""
    lab = label.encode("utf-8")
    if len(lab) > 32:
        raise Pkcs11Error(f"token label {label!r} is longer than 32 bytes")
    try:
        dll = ctypes.CDLL(module)
    except OSError as exc:
        raise Pkcs11Error(f"cannot load PKCS#11 module {module!r}: {exc}") from exc
    fl = ctypes.POINTER(_CkFunctionListHead)()
    rv = dll.C_GetFunctionList(ctypes.byref(fl))
    if rv != 0 or not fl or not fl.contents.C_InitToken:
        raise Pkcs11Error(f"C_GetFunctionList failed (rv=0x{rv:x})")
    proto = ctypes.CFUNCTYPE(ctypes.c_ulong, ctypes.c_ulong, ctypes.c_char_p, ctypes.c_ulong, ctypes.c_char_p)
    init_token = proto(fl.contents.C_InitToken)
    pin = so_pin.encode("utf-8")
    rv = init_token(slot_id, pin, len(pin), lab.ljust(32, b" "))
    if rv != 0:
        raise Pkcs11Error(f"C_InitToken failed on slot {slot_id} (rv=0x{rv:x})")


# ---------------------------------------------------------------------------
# Provisioning
# ---------------------------------------------------------------------------
def _validate(cfg: Pkcs11Config) -> None:
    cfg.validate()
    if not cfg.token_label:
        raise Pkcs11Error("provisioning selects and labels the token by PKCS11_TOKEN_LABEL")
    if not cfg.pin_file or cfg.pin:
        raise Pkcs11Error("provisioning writes the user PIN to PKCS11_PIN_FILE; set it and unset PKCS11_PIN")
    if not (cfg.sign_key_label and cfg.kem_key_label):
        raise Pkcs11Error("provisioning creates keys by label; set PKCS11_SIGN_KEY_LABEL and PKCS11_KEM_KEY_LABEL")


def _tokens(lib) -> List[Any]:
    return [slot.get_token() for slot in lib.get_slots(token_present=True)]


def _labelled(lib, label: str):
    found = [t for t in _tokens(lib) if (t.label or "").strip() == label]
    if len(found) > 1:
        raise Pkcs11Error(f"{len(found)} tokens are labelled {label!r}; refusing to choose")
    return found[0] if found else None


def _require_mechanisms(P, slot) -> None:
    mechs = set(slot.get_mechanisms())
    missing = [m.name for m in (P.Mechanism.EC_KEY_PAIR_GEN, P.Mechanism.ECDH1_DERIVE) if m not in mechs]
    if not ({P.Mechanism.ECDSA_SHA256, P.Mechanism.ECDSA} & mechs):
        missing.append("ECDSA_SHA256 or ECDSA")
    if missing:
        raise Pkcs11Error(f"token lacks required mechanisms: {', '.join(missing)}")


def _ensure_keys(P, session, cfg: Pkcs11Config, actions: List[str]) -> None:
    from pkcs11.util.ec import encode_named_curve_parameters  # type: ignore
    A, OC = P.Attribute, P.ObjectClass
    base = {A.SENSITIVE: True, A.EXTRACTABLE: False, A.TOKEN: True, A.PRIVATE: True}
    roles = (("sign", cfg.sign_key_label, cfg.sign_key_id, {A.SIGN: True, A.DERIVE: False}),
             ("kem", cfg.kem_key_label, cfg.kem_key_id, {A.DERIVE: True, A.SIGN: False}))
    params = None
    for role, label, key_id, flags in roles:
        privs = list(session.get_objects({A.CLASS: OC.PRIVATE_KEY, A.LABEL: label}))
        pubs = list(session.get_objects({A.CLASS: OC.PUBLIC_KEY, A.LABEL: label}))
        if (len(privs), len(pubs)) == (1, 1):
            continue                                     # existing pair: checked by the runtime policy below
        if privs or pubs:
            raise Pkcs11Error(f"{role} key {label!r} is incomplete or ambiguous ({len(privs)} private, "
                              f"{len(pubs)} public); refusing to create or replace it")
        if params is None:
            params = session.create_domain_parameters(
                P.KeyType.EC, {A.EC_PARAMS: encode_named_curve_parameters("secp256r1")}, local=True)
        extra = {"id": key_id} if key_id is not None else {}
        params.generate_keypair(store=True, label=label, public_template={A.TOKEN: True},
                                private_template={**base, **flags}, **extra)
        actions.append(f"created {role} key {label!r}")


def provision(cfg: Pkcs11Config, *, pin_group: Optional[str] = None,
              lib_factory: Optional[Callable[[str], Any]] = None) -> Dict[str, Any]:
    """Provision the token described by `cfg` (see the module docstring). Returns a JSON-serialisable report."""
    _validate(cfg)
    P = _import_pkcs11()
    if pin_group:
        import grp                                       # POSIX only; the PIN file group needs it
        gid: Optional[int] = grp.getgrnam(pin_group).gr_gid
    else:
        gid = None
    pin_mode = 0o440 if pin_group else 0o400
    try:
        lib = (lib_factory or P.lib)(cfg.module)
    except Exception as exc:
        raise Pkcs11Error(f"cannot load PKCS#11 module {cfg.module!r}: {exc}") from exc
    label, pin_file = cfg.token_label, os.path.expanduser(cfg.pin_file)
    so_pending = pin_file + SO_PENDING_SUFFIX
    actions: List[str] = []
    TF = P.TokenFlag

    tok = _labelled(lib, label)
    if tok is None:
        uninit = [t for t in _tokens(lib) if not (t.flags & TF.TOKEN_INITIALIZED)]
        if not uninit:
            raise Pkcs11Error(f"no token labelled {label!r} and no uninitialised token on {cfg.module}; "
                              "refusing to initialise an existing token")
        target = uninit[0]
        _require_mechanisms(P, target.slot)
        if os.path.exists(pin_file):
            actions.append("reused existing PIN file")
        else:
            _create_exclusive(pin_file, secrets.token_hex(16), pin_mode, gid)
            actions.append("created PIN file")
        if os.path.exists(so_pending):                   # left by a run that stopped before C_InitToken
            os.unlink(so_pending)
        so_pin = secrets.token_hex(16)
        _create_exclusive(so_pending, so_pin, 0o400)
        _c_init_token(cfg.module, target.slot.slot_id, so_pin, label)
        actions.append(f"initialised token {label!r} in slot {target.slot.slot_id}")
        tok = _labelled(lib, label)
        if tok is None:
            raise Pkcs11Error(f"token {label!r} not found after initialisation")

    if not (tok.flags & TF.USER_PIN_INITIALIZED):
        if not os.path.exists(so_pending):
            raise Pkcs11Error(f"token {label!r} has no user PIN and its SO PIN is not available; "
                              "this needs a manual repair")
        so_pin = _read_secret(so_pending, "SO PIN file")
        user_pin = _read_secret(pin_file, "PIN file")
        with tok.open(rw=True, so_pin=so_pin) as s:
            s.init_pin(user_pin)
        actions.append("set the user PIN")
        tok = _labelled(lib, label)
    if os.path.exists(so_pending):
        _discard(so_pending)
        actions.append("discarded the SO PIN")

    if not os.path.exists(pin_file):
        raise Pkcs11Error(f"token {label!r} exists but {pin_file} does not; this needs a manual repair "
                          "(a new PIN file would not match the token)")
    if tok.flags & (TF.USER_PIN_LOCKED | TF.USER_PIN_FINAL_TRY):
        raise Pkcs11Error(f"user PIN of token {label!r} is locked or one failure from locking; "
                          "not trying it")
    _require_mechanisms(P, tok.slot)
    try:
        with tok.open(rw=True, user_pin=cfg.resolve_pin()) as s:
            _ensure_keys(P, s, cfg, actions)
    except P.exceptions.PinIncorrect as exc:
        raise Pkcs11Error(f"PIN in {pin_file} is incorrect for token {label!r}; stopping "
                          "(repeated attempts lock the token)") from exc

    prov = Pkcs11Provider(cfg, lib_factory=lambda _m: lib)  # runtime policy: sensitive, role-separated, P-256
    report: Dict[str, Any] = {
        "status": "provisioned" if actions else "already_provisioned",
        "actions": actions,
        "token": prov.token_info(),
        "slot_id": tok.slot.slot_id,
        "pin_file": pin_file,
    }
    for role in ("sign", "kem"):
        report[role] = {"label": cfg.key_selector(role)[1], "spki_sha256": spki_sha256_hex(prov.public_key_pem(role))}
    return report
