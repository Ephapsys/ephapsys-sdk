# SPDX-License-Identifier: Apache-2.0
"""
Idempotent, non-destructive provisioning of a PKCS#11 token for the Ephapsys `hsm` anchor.

Brings a device from "token never set up" to "token, PIN file and both keys present", and is safe to run
again on a device that is already provisioned (it then changes nothing):

  1. Token. Use the token whose label is PKCS11_TOKEN_LABEL. If none exists, initialise a token the module
     reports as NOT initialised (CKF_TOKEN_INITIALIZED clear): the only one, or the one chosen by `init_slot`
     when several are eligible. An initialised token is never re-initialised, so other tokens on the module
     and their keys cannot be wiped.
  2. PINs. A random user PIN is written to PKCS11_PIN_FILE, created exclusively (an existing file is
     validated and reused, never overwritten). A random SO PIN is generated and discarded once the user PIN
     is set. Until then it is kept in a recovery file "<pin file>.so-pending" (owner-only, bound to module
     and label), so an interrupted run can finish instead of leaving a token nobody can administer. The
     overwrite on discard is best effort; it cannot guarantee erasure on flash or copy-on-write storage.
  3. Keys. Create the sign key (EC P-256, CKA_SIGN only) and the KEM key (EC P-256, CKA_DERIVE only),
     sensitive and non-extractable, selected by PKCS11_SIGN_KEY_LABEL / PKCS11_KEM_KEY_LABEL, only if they
     are absent. Existing keys are never replaced; they are checked with the same policy as at runtime.

Runs are serialised by an exclusive lock next to the PIN file, and every input that can be checked without
touching the token (configuration, group, directory, existing secret files, PIN format, slot choice) is
checked before the first write. Secret files must be regular files (no symlinks) owned by the invoking user,
in a directory owned by that user or root and not writable by group or others. Anything inconsistent (a
labelled token without a PIN file, a locked PIN, half a key pair, missing mechanisms) is refused with an
error rather than repaired. Provisioning stops at the first incorrect PIN, and does not try a PIN the token
reports as locked or on its final try.
"""
from __future__ import annotations

import ctypes
import json
import os
import secrets
import stat
import tempfile
from typing import Any, Callable, Dict, List, Optional, Union

from .pkcs11 import Pkcs11Config, Pkcs11Error, Pkcs11Provider, _import_pkcs11, spki_sha256_hex

SO_PENDING_SUFFIX = ".so-pending"
LOCK_SUFFIX = ".lock"
TMP_PREFIX = ".ephapsys-provision-"
MIN_PIN_LEN, MAX_PIN_LEN = 4, 64                         # within the bounds of common tokens (OP-TEE, SoftHSM)


# ---------------------------------------------------------------------------
# Files
# ---------------------------------------------------------------------------
def _fsync_dir(path: str) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _check_dir(d: str) -> None:
    st = os.lstat(d)
    if stat.S_ISLNK(st.st_mode) or not stat.S_ISDIR(st.st_mode):
        raise Pkcs11Error(f"{d} must be a directory, not a symlink")
    if st.st_uid not in (0, os.geteuid()):
        raise Pkcs11Error(f"{d} must be owned by root or the invoking user")
    if st.st_mode & 0o022:
        raise Pkcs11Error(f"{d} must not be writable by group or others")


def _open_checked(path: str, what: str, flags: int, *, gid: Optional[int] = None) -> int:
    """Open a secret file without following symlinks and check type, owner and mode on the descriptor.
    With `gid`, the file must belong to that group and be group-readable; otherwise it must be owner-only."""
    try:
        fd = os.open(path, flags | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0))
    except OSError as exc:
        if os.path.islink(path):
            raise Pkcs11Error(f"{what} {path} is a symlink; symlinks are not accepted") from exc
        raise Pkcs11Error(f"cannot open {what} {path}: {exc}") from exc
    try:
        st = os.fstat(fd)
        mode = stat.S_IMODE(st.st_mode)
        if not stat.S_ISREG(st.st_mode):
            raise Pkcs11Error(f"{what} {path} must be a regular file")
        if st.st_uid != os.geteuid():
            raise Pkcs11Error(f"{what} {path} must be owned by the invoking user (uid {os.geteuid()})")
        if gid is None:
            if mode & 0o077:
                raise Pkcs11Error(f"{what} {path} must not be accessible by group or others (mode {mode:o})")
        elif st.st_gid != gid or not mode & 0o040 or mode & 0o027:
            raise Pkcs11Error(f"{what} {path} must be group {gid}, group-readable only (mode {mode:o}, gid {st.st_gid})")
    except BaseException:
        os.close(fd)
        raise
    return fd


def _read_checked(path: str, what: str, *, gid: Optional[int] = None) -> str:
    fd = _open_checked(path, what, os.O_RDONLY, gid=gid)
    with os.fdopen(fd, "r", encoding="utf-8") as f:
        return f.read().strip()


def _check_pin(pin: str, path: str) -> str:
    if not (MIN_PIN_LEN <= len(pin) <= MAX_PIN_LEN) or not (pin.isascii() and pin.isprintable()) or " " in pin:
        raise Pkcs11Error(f"PIN in {path} must be {MIN_PIN_LEN}-{MAX_PIN_LEN} printable ASCII characters without spaces")
    return pin


def _create_exclusive(path: str, content: str, mode: int, gid: Optional[int] = None) -> None:
    """Write `content` to `path` atomically; raise FileExistsError if `path` already exists."""
    d = os.path.dirname(os.path.abspath(path))
    fd, tmp = tempfile.mkstemp(dir=d, prefix=TMP_PREFIX)
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


def _discard(path: str) -> None:
    """Overwrite then remove the recovery file (best effort against casual recovery)."""
    ro = _open_checked(path, "SO PIN recovery file", os.O_RDONLY)  # the file is 0400: validate, then make writable
    try:
        st = os.fstat(ro)
        os.fchmod(ro, 0o600)
        fd = os.open(path, os.O_WRONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0))
        try:
            if (os.fstat(fd).st_dev, os.fstat(fd).st_ino) != (st.st_dev, st.st_ino):
                raise Pkcs11Error(f"SO PIN recovery file {path} changed while being discarded")
            os.write(fd, b"\0" * st.st_size)
            os.fsync(fd)
        finally:
            os.close(fd)
    finally:
        os.close(ro)
    os.unlink(path)                                      # the directory is trusted and the run holds the lock
    _fsync_dir(os.path.dirname(os.path.abspath(path)))


def _cleanup_orphans(d: str) -> None:
    """Remove temporary files left by an interrupted run (only regular files this user owns)."""
    for name in os.listdir(d):
        if name.startswith(TMP_PREFIX):
            p = os.path.join(d, name)
            st = os.lstat(p)
            if stat.S_ISREG(st.st_mode) and st.st_uid == os.geteuid():
                os.unlink(p)


class _Lock:
    """Exclusive, non-blocking lock serialising provisioning runs that share a PIN file."""

    def __init__(self, path: str):
        self.path, self.fd = path, -1

    def __enter__(self):
        import fcntl                                     # POSIX only, like the PIN file ownership model
        self.fd = os.open(self.path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
        try:
            fcntl.flock(self.fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            os.close(self.fd)
            raise Pkcs11Error(f"another provisioning run holds {self.path}") from exc
        return self

    def __exit__(self, *exc):
        os.close(self.fd)                                # releases the lock; the file is kept on purpose


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
    """C_InitToken on an already C_Initialize'd module (python-pkcs11 loaded it in this process; never finalised here)."""
    lab = label.encode("utf-8")
    if len(lab) > 32:
        raise Pkcs11Error(f"token label {label!r} is longer than 32 bytes")
    try:
        get_list = ctypes.CDLL(module).C_GetFunctionList
    except (OSError, AttributeError) as exc:
        raise Pkcs11Error(f"PKCS#11 module {module!r} does not provide C_GetFunctionList: {exc}") from exc
    get_list.argtypes = [ctypes.POINTER(ctypes.POINTER(_CkFunctionListHead))]
    get_list.restype = ctypes.c_ulong
    fl = ctypes.POINTER(_CkFunctionListHead)()
    rv = get_list(ctypes.byref(fl))
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
InitSlot = Union[None, str, int]


def _validate(cfg: Pkcs11Config, init_slot: InitSlot) -> None:
    cfg.validate()
    if not cfg.token_label or cfg.token_serial:
        raise Pkcs11Error("provisioning selects and labels the token by PKCS11_TOKEN_LABEL only; unset PKCS11_TOKEN_SERIAL")
    if len(cfg.token_label.encode("utf-8")) > 32:
        raise Pkcs11Error(f"token label {cfg.token_label!r} is longer than 32 bytes")
    if not cfg.pin_file or cfg.pin:
        raise Pkcs11Error("provisioning writes the user PIN to PKCS11_PIN_FILE; set it and unset PKCS11_PIN")
    if not (cfg.sign_key_label and cfg.kem_key_label):
        raise Pkcs11Error("provisioning creates keys by label; set PKCS11_SIGN_KEY_LABEL and PKCS11_KEM_KEY_LABEL")
    if not (init_slot is None or init_slot == "first" or (isinstance(init_slot, int) and init_slot >= 0)):
        raise Pkcs11Error("init_slot must be None, 'first' or a slot id")


def _tokens(lib) -> List[Any]:
    return [slot.get_token() for slot in lib.get_slots(token_present=True)]


def _labelled(lib, label: str):
    found = [t for t in _tokens(lib) if (t.label or "").strip() == label]
    if len(found) > 1:
        raise Pkcs11Error(f"{len(found)} tokens are labelled {label!r}; refusing to choose")
    return found[0] if found else None


def _pick_uninitialised(P, lib, init_slot: InitSlot, module: str):
    uninit = sorted((t for t in _tokens(lib) if not (t.flags & P.TokenFlag.TOKEN_INITIALIZED)),
                    key=lambda t: t.slot.slot_id)
    if not uninit:
        raise Pkcs11Error(f"no uninitialised token on {module}; refusing to initialise an existing token")
    if isinstance(init_slot, int):
        chosen = [t for t in uninit if t.slot.slot_id == init_slot]
        if not chosen:
            raise Pkcs11Error(f"slot {init_slot} has no uninitialised token; refusing to initialise it")
        return chosen[0]
    if len(uninit) > 1 and init_slot != "first":
        ids = ", ".join(str(t.slot.slot_id) for t in uninit)
        raise Pkcs11Error(f"{len(uninit)} uninitialised tokens (slots {ids}); choose one with --init-slot")
    return uninit[0]


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


def provision(cfg: Pkcs11Config, *, pin_group: Optional[str] = None, init_slot: InitSlot = None,
              lib_factory: Optional[Callable[[str], Any]] = None) -> Dict[str, Any]:
    """Provision the token described by `cfg` (see the module docstring). Returns a JSON-serialisable report.

    `init_slot` matters only when no token carries the label and the module reports several uninitialised
    tokens: None refuses, "first" takes the lowest slot id, an int names the slot id. Every expected failure
    is raised as Pkcs11Error, with no secret in the message."""
    _validate(cfg, init_slot)
    P = _import_pkcs11()
    try:
        return _provision(P, cfg, pin_group, init_slot, lib_factory)
    except Pkcs11Error:
        raise
    except P.exceptions.PinIncorrect as exc:
        raise Pkcs11Error("incorrect PIN; stopping (repeated attempts lock the token)") from exc
    except (OSError, KeyError, P.exceptions.PKCS11Error) as exc:
        raise Pkcs11Error(f"provisioning failed: {type(exc).__name__}: {exc}") from exc


def _provision(P, cfg: Pkcs11Config, pin_group: Optional[str], init_slot: InitSlot,
               lib_factory: Optional[Callable[[str], Any]]) -> Dict[str, Any]:
    TF = P.TokenFlag
    if pin_group:
        import grp                                       # POSIX only; the PIN file group needs it
        try:
            gid: Optional[int] = grp.getgrnam(pin_group).gr_gid
        except KeyError as exc:
            raise Pkcs11Error(f"unknown group {pin_group!r}") from exc
    else:
        gid = None
    pin_mode = 0o440 if gid is not None else 0o400
    label, pin_file = cfg.token_label, os.path.abspath(os.path.expanduser(cfg.pin_file))
    pin_dir, so_pending = os.path.dirname(pin_file), pin_file + SO_PENDING_SUFFIX
    if not os.path.lexists(pin_dir):
        os.makedirs(pin_dir, mode=0o755)
    _check_dir(pin_dir)
    actions: List[str] = []

    with _Lock(pin_file + LOCK_SUFFIX):
        _cleanup_orphans(pin_dir)
        try:
            lib = (lib_factory or P.lib)(cfg.module)
        except Exception as exc:
            raise Pkcs11Error(f"cannot load PKCS#11 module {cfg.module!r}: {exc}") from exc
        pin_exists = os.path.lexists(pin_file)
        user_pin = _check_pin(_read_checked(pin_file, "PIN file", gid=gid), pin_file) if pin_exists else None

        tok = _labelled(lib, label)
        if tok is None:
            # Preflight, no writes yet: PIN file (read above), slot choice, mechanisms.
            target = _pick_uninitialised(P, lib, init_slot, cfg.module)
            _require_mechanisms(P, target.slot)
            if os.path.lexists(so_pending):              # no labelled token: C_InitToken never succeeded
                _discard(so_pending)
            if user_pin is None:
                user_pin = secrets.token_hex(16)
                _create_exclusive(pin_file, user_pin, pin_mode, gid)
                actions.append("created PIN file")
            else:
                actions.append("reused existing PIN file")
            so_pin = secrets.token_hex(16)
            slot_id = target.slot.slot_id
            _create_exclusive(so_pending, json.dumps({"v": 1, "module": cfg.module, "label": label,
                                                      "slot_id": slot_id, "so_pin": so_pin}), 0o400)
            if _labelled(lib, label) is not None or any(
                    t.slot.slot_id == slot_id and t.flags & TF.TOKEN_INITIALIZED for t in _tokens(lib)):
                raise Pkcs11Error(f"slot {slot_id} changed before initialisation; rerun provisioning")
            _c_init_token(cfg.module, slot_id, so_pin, label)
            actions.append(f"initialised token {label!r} in slot {slot_id}")
            tok = _labelled(lib, label)
            if tok is None:
                raise Pkcs11Error(f"token {label!r} not found after initialisation")
        elif user_pin is None:
            raise Pkcs11Error(f"token {label!r} exists but {pin_file} does not; this needs a manual repair "
                              "(a new PIN file would not match the token)")

        if not (tok.flags & TF.USER_PIN_INITIALIZED):
            if not os.path.lexists(so_pending):
                raise Pkcs11Error(f"token {label!r} has no user PIN and its SO PIN is not available; "
                                  "this needs a manual repair")
            try:
                rec = json.loads(_read_checked(so_pending, "SO PIN recovery file"))
                so_pin = rec["so_pin"]
            except (ValueError, KeyError, TypeError) as exc:
                raise Pkcs11Error(f"SO PIN recovery file {so_pending} is malformed") from exc
            if rec.get("module") != cfg.module or rec.get("label") != label:
                raise Pkcs11Error(f"SO PIN recovery file {so_pending} belongs to another module or label")
            if tok.flags & (TF.SO_PIN_LOCKED | TF.SO_PIN_FINAL_TRY):
                raise Pkcs11Error(f"SO PIN of token {label!r} is locked or on its final try; not trying it")
            with tok.open(rw=True, so_pin=so_pin) as s:
                s.init_pin(user_pin)
            actions.append("set the user PIN")
            tok = _labelled(lib, label)
        if os.path.lexists(so_pending):
            _discard(so_pending)
            actions.append("discarded the SO PIN")

        if tok.flags & (TF.USER_PIN_LOCKED | TF.USER_PIN_FINAL_TRY):
            raise Pkcs11Error(f"user PIN of token {label!r} is locked or on its final try; not trying it")
        _require_mechanisms(P, tok.slot)
        with tok.open(rw=True, user_pin=user_pin) as s:
            _ensure_keys(P, s, cfg, actions)

        # Runtime policy: sensitive, non-extractable, role-separated, distinct keys, KEM on P-256.
        prov = Pkcs11Provider(cfg, lib_factory=lambda _m: lib)
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
