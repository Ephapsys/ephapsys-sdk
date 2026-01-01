# SPDX-License-Identifier: Apache-2.0
#!/usr/bin/env python3
"""
Ephapsys SDK TPM integration (Linux + tpm2-tools)

Supports:
- Classic PCR-bound sealing/unsealing for DEKs
- PolicySigned + optional PCR composite policy for time-bounded "leases"
"""

import base64, os, subprocess, tempfile, shutil, json, sys, time
from typing import Sequence, Optional, Dict, Any

ALLOW_INSECURE = os.getenv("EPHAPSYS_ALLOW_INSECURE_TPM", "0") == "1"
TPM_SUPPORTED = sys.platform.startswith("linux")

# ---------------- Platform check ----------------
def _require_tpm():
    if not TPM_SUPPORTED:
        raise RuntimeError("TPM not supported on this platform (requires Linux + tpm2-tools)")

# ---------------- Helpers ----------------
def _run(cmd: list[str]) -> str:
    """Run a TPM2 command, raising on error."""
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(
            f"{' '.join(cmd)}\nEXIT {r.returncode}\nSTDERR:\n{r.stderr}\nSTDOUT:\n{r.stdout}"
        )
    return r.stdout

def _flush_session(sess_path: str) -> None:
    """Flush a session (handle both tpm2-tools variants)."""
    try:
        subprocess.run(["tpm2_flushcontext", sess_path], check=True, capture_output=True, text=True)
        return
    except Exception:
        subprocess.run(["tpm2_flushcontext", "-S", sess_path], check=True, capture_output=True, text=True)

def _pcr_sel(bank: str, pcrs: Sequence[int]) -> str:
    return f"{bank}:{','.join(str(i) for i in sorted(set(pcrs)))}"

# ---------------- Classic PCR-bound Seal/Unseal ----------------
def seal(dek: bytes, *, bank: str = "sha256", pcrs: Sequence[int] = (7,)) -> str:
    """
    Seal a 32-byte DEK to a PCR-bound policy using tpm2-tools.
    Returns a base64 JSON blob containing the sealed object + policy parameters.
    """
    if len(dek) != 32:
        raise ValueError("DEK must be 32 bytes")
    if ALLOW_INSECURE:
        return "INSECURE::" + base64.b64encode(dek).decode()
    _require_tpm()

    td = tempfile.mkdtemp(prefix="tpmseal_")
    try:
        primary = os.path.join(td, "primary.ctx")
        sess    = os.path.join(td, "sess.ctx")
        pol     = os.path.join(td, "policy.digest")
        pubf    = os.path.join(td, "key.pub")
        prvf    = os.path.join(td, "key.priv")
        dek_bin = os.path.join(td, "dek.bin")
        with open(dek_bin, "wb") as f:
            f.write(dek)

        _run(["tpm2_createprimary", "-C", "o", "-g", bank, "-G", "rsa", "-c", primary])
        sel = _pcr_sel(bank, pcrs)
        _run(["tpm2_startauthsession", "--policy-session", "-S", sess])
        _run(["tpm2_policypcr", "-S", sess, "-l", sel, "-L", pol])
        _flush_session(sess)

        _run(["tpm2_create", "-C", primary, "-u", pubf, "-r", prvf, "-L", pol, "-i", dek_bin])

        blob = {
            "type": "pcr",
            "bank": bank,
            "pcrs": list(sorted(set(int(i) for i in pcrs))),
            "pub_b64": base64.b64encode(open(pubf, "rb").read()).decode(),
            "priv_b64": base64.b64encode(open(prvf, "rb").read()).decode(),
        }
        return base64.b64encode(json.dumps(blob, separators=(",", ":")).encode()).decode()
    finally:
        shutil.rmtree(td, ignore_errors=True)

def unseal(sealed_blob_b64: str) -> bytes:
    """
    Unseal a PCR-bound DEK (PCRs must match current system state).
    """
    if sealed_blob_b64.startswith("INSECURE::"):
        return base64.b64decode(sealed_blob_b64.split("::", 1)[1])
    _require_tpm()

    blob = json.loads(base64.b64decode(sealed_blob_b64).decode())
    bank = blob["bank"]
    pcrs = [int(i) for i in blob["pcrs"]]
    pub  = base64.b64decode(blob["pub_b64"])
    prv  = base64.b64decode(blob["priv_b64"])

    td = tempfile.mkdtemp(prefix="tpmseal_")
    try:
        primary = os.path.join(td, "primary.ctx")
        keyctx  = os.path.join(td, "key.ctx")
        pubf    = os.path.join(td, "key.pub")
        prvf    = os.path.join(td, "key.priv")
        sess    = os.path.join(td, "sess.ctx")

        open(pubf, "wb").write(pub)
        open(prvf, "wb").write(prv)

        _run(["tpm2_createprimary", "-C", "o", "-g", bank, "-G", "rsa", "-c", primary])
        _run(["tpm2_load", "-C", primary, "-u", pubf, "-r", prvf, "-c", keyctx])

        sel = _pcr_sel(bank, pcrs)
        _run(["tpm2_startauthsession", "--policy-session", "-S", sess])
        _run(["tpm2_policypcr", "-S", sess, "-l", sel])
        dek = subprocess.check_output(["tpm2_unseal", "-c", keyctx, "-p", f"session:{sess}"])
        _flush_session(sess)
        return dek
    finally:
        shutil.rmtree(td, ignore_errors=True)

# ---------------- PolicySigned Composite Seal/Unseal ----------------
def seal_with_policy_signed(
    dek: bytes,
    issuer_pub_pem: str,
    *,
    bank: str = "sha256",
    pcrs: Sequence[int] = (7,),
) -> str:
    """
    Seal a DEK under a composite policy:
    PolicyPCR(bank,pcrs) ∧ PolicySigned(issuer_pub_pem)

    Returns a base64 JSON blob including policy + sealed object.
    """
    if len(dek) != 32:
        raise ValueError("DEK must be 32 bytes")
    if ALLOW_INSECURE:
        return "INSECURE::" + base64.b64encode(dek).decode()
    _require_tpm()

    td = tempfile.mkdtemp(prefix="tpmseal_")
    try:
        primary = os.path.join(td, "primary.ctx")
        sess    = os.path.join(td, "sess.ctx")
        pol     = os.path.join(td, "policy.digest")
        pubf    = os.path.join(td, "key.pub")
        prvf    = os.path.join(td, "key.priv")
        dek_bin = os.path.join(td, "dek.bin")
        signer  = os.path.join(td, "issuer.pub")
        with open(dek_bin, "wb") as f:
            f.write(dek)
        with open(signer, "wb") as f:
            f.write(issuer_pub_pem.encode())

        _run(["tpm2_createprimary", "-C", "o", "-g", bank, "-G", "rsa", "-c", primary])

        # Start composite policy session
        _run(["tpm2_startauthsession", "--policy-session", "-S", sess])
        sel = _pcr_sel(bank, pcrs)
        _run(["tpm2_policypcr", "-S", sess, "-l", sel])
        _run(["tpm2_policysigned", "-S", sess, "-L", pol, "-f", signer, "-g", bank])
        _flush_session(sess)

        _run(["tpm2_create", "-C", primary, "-u", pubf, "-r", prvf, "-L", pol, "-i", dek_bin])

        blob = {
            "type": "policy_signed",
            "bank": bank,
            "pcrs": list(sorted(set(int(i) for i in pcrs))),
            "issuer_pub_pem": issuer_pub_pem,
            "pub_b64": base64.b64encode(open(pubf, "rb").read()).decode(),
            "priv_b64": base64.b64encode(open(prvf, "rb").read()).decode(),
        }
        return base64.b64encode(json.dumps(blob, separators=(",", ":")).encode()).decode()
    finally:
        shutil.rmtree(td, ignore_errors=True)

def unseal_with_lease(
    sealed_blob_b64: str,
    issuer_sig_b64: str,
    lease_json: str,
) -> bytes:
    """
    Unseal a PolicySigned DEK using a lease signed by issuer.

    lease_json: JSON string the org signed.
    issuer_sig_b64: base64(issuer_signature over lease_json)

    The TPM verifies the signature inside the policy session.
    """
    if sealed_blob_b64.startswith("INSECURE::"):
        return base64.b64decode(sealed_blob_b64.split("::", 1)[1])
    _require_tpm()

    blob = json.loads(base64.b64decode(sealed_blob_b64).decode())
    bank = blob["bank"]
    pcrs = [int(i) for i in blob["pcrs"]]
    issuer_pub_pem = blob["issuer_pub_pem"]
    pub  = base64.b64decode(blob["pub_b64"])
    prv  = base64.b64decode(blob["priv_b64"])

    lease = json.loads(lease_json)
    exp = lease.get("expires_at")
    if exp and time.time() > _parse_expiration(exp):
        raise RuntimeError("Lease expired")

    td = tempfile.mkdtemp(prefix="tpmlease_")
    try:
        primary = os.path.join(td, "primary.ctx")
        keyctx  = os.path.join(td, "key.ctx")
        pubf    = os.path.join(td, "key.pub")
        prvf    = os.path.join(td, "key.priv")
        sess    = os.path.join(td, "sess.ctx")
        signer  = os.path.join(td, "issuer.pub")
        msg     = os.path.join(td, "lease.json")
        sig     = os.path.join(td, "issuer.sig")

        open(pubf, "wb").write(pub)
        open(prvf, "wb").write(prv)
        open(signer, "wb").write(issuer_pub_pem.encode())
        open(msg, "w").write(lease_json)
        open(sig, "wb").write(base64.b64decode(issuer_sig_b64))

        _run(["tpm2_createprimary", "-C", "o", "-g", bank, "-G", "rsa", "-c", primary])
        _run(["tpm2_load", "-C", primary, "-u", pubf, "-r", prvf, "-c", keyctx])

        sel = _pcr_sel(bank, pcrs)
        _run(["tpm2_startauthsession", "--policy-session", "-S", sess])
        _run(["tpm2_policypcr", "-S", sess, "-l", sel])
        # Apply the signed lease
        _run(["tpm2_policysigned", "-S", sess, "-f", signer, "-g", bank,
              "-p", msg, "-s", sig])
        dek = subprocess.check_output(["tpm2_unseal", "-c", keyctx, "-p", f"session:{sess}"])
        _flush_session(sess)
        return dek
    finally:
        shutil.rmtree(td, ignore_errors=True)

def _parse_expiration(exp_val: Any) -> float:
    """Parse ISO8601 or numeric timestamp."""
    if isinstance(exp_val, (int, float)):
        return float(exp_val)
    try:
        import datetime
        return datetime.datetime.fromisoformat(exp_val.replace("Z", "+00:00")).timestamp()
    except Exception:
        raise RuntimeError(f"Invalid expires_at format: {exp_val}")
