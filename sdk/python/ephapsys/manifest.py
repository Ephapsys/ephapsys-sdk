# SPDX-License-Identifier: Apache-2.0
import json, hashlib, time
from pathlib import Path

NON_SECRET_FIELDS = {"agent_id","label","org_id","version","policy","models","certificate","allowed_hosts","created_at"}

def _sha256_bytes(b: bytes) -> str:
    return "sha256:" + hashlib.sha256(b).hexdigest()

def export_agent_manifest(
    agent_id: str,
    label: str,
    org_id: str,
    version: str,
    models: list,
    policy: dict | None = None,
    certificate: dict | None = None,
    allowed_hosts: list[str] | None = None,
    out_path: str | None = None,
) -> dict:
    """Create a NON-SECRET agent manifest (.agent.json).

    The manifest MUST NOT contain private keys, sealed seeds, or extractable secrets.
    It enumerates model digests, ECM URIs, hyperparameter signatures, and policy.
    Returns a dict with 'manifest', 'digest', and 'path' (if saved).
    """
    man = {
        "agent_id": agent_id,
        "label": label,
        "org_id": org_id,
        "version": version,
        "policy": policy or {},
        "models": models,
        "certificate": certificate or {},
        "allowed_hosts": allowed_hosts or [],
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "format": "ephapsys.agent+json;v=0.1"
    }
    # backfill rms_hash for any model entries that include weights/ecm and hyperparams
    try:
        from .digests import rms_hash as _rms_hash
    except Exception:
        _rms_hash = None
    for m in man.get("models", []):
        if "rms_hash" not in m:
            w = m.get("storage_path") or ""
            e = m.get("ecm_uri") or ""
            hp = m.get("hyperparams") if isinstance(m.get("hyperparams"), dict) else None
            if _rms_hash and w and e and hp and Path(w).exists() and Path(e).exists():
                try:
                    m["rms_hash"] = _rms_hash(w, e, hp)
                except Exception:
                    pass

    # sanity check for secrets (best-effort)
    s = json.dumps(man, separators=(",", ":"), sort_keys=True).encode("utf-8")
    digest = _sha256_bytes(s)
    # write file if requested
    saved = None
    if out_path:
        outp = Path(out_path)
        outp.parent.mkdir(parents=True, exist_ok=True)
        outp.write_text(json.dumps(man, indent=2), encoding="utf-8")
        saved = str(outp)
    return {"manifest": man, "digest": digest, "path": saved}
