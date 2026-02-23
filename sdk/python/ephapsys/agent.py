# SPDX-License-Identifier: Apache-2.0
# agent.py
from __future__ import annotations

import os, json, pathlib, hashlib, shutil, base64, time, subprocess, sys, platform, io, warnings, re, shlex, glob
from contextlib import contextmanager, redirect_stdout, redirect_stderr
from typing import Dict, Any, List, Tuple, Optional, Callable, Set, Union

from .http import request
from .auth import get_api_key

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import NameOID


import logging

logger = logging.getLogger("ephapsys.sdk")

# default: INFO unless overridden
log_level = os.getenv("EPHAPSYS_LOG_LEVEL", "WARNING").upper()
logging.basicConfig(level=getattr(logging, log_level, logging.INFO))

DEFAULT_OUTPUT_GUARDRAILS: Dict[str, Any] = {
    "enabled": False,
    "categories": ["pii", "hate"],
    "threshold": 0.5,
    "action": "block",
}
DEFAULT_MAX_INPUT_BYTES = 5 * 1024 * 1024  # 5 MB safeguard
DEFAULT_POLICY_TELEMETRY = os.getenv("AOC_POLICY_TELEMETRY", "1") != "0"
DEFAULT_ALLOWED_AUDIO_MIME = [m.strip().lower() for m in (os.getenv("AOC_ALLOWED_AUDIO_MIME", "audio/wav,audio/mpeg").split(",")) if m.strip()]
DEFAULT_MAX_TOKEN_LENGTH = int(os.getenv("AOC_MAX_TOKENS", "4096"))
DEFAULT_AV_SCAN = os.getenv("AOC_AV_SCAN", "0") == "1"
REGULATED_MODE = os.getenv("AOC_REGULATED", "0") == "1"
RESIDENCY_TAG = os.getenv("AOC_RESIDENCY_TAG", "").strip()
SAFETY_BLOCKLIST = [p.strip() for p in os.getenv("AOC_SAFETY_BLOCKLIST", "").split(",") if p.strip()]

# Prompt-injection signatures (expanded). Can be extended via policy patterns.
PROMPT_INJECTION_PATTERNS = [
    r"ignore (all )?previous instructions",
    r"you are now (the|my) system",
    r"disregard (?:rules|safety|guardrails)",
    r"begin jailbreak",
    r"### system prompt ###",
    r"unfiltered output",
    r"base64 decode",
    r"role ?play",
    r"system: .*assistant:",
    r"act as (an?|the) system",
    r"pretend to be",
    r"simulate (?:a|the) terminal",
    r"sysprompt",
    r"developer message",
    r"strip safety",
    r"override .*safety",
    r"prompt injection",
    r"follow my new rules",
    r"BEGIN( SYSTEM)? PROMPT",
    r"### instruction ###",
    r"disobey previous",
    r"bypass filter",
    r"answer unfiltered",
    r"remove content policy",
    r"system override",
    r"internal prompt",
    r"developer prompt",
    r"primary instructions",
    r"forget .* rules",
    r"do not follow safety",
    r"respond in character .* no matter what",
]

GUARDRAIL_PATTERNS: Dict[str, List[str]] = {
    "profanity": [
        r"\b(fuck|shit|bitch|asshole|bastard|damn)\b",
    ],
    "hate": [
        r"\b(?:kill|hurt|attack)\s+(?:them|him|her|you|people)\b",
        r"\b(?:slur|racist|bigot)\b",
    ],
    "sexual": [
        r"\b(?:porn|xxx|sexual|explicit)\b",
    ],
    "violence": [
        r"\b(?:kill|murder|shoot|stab|violent)\b",
    ],
    "pii": [
        r"\b\d{3}-\d{2}-\d{4}\b",
        r"\b\d{4}\s?\d{4}\s?\d{4}\s?\d{4}\b",
        r"\b\d{9}\b",
    ],
    "self_harm": [
        r"\b(?:suicide|self-harm|kill myself)\b",
    ],
    "medical": [
        r"\b(?:prescription|diagnosed|medical record)\b",
    ],
}

def _moderate_text(text: str, categories: List[str], action: str, threshold: float = 0.5) -> Tuple[Optional[str], List[str]]:
    """
    Simple regex-based moderation. Returns (possibly redacted text or None, hits).
    If action == block and hits, returns None.
    If action == redact and hits, redacts matched spans.
    If action == flag, returns original text and hits.
    """
    hits: List[str] = []
    redacted = text
    for cat in categories:
        pats = GUARDRAIL_PATTERNS.get(cat, [])
        for pat in pats:
            try:
                if re.search(pat, text, flags=re.IGNORECASE):
                    hits.append(cat)
                    if action == "redact":
                        redacted = re.sub(pat, "[REDACTED]", redacted, flags=re.IGNORECASE)
                    # threshold is unused in regex-only pass; kept for API parity
            except re.error:
                continue
    if hits:
        if action == "block":
            return None, hits
        if action == "redact":
            return redacted, hits
    return text, hits


def _safety_scan(text: str) -> Tuple[bool, List[str]]:
    """
    Lightweight safety/abuse heuristic: flags jailbreak/malware/self-harm cues.
    """
    cues = [
        r"jailbreak",
        r"rootkit",
        r"malware",
        r"exploit",
        r"ransomware",
        r"kill everyone",
        r"self[- ]harm",
        r"suicide",
        r"override safety",
        r"disable guardrails",
        r"weapon",
        r"bomb",
        r"violent attack",
        r"harm humans",
        r"chemical weapon",
        r"bioweapon",
        r"shoot .*school",
        r"mass casualty",
        r"child abuse",
        r"terrorist",
        r"bomb recipe",
        r"build a weapon",
        r"kill list",
        r"violent manifesto",
        r"hate speech",
        r"racial slur",
        r"genocide",
    ]
    # Allow env-provided blocklist patterns
    cues.extend(SAFETY_BLOCKLIST)
    hits = []
    for pat in cues:
        try:
            if re.search(pat, text, re.IGNORECASE):
                hits.append(pat)
        except re.error:
            continue
    return bool(hits), hits


def _validate_io(kind: str, phase: str, value: Any, max_bytes: int, av_scanner: Optional[Callable[[bytes], bool]]) -> None:
    """
    Basic input/output validation by model kind.
    Raises ValueError if validation fails.
    """
    k = (kind or "").lower()
    if phase == "input":
        if k == "language" or k == "tts":
            if not isinstance(value, str):
                raise ValueError(f"{k} input must be a string")
        elif k in {"stt", "audio"}:
            if not isinstance(value, (bytes, bytearray)):
                raise ValueError(f"{k} input must be raw bytes")
            if len(value) > max_bytes:
                raise ValueError(f"{k} input exceeds max size of {max_bytes} bytes")
            # basic MIME sniff via headers if present
            mime = None
            if isinstance(value, (bytes, bytearray)) and value[:4]:
                # crude header checks (RIFF/WAV, ID3/MP3, Ogg)
                header = value[:4]
                if header.startswith(b"RIFF"):
                    mime = "audio/wav"
                elif header.startswith(b"ID3") or header[0:1] == b"\xff":
                    mime = "audio/mpeg"
                elif header.startswith(b"OggS"):
                    mime = "audio/ogg"
            if mime and DEFAULT_ALLOWED_AUDIO_MIME and mime.lower() not in DEFAULT_ALLOWED_AUDIO_MIME:
                raise ValueError(f"{k} input MIME {mime} not allowed")
            if (av_scanner and DEFAULT_AV_SCAN) or (av_scanner and REGULATED_MODE):
                clean = av_scanner(value)
                if not clean:
                    raise ValueError(f"{k} input failed AV scan")
            elif REGULATED_MODE and not av_scanner:
                raise ValueError(f"{k} input requires AV scan in regulated mode")
        elif k == "vision":
            try:
                from PIL import Image  # type: ignore
            except ImportError:
                Image = None
            if not (Image and isinstance(value, Image.Image)) and not (
                isinstance(value, dict) and value.get("image") is not None
            ):
                raise ValueError("vision input must be a PIL.Image or dict with image key")
        # other kinds: accept as-is
    else:
        # output phase: minimal validation (ensure not None)
        if value is None:
            raise ValueError(f"{k} output is None")


def _validate_schema(value: Any, schema: Optional[Dict[str, Any]]) -> None:
    """
    Lightweight schema validation: ensure required keys/types match a simple {key: type} map.
    """
    if not schema or not isinstance(value, dict):
        return
    for key, expected in schema.items():
        if key not in value:
            raise ValueError(f"schema missing key: {key}")
        if expected is None:
            continue
        if isinstance(expected, type):
            if not isinstance(value.get(key), expected):
                raise ValueError(f"schema type mismatch for {key}: expected {expected}, got {type(value.get(key))}")
        elif isinstance(expected, str):
            # allow simple type names
            if expected == "number" and not isinstance(value.get(key), (int, float)):
                raise ValueError(f"schema type mismatch for {key}: expected number")
            if expected == "string" and not isinstance(value.get(key), str):
                raise ValueError(f"schema type mismatch for {key}: expected string")


def _enforce_network_allowlist(url: str, allowed: Optional[List[str]]) -> None:
    """
    Enforce a simple domain allowlist for outbound calls (tooling/downloads).
    """
    if not allowed:
        return
    try:
        from urllib.parse import urlparse
        host = urlparse(url).hostname or ""
    except Exception:
        host = ""
    host = host.lower()
    allowed_hosts = [h.lower() for h in allowed if h]
    if host and any(host.endswith(a) or host == a for a in allowed_hosts):
        return
    raise RuntimeError(f"Outbound network to {host or url} not allowed (allowed: {allowed_hosts})")


def _record_policy_telemetry(
    request_fn,
    api_base: str,
    headers: Dict[str, str],
    verify_ssl: bool,
    *,
    agent_id: str,
    model_kind: str,
    phase: str,
    decision: str,
    applied: List[str],
) -> None:
    if not DEFAULT_POLICY_TELEMETRY:
        return
    payload = {
        "event": "policy_decision",
        "agent_id": agent_id,
        "model_kind": model_kind,
        "phase": phase,
        "decision": decision,
        "applied": applied,
    }
    try:
        request_fn("POST", api_base, "/telemetry", headers=headers, json_body=payload, verify_ssl=verify_ssl)
    except Exception:
        # best-effort; do not break inference
        pass


def _record_policy_audit(
    request_fn,
    api_base: str,
    headers: Dict[str, str],
    verify_ssl: bool,
    *,
    agent_id: str,
    model_kind: str,
    decision: str,
    attestation_digest: Optional[str],
    applied: Optional[List[str]] = None,
) -> None:
    if not DEFAULT_POLICY_TELEMETRY:
        return
    payload = {
        "event": "policy_audit",
        "agent_id": agent_id,
        "model_kind": model_kind,
        "decision": decision,
    }
    if attestation_digest:
        payload["attestation_digest"] = attestation_digest
    if applied:
        payload["applied"] = applied
    try:
        request_fn("POST", api_base, "/telemetry", headers=headers, json_body=payload, verify_ssl=verify_ssl)
    except Exception:
        pass


# ---------- utils ----------
def sha256_file(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _mkdir(p: pathlib.Path) -> pathlib.Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def _is_http(url: str) -> bool:
    return url.startswith("http://") or url.startswith("https://")


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


# FIXME: Delete this for production; used to allow dev evidence stubs.
def _dev_allow_insecure() -> bool:
    return os.getenv("EPHAPSYS_ALLOW_INSECURE_PERSONALIZE", "0") == "1"


# ---------- TrustedAgent ----------
class TrustedAgent:
    """
    Talks to backend for agent status/certs,
    performs personalization (TPM/TEE/HSM/dSIM),
    prepares runtime artifacts (base model + ECM).
    """

    def __init__(
        self,
        agent_id: str,
        api_base: str,
        api_key: Optional[str] = None,
        storage_dir: str = ".ephapsys_state",
        verify_ssl: bool = True,
    ):
        self.agent_id = agent_id
        # self.api_base = api_base.rstrip("/")
        self.api_base = api_base
        self.api_key = get_api_key(
            api_key,
            base_url=api_base,
            agent_instance_id=agent_id,
            verify_ssl=verify_ssl,
        )
        self.verify_ssl = verify_ssl
        self.storage_dir = pathlib.Path(storage_dir)
        _mkdir(self.storage_dir)
        # Cached runtime map; populated by prepare_runtime()/run()
        self._runtime_cache: Optional[Dict[str, Dict[str, Any]]] = None
        self._identity_history: List[str] = []
        self._initial_agent_id = agent_id
        self._record_identity(agent_id)
        self._optional_artifacts: Set[str] = {
            "metrics.json",
            "metrics.csv",
            "loss_comparison.png",
            "accuracy_comparison.png",
            "perplexity_comparison.png",
            "report.md",
            "ecm.json",
        }
        self._geo: Optional[Tuple[float, float]] = None
        self._attestation_digest: Optional[str] = None
        self._av_scanner: Optional[Callable[[bytes], bool]] = None
        self._output_schema: Optional[Dict[str, Any]] = None
        self._max_tokens_cap: int = DEFAULT_MAX_TOKEN_LENGTH
        self._minimal_logging: bool = False
        if REGULATED_MODE:
            self._minimal_logging = True

    @classmethod
    def from_env(cls):
        storage = os.getenv("EPHAPSYS_STORAGE_DIR", ".ephapsys_state")
        path = pathlib.Path(storage) / "agent_id"
        if path.exists():
            agent_id = path.read_text().strip()
        else:
            agent_id = (
                os.getenv("AGENT_TEMPLATE_ID")
            )
        api_base = os.getenv("AOC_BASE_URL") or os.getenv("AOC_API_URL")
        verify = os.getenv("AOC_VERIFY_SSL", "1") != "0"
        return cls(agent_id, api_base, api_key=None, storage_dir=storage, verify_ssl=verify)

    def _headers(self) -> Dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}", "Accept": "application/json"}

    def set_av_scanner(self, scanner: Callable[[bytes], bool]) -> None:
        """
        Register an AV scanner callback that returns True if clean, False otherwise.
        """
        self._av_scanner = scanner

    def set_output_schema(self, schema: Dict[str, Any]) -> None:
        """
        Register an output schema (simple key/type map) to validate structured outputs.
        """
        self._output_schema = schema

    def update_geo(self, latitude: float, longitude: float) -> None:
        """Persist the latest geolocation to include in telemetry."""
        try:
            self._geo = (float(latitude), float(longitude))
        except Exception as exc:
            logger.warning("⚠️ Invalid geo coordinates (%s, %s): %s", latitude, longitude, exc)
            self._geo = None

    # ---------- agent info/status/models/certs ----------
    def info(self) -> Dict[str, Any]:
        return request(
            "GET",
            self.api_base,
            f"/agents/{self.agent_id}",
            headers=self._headers(),
            verify_ssl=self.verify_ssl,
        )

    def get_status(self) -> Dict[str, Any]:
        return request(
            "GET",
            self.api_base,
            f"/agents/{self.agent_id}/status",
            headers=self._headers(),
            verify_ssl=self.verify_ssl,
        )

    # ---------- Verify() Method ----------
    def verify(self) -> Tuple[bool, Dict[str, Any]]:
        """
        Verify agent status, certificates, models, and policies.
        Returns (ok, report) where report includes detailed checks.
        """
        report = {"agent_id": self.agent_id, "checks": []}
        ok = True

        # --- Agent status check (backend status endpoint) ---
        logger.debug("Verifying agent_id=%s", self.agent_id)
        st_full = self.get_status()
        logger.debug("get_status() → status=%s enabled=%s personalized=%s",
             st_full.get('status'), st_full.get('enabled'), st_full.get('personalized'))

        status = (st_full.get("status") or "disabled").lower()
        state = st_full.get("state", {})

        is_enabled = (status == "enabled") or st_full.get("enabled")
        is_revoked = state.get("revoked", False) or (status == "revoked") or st_full.get("revoked")
        is_personalized = state.get("personalized", False) or st_full.get("personalized")

        # Use helper for anchor/mode
        anchor, mode = self.get_anchor_mode()
        logger.debug("Parsed: is_enabled=%s, is_revoked=%s, is_personalized=%s, anchor=%s, mode=%s",
             is_enabled, is_revoked, is_personalized, anchor, mode)

        status_ok = is_enabled and not is_revoked and is_personalized
        if not status_ok:
            logger.warning("❌ Agent status check failed")
            ok = False
        else:
            logger.debug("✅ Agent status check passed")

        report["checks"].append(
            {
                "check": "agent_status",
                "ok": status_ok,
                "status": status,
                "personalized": is_personalized,
                "anchor": anchor,
                "mode": mode,
            }
        )

        # --- Certificates check ---
        certs = self.certificates()
        active = [c for c in certs if str(c.get("status", "")).upper() == "ISSUED"]
        revoked = [c for c in certs if str(c.get("status", "")).upper() == "REVOKED"]

        certs_ok = bool(active)  # require at least one valid cert
        if not certs_ok:
            logger.warning("❌ Certificates check failed (no active certs found)")
            ok = False
        else:
            logger.debug("✅ Certificates check passed (active=%d)", len(active))

        report["checks"].append(
            {
                "check": "certificates",
                "ok": certs_ok,
                "active": len(active),
                "revoked_ignored": len(revoked),
            }
        )

        # --- Models check ---
        models = []
        agent_doc = st_full.get("agent") or {}
        if isinstance(agent_doc, dict):
            models = agent_doc.get("models") or []

        if not models:
            # fallback to API call
            models = self.models()

        if not models:
            logger.warning("❌ Models check failed (no models attached)")
            ok = False
            report["checks"].append({"check": "models", "ok": False, "why": "No models attached"})
        else:
            logger.debug("✅ Models check passed (count=%d)", len(models))
            report["checks"].append({"check": "models", "ok": True, "count": len(models)})

        # --- Agent-level policies check (strict enforcement) ---
        policies = st_full.get("policy", {})
        if policies:
            try:
                applied = self.enforce_policies_agent()
                logger.info("✅ Agent-level policies enforced: %s", applied)
                report["checks"].append({
                    "check": "policy",
                    "ok": True,
                    "policies": policies,
                    "applied": applied,
                })
            except RuntimeError as e:
                logger.error("❌ Agent-level policy violation: %s", e)
                ok = False
                report["checks"].append({
                    "check": "policy",
                    "ok": False,
                    "error": str(e),
                    "policies": policies
                })
        else:
            logger.info("ℹ️ No agent-level policies attached")
            report["checks"].append({
                "check": "policy",
                "ok": True,
                "policies": {}
            })

        # --- Model-level policies check (report only, enforcement happens at runtime) ---
        model_policies = st_full.get("model_policies", [])
        if model_policies:
            logger.debug("✅ Model-level policies present:")
            for mp in model_policies:
                mid = mp.get("id")
                mtype = mp.get("type")
                for p in mp.get("policies", []):
                    val = p.get("value")
                    if isinstance(val, (dict, list)):
                        pretty_val = json.dumps(val, indent=2)
                    else:
                        pretty_val = str(val)
                    logger.debug("   • [%s:%s] %s =", mid, mtype, p.get("type"))
                    for line in pretty_val.splitlines():
                        logger.debug("       %s", line)
            report["checks"].append({
                "check": "model_policies",
                "ok": True,
                "model_policies": model_policies
            })
        else:
            logger.debug("ℹ️ No model-level policies attached")
            report["checks"].append({
                "check": "model_policies",
                "ok": True,
                "model_policies": []
            })

        # --- Quick Telemetry  ---
        try:
            request("POST", self.api_base, "/telemetry", headers=self._headers(),
            json_body={"event": "verification", "agent_id": self.agent_id},
            verify_ssl=self.verify_ssl)
        except Exception as e:
            logger.warning("⚠️ Telemetry logging failed: %s", e)

        # --- Final result ---
        logger.debug("FINAL RESULT: ok=%s", ok)
        return ok, report

    # --------- Network allowlist hook (optional) ----------
    def enforce_network_allowlist(self, url: str) -> None:
        """
        Enforce outbound network allowlist if configured via env AOC_NETWORK_ALLOWLIST (comma-separated domains).
        """
        raw = os.getenv("AOC_NETWORK_ALLOWLIST", "")
        allowed = [t.strip() for t in raw.split(",") if t.strip()]
        _enforce_network_allowlist(url, allowed)



    def enforce_policies_agent(self) -> List[str]:
        """
        Enforce agent-level policies that apply globally (e.g., lease_duration, jurisdiction, resource caps).
        Returns a list of applied policy labels.
        Raises RuntimeError if a hard policy violation occurs.
        """
        import psutil, locale

        applied = []
        st_full = self.get_status()
        agent_policies = st_full.get("policy", {})
        if not agent_policies:
            return applied

        now = int(time.time())

        # --- Lease duration ---
        if "lease_duration" in agent_policies:
            lease_days = int(agent_policies["lease_duration"] or 0)
            created_at = st_full.get("agent", {}).get("created_at", now)
            expires_at = created_at + lease_days * 86400
            if lease_days > 0 and now > expires_at:
                raise RuntimeError(f"Lease expired {lease_days}d after creation (created_at={created_at})")
            applied.append(f"lease_duration={lease_days}")

        # --- Jurisdiction enforcement (strict, with fallback to locale) ---
        if agent_policies.get("jurisdiction"):
            allowed_raw = str(agent_policies["jurisdiction"]).upper()

            # Resolve current jurisdiction
            current = os.getenv("CURRENT_JURISDICTION", "").upper()
            if not current:
                # fallback: use system locale (e.g., "en_US" → "US")
                try:
                    loc = locale.getdefaultlocale()[0] or ""
                    current = loc.split("_")[1].upper() if "_" in loc else loc.upper()
                except Exception:
                    current = ""

            if not current:
                raise RuntimeError(f"Jurisdiction enforcement active but cannot resolve current jurisdiction (allowed={allowed_raw})")

            logger.info("Jurisdiction check: current=%s allowed=%s", current, allowed_raw)

            # Allow comma/semicolon separated lists and treat WORLDWIDE as wildcard.
            tokens = [token.strip() for token in re.split(r"[;,]", allowed_raw) if token.strip()]
            allowed_values = tokens or [allowed_raw]
            allowed_all = "WORLDWIDE" in allowed_values

            if not allowed_all and current not in allowed_values:
                raise RuntimeError(
                    f"Jurisdiction violation: current={current} not allowed (expected one of {allowed_values})"
                )

            applied.append(f"jurisdiction={allowed_raw}")

        # --- Logging level ---
        if agent_policies.get("logging"):
            applied.append(f"logging={agent_policies['logging']}")

        # --- RAM cap (hard enforcement) ---
        if agent_policies.get("ram_cap"):
            try:
                ram_limit_mb = int(agent_policies["ram_cap"])
                ram_used_mb = psutil.virtual_memory().used // (1024 * 1024)
                logger.debug("RAM check: used=%dMB cap=%dMB", ram_used_mb, ram_limit_mb)
                if ram_used_mb > ram_limit_mb:
                    raise RuntimeError(f"RAM cap exceeded: used={ram_used_mb}MB > cap={ram_limit_mb}MB")
                applied.append(f"ram_cap={ram_limit_mb}MB")
            except Exception as e:
                raise RuntimeError(f"RAM cap enforcement failed: {e}")

        # --- VRAM cap (hard enforcement) ---
        if agent_policies.get("vram_cap"):
            try:
                import torch
                if torch.cuda.is_available():
                    free, total = torch.cuda.mem_get_info()
                    vram_used_mb = (total - free) // (1024 * 1024)
                    vram_limit_mb = int(agent_policies["vram_cap"])
                    logger.debug("VRAM check: used=%dMB cap=%dMB", vram_used_mb, vram_limit_mb)
                    if vram_used_mb > vram_limit_mb:
                        raise RuntimeError(f"VRAM cap exceeded: used={vram_used_mb}MB > cap={vram_limit_mb}MB")
                    applied.append(f"vram_cap={vram_limit_mb}MB")
            except Exception as e:
                raise RuntimeError(f"VRAM cap enforcement failed: {e}")

        return applied


    def enforce_policies_model_kind(
        self, value: Any, kind: str, phase: str
    ) -> Tuple[Optional[Any], List[str]]:
        """
        Enforce policies for a given model kind and phase ("input" or "output").
        Returns (possibly modified value, list of applied policy labels).
        - If enforcement blocks the value, returns (None, [applied_policies]).
        - Language policies act on strings.
        - Vision policies act on image inputs or labels.
        - RL policies act on reward/obs/action values.
        - STT/TTS policies act on audio/text payloads.
        """
        applied = []
        decision = "ok"
        st_full = self.get_status()
        model_policies = st_full.get("model_policies", [])
        if not model_policies:
            return value, applied

        for mp in model_policies:
            if mp.get("type") != kind:
                continue
            for p in mp.get("policies", []):
                ptype, val = p.get("type"), p.get("value")

                # ---------------- LANGUAGE ----------------
                if kind == "language":
                    if ptype == "prompt_filter" and phase == "input":
                        regexes = val.get("regex_blocklist", [])
                        blocked = any(rex.lower() in str(value).lower() for rex in regexes)
                        if blocked:
                            reason = f"prompt_filter:block:{regexes}"
                            decision = "blocked"
                            _record_policy_audit(request, self.api_base, self._headers(), self.verify_ssl,
                                                 agent_id=self.agent_id, model_kind=kind, decision=decision,
                                                 attestation_digest=self._attestation_digest, applied=applied + [reason])
                            return None, applied + [reason]
                        applied.append("prompt_filter=ok")

                    # Resource/cost cap: hard max token length across all language inputs
                    toks = str(value).split()
                    if phase == "input" and len(toks) > self._max_tokens_cap:
                        value = " ".join(toks[:self._max_tokens_cap])
                        applied.append(f"resource_limit:max_tokens={self._max_tokens_cap}")

                    if ptype == "prompt_injection" and phase == "input":
                        sensitivity = float(val.get("sensitivity", 0.5))
                        action = val.get("action", "block")
                        patterns = val.get("patterns") or []
                        haystack = str(value).lower()
                        score = 0.0
                        hits: List[str] = []
                        for pat in PROMPT_INJECTION_PATTERNS + patterns:
                            try:
                                if re.search(pat, haystack, re.IGNORECASE):
                                    hits.append(pat)
                            except re.error:
                                # if user provided bad regex, skip it
                                continue
                        if hits:
                            # simple scoring: hits normalized by length
                            score = min(1.0, 0.3 + 0.1 * len(hits))
                        if score >= sensitivity:
                            if action == "block":
                                reason = f"prompt_injection:block@{score:.2f}:{hits}"
                                decision = "blocked"
                                _record_policy_audit(request, self.api_base, self._headers(), self.verify_ssl,
                                                     agent_id=self.agent_id, model_kind=kind, decision=decision,
                                                     attestation_digest=self._attestation_digest, applied=applied + [reason])
                                return None, applied + [reason]
                            applied.append(f"prompt_injection:flag@{score:.2f}:{hits}")
                        else:
                            applied.append(f"prompt_injection:ok@{score:.2f}")

                    if ptype == "context_truncation" and phase == "input":
                        max_tokens = int(val.get("max_tokens") or 2048)
                        strategy = (val.get("strategy") or "tail").lower()
                        toks = str(value).split()
                        if len(toks) > max_tokens:
                            if strategy == "head":
                                toks = toks[:max_tokens]
                            elif strategy == "middle":
                                keep_head = max_tokens // 2
                                keep_tail = max_tokens - keep_head
                                toks = toks[:keep_head] + toks[-keep_tail:]
                            else:  # tail default
                                toks = toks[-max_tokens:]
                            value = " ".join(toks)
                            applied.append(f"context_truncation={max_tokens}:{strategy}")

                    if ptype == "tool_allowlist" and phase == "input":
                        allowed = [t.strip().lower() for t in (val.get("allowed") or []) if t.strip()]
                        # assume a simple convention: value may be dict with tool name, or string containing tool name
                        tool_name = None
                        if isinstance(value, dict):
                            tool_name = str(value.get("tool") or value.get("tool_name") or "").lower()
                        elif isinstance(value, str):
                            tool_name = value.strip().lower()
                        if allowed and tool_name and tool_name not in allowed:
                            reason = f"tool_allowlist:block:{tool_name}"
                            decision = "blocked"
                            _record_policy_audit(request, self.api_base, self._headers(), self.verify_ssl,
                                                 agent_id=self.agent_id, model_kind=kind, decision=decision,
                                                 attestation_digest=self._attestation_digest, applied=applied + [reason])
                            return None, applied + [reason]
                        if allowed:
                            applied.append(f"tool_allowlist=ok:{allowed}")

                    if ptype == "max_tokens" and phase == "input":
                        max_toks = int(val or 256)
                        toks = str(value).split()
                        if len(toks) > max_toks:
                            value = " ".join(toks[:max_toks])
                            applied.append(f"max_tokens={max_toks}")

                    if ptype == "output_moderation" and phase == "output":
                        cats = [str(c).lower() for c in (val.get("categories") or []) if str(c).strip()]
                        action = str(val.get("action", "block")).lower()
                        threshold = float(val.get("threshold", 0.5))
                        moderated, hits = _moderate_text(str(value), cats or list(GUARDRAIL_PATTERNS.keys()), action, threshold)
                        if hits and moderated is None:
                            reason = f"output_moderation:block:{hits}"
                            decision = "blocked"
                            _record_policy_audit(request, self.api_base, self._headers(), self.verify_ssl,
                                                 agent_id=self.agent_id, model_kind=kind, decision=decision,
                                                 attestation_digest=self._attestation_digest, applied=applied + [reason])
                            return None, applied + [reason]
                        if hits and action == "redact":
                            value = moderated
                            applied.append(f"output_moderation:redact:{hits}")
                        elif hits and action == "flag":
                            applied.append(f"output_moderation:flag:{hits}")
                        else:
                            applied.append("output_moderation=ok")

                # ---------------- VISION ----------------
                if kind == "vision":
                    if ptype == "resolution_cap" and phase == "input":
                        max_res = int(val)
                        # Assume value is a PIL image
                        if hasattr(value, "resize"):
                            w, h = value.size
                            if max(w, h) > max_res:
                                value = value.resize((max_res, max_res))
                                applied.append(f"resolution_cap={max_res}")

                    if ptype == "class_whitelist" and phase == "output":
                        whitelist = [c.strip().lower() for c in str(val).split(",")]
                        if str(value).lower() not in whitelist:
                            reason = f"class_whitelist:block:{whitelist}"
                            decision = "blocked"
                            _record_policy_audit(request, self.api_base, self._headers(), self.verify_ssl,
                                                 agent_id=self.agent_id, model_kind=kind, decision=decision,
                                                 attestation_digest=self._attestation_digest, applied=applied + [reason])
                            return None, applied + [reason]

                    if ptype == "face_redaction" and phase == "output":
                        if val == "enabled":
                            try:
                                from PIL import Image, ImageFilter
                                if isinstance(value, Image.Image):
                                    value = value.filter(ImageFilter.GaussianBlur(radius=8))
                                    applied.append("face_redaction=blurred")
                                elif isinstance(value, dict):
                                    img = value.get("image")
                                    faces = value.get("faces") or value.get("bboxes") or []
                                    if isinstance(img, Image.Image) and faces:
                                        img = img.copy()
                                        for box in faces:
                                            try:
                                                x1, y1, x2, y2 = [int(float(b)) for b in box]
                                                crop = img.crop((x1, y1, x2, y2)).filter(ImageFilter.GaussianBlur(radius=12))
                                                img.paste(crop, (x1, y1))
                                            except Exception:
                                                continue
                                        value["image"] = img
                                        applied.append("face_redaction=blurred_faces")
                                    elif isinstance(img, Image.Image):
                                        value["image"] = img.filter(ImageFilter.GaussianBlur(radius=8))
                                        applied.append("face_redaction=blurred_full")
                            except ImportError:
                                applied.append("face_redaction=skipped_missing_PIL")

                    if ptype == "vision_moderation":
                        sensitive = [c.strip().lower() for c in (val.get("sensitive_classes") or []) if c.strip()]
                        minors_action = (val.get("minors_action") or "blur").lower()
                        action = (val.get("action") or "block").lower()

                        # Try to read detections from structured payloads (if present)
                        detections: List[str] = []
                        bboxes: List[Tuple[int, int, int, int]] = []
                        if isinstance(value, dict):
                            for key in ("detections", "labels", "classes"):
                                arr = value.get(key)
                                if isinstance(arr, (list, tuple)):
                                    detections.extend([str(x).lower() for x in arr])
                            if isinstance(value.get("detected_classes"), dict):
                                detections.extend([str(k).lower() for k in value.get("detected_classes").keys()])
                            if isinstance(value.get("faces"), (list, tuple)):
                                try:
                                    bboxes = [tuple(map(int, f)) for f in value.get("faces")]
                                except Exception:
                                    bboxes = []

                        hits = [d for d in detections if d in sensitive]
                        minor_hit = any("minor" in d or "child" in d or "kid" in d for d in detections)

                        # Apply minors handling first
                        if minor_hit:
                            if minors_action == "block":
                                reason = "vision_moderation:block:minor"
                                decision = "blocked"
                                _record_policy_audit(request, self.api_base, self._headers(), self.verify_ssl,
                                                     agent_id=self.agent_id, model_kind=kind, decision=decision,
                                                     attestation_digest=self._attestation_digest, applied=applied + [reason])
                                return None, applied + [reason]
                            if minors_action == "blur":
                                try:
                                    from PIL import Image, ImageFilter
                                    if isinstance(value, Image.Image):
                                        img = value.copy()
                                        for box in bboxes:
                                            try:
                                                x1, y1, x2, y2 = box
                                                crop = img.crop((x1, y1, x2, y2)).filter(ImageFilter.GaussianBlur(radius=12))
                                                img.paste(crop, (x1, y1))
                                            except Exception:
                                                continue
                                        value = img
                                    elif isinstance(value, dict) and isinstance(value.get("image"), Image.Image):
                                        img = value["image"].copy()
                                        for box in bboxes:
                                            try:
                                                x1, y1, x2, y2 = box
                                                crop = img.crop((x1, y1, x2, y2)).filter(ImageFilter.GaussianBlur(radius=12))
                                                img.paste(crop, (x1, y1))
                                            except Exception:
                                                continue
                                        value["image"] = img
                                    applied.append("vision_moderation:blur_minor")
                                except ImportError:
                                    applied.append("vision_moderation:minor_detected_no_PIL")

                        if hits:
                            if action == "block":
                                reason = f"vision_moderation:block:{hits}"
                                decision = "blocked"
                                _record_policy_audit(request, self.api_base, self._headers(), self.verify_ssl,
                                                     agent_id=self.agent_id, model_kind=kind, decision=decision,
                                                     attestation_digest=self._attestation_digest, applied=applied + [reason])
                                return None, applied + [reason]
                            if action == "flag":
                                applied.append(f"vision_moderation:flag:{hits}")
                            if action == "blur":
                                try:
                                    from PIL import Image, ImageFilter
                                    if isinstance(value, Image.Image):
                                        value = value.filter(ImageFilter.GaussianBlur(radius=8))
                                    elif isinstance(value, dict) and isinstance(value.get("image"), Image.Image):
                                        value["image"] = value["image"].filter(ImageFilter.GaussianBlur(radius=8))
                                    applied.append(f"vision_moderation:blur:{hits}")
                                except ImportError:
                                    applied.append(f"vision_moderation:blur_requested_no_PIL:{hits}")
                        else:
                            applied.append("vision_moderation=ok")

                # ---------------- RL ----------------
                if kind == "rl":
                    if ptype == "reward_bounds" and phase == "output":
                        try:
                            lo, hi = [float(x) for x in val.strip("[]").split(",")]
                            if isinstance(value, (int, float)):
                                value = max(lo, min(hi, float(value)))
                                applied.append(f"reward_bounds=[{lo},{hi}]")
                            elif isinstance(value, list):
                                value = [max(lo, min(hi, float(v))) if isinstance(v, (int, float)) else v for v in value]
                                applied.append(f"reward_bounds_list=[{lo},{hi}]")
                        except Exception:
                            applied.append("reward_bounds=parse_error")

                    if ptype == "exploration_limit" and phase == "input":
                        try:
                            limit = float(val)
                            if isinstance(value, dict):
                                if "exploration_rate" in value:
                                    value["exploration_rate"] = min(float(value["exploration_rate"]), limit)
                                if "epsilon" in value:
                                    value["epsilon"] = min(float(value["epsilon"]), limit)
                                applied.append(f"exploration_limit_dict={limit}")
                            elif isinstance(value, (int, float)):
                                value = max(-limit, min(limit, float(value)))
                                applied.append(f"exploration_limit_scalar={limit}")
                            elif isinstance(value, list):
                                value = [max(-limit, min(limit, float(v))) if isinstance(v, (int, float)) else v for v in value]
                                applied.append(f"exploration_limit_list={limit}")
                        except Exception:
                            applied.append("exploration_limit=parse_error")

                # ---------------- STT ----------------
                if kind == "stt":
                    if ptype == "energy_limit":
                        try:
                            limit_kb = float(val)
                            if isinstance(value, (bytes, bytearray)):
                                size_kb = len(value) / 1024.0
                                if size_kb > limit_kb:
                                    reason = f"energy_limit:block:{size_kb:.1f}kb>{limit_kb}kb"
                                    decision = "blocked"
                                    _record_policy_audit(request, self.api_base, self._headers(), self.verify_ssl,
                                                         agent_id=self.agent_id, model_kind=kind, decision=decision,
                                                         attestation_digest=self._attestation_digest, applied=applied + [reason])
                                    return None, applied + [reason]
                                applied.append(f"energy_limit=ok:{size_kb:.1f}kb/{limit_kb}kb")
                            else:
                                applied.append(f"energy_limit=not_applicable")
                        except Exception:
                            applied.append("energy_limit=parse_error")
                    if ptype == "logging":
                        applied.append(f"logging={val}")

                # ---------------- TTS ----------------
                if kind == "tts":
                    if ptype == "voice_style":
                        applied.append(f"voice_style={val}")  # recorded for telemetry/debug
                    if ptype == "energy_limit":
                        try:
                            limit_kb = float(val)
                            if isinstance(value, (bytes, bytearray)):
                                size_kb = len(value) / 1024.0
                                if size_kb > limit_kb:
                                    reason = f"energy_limit:block:{size_kb:.1f}kb>{limit_kb}kb"
                                    decision = "blocked"
                                    _record_policy_audit(request, self.api_base, self._headers(), self.verify_ssl,
                                                         agent_id=self.agent_id, model_kind=kind, decision=decision,
                                                         attestation_digest=self._attestation_digest, applied=applied + [reason])
                                    return None, applied + [reason]
                                applied.append(f"energy_limit=ok:{size_kb:.1f}kb/{limit_kb}kb")
                            else:
                                applied.append(f"energy_limit=not_applicable")
                        except Exception:
                            applied.append("energy_limit=parse_error")

                # -------- Global safety pass (textual outputs)
                if phase == "output" and isinstance(value, str):
                    flagged, hits = _safety_scan(str(value))
                    if flagged:
                        reason = f"safety:block:{hits}"
                        decision = "blocked"
                        _record_policy_audit(request, self.api_base, self._headers(), self.verify_ssl,
                                             agent_id=self.agent_id, model_kind=kind, decision=decision,
                                             attestation_digest=self._attestation_digest, applied=applied + [reason])
                        return None, applied + [reason]

        _record_policy_audit(request, self.api_base, self._headers(), self.verify_ssl,
                             agent_id=self.agent_id, model_kind=kind, decision=decision,
                             attestation_digest=self._attestation_digest, applied=applied)
        return value, applied

    def _enforce_output_guardrails(
        self,
        runtime: Dict[str, Any],
        value: Any,
        kind: str,
    ) -> Tuple[Optional[Any], List[str]]:
        cfg = runtime.get("config") or {}
        guard = cfg.get("output_guardrails") or {}
        if not guard.get("enabled"):
            return value, []

        categories = guard.get("categories") or []
        categories = [str(c).lower() for c in categories if str(c).strip()]
        if not categories:
            return value, []

        if not isinstance(value, str):
            return value, []

        text = value
        matched: List[str] = []
        for cat in categories:
            patterns = GUARDRAIL_PATTERNS.get(cat, [])
            for pattern in patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    matched.append(cat)
                    break

        if not matched:
            return value, []

        threshold = guard.get("threshold", 0.5)
        try:
            threshold = float(threshold)
        except Exception:
            threshold = 0.5
        threshold = max(0.0, min(1.0, threshold))
        score = len(matched) / max(len(categories), 1)
        if score < threshold:
            return value, []

        action = str(guard.get("action", "block")).lower()
        applied: List[str] = []

        if action == "flag":
            applied.append(f"guardrails:flag:{','.join(matched)}@{score:.2f}")
            return value, applied

        if action == "redact":
            redacted = text
            for cat in matched:
                for pattern in GUARDRAIL_PATTERNS.get(cat, []):
                    redacted = re.sub(pattern, "[REDACTED]", redacted, flags=re.IGNORECASE)
            applied.append(f"guardrails:redact:{','.join(matched)}@{score:.2f}")
            return redacted, applied

        applied.append(f"guardrails:block:{','.join(matched)}@{score:.2f}")
        return None, applied

    def _apply_guardrails_with_logging(
        self,
        runtime: Dict[str, Any],
        value: Any,
        kind: str,
    ) -> Tuple[Optional[Any], List[str]]:
        result, applied = self._enforce_output_guardrails(runtime, value, kind)
        for marker in applied:
            marker_lower = marker.lower()
            if ":block:" in marker_lower:
                logger.warning("⚠️ %s output guardrails blocked response: %s", kind, marker)
            elif ":flag:" in marker_lower:
                logger.warning("⚠️ %s output guardrails flagged response: %s", kind, marker)
            else:
                logger.info("[SDK] %s output guardrails applied: %s", kind, marker)
        return result, applied


    # ---------- get_anchor_mode() Method ----------

    def get_anchor_mode(self) -> Tuple[Optional[str], str]:
        """
        Returns (anchor, mode) for this agent.
          - anchor: "tpm" | "tee" | "dsim" | "hsm" | "none" | None
          - mode:   "secure" | "insecure"
        """
        st_full = self.get_status()
        state = st_full.get("state", {})

        anchor = st_full.get("anchor") or state.get("anchor")
        mode = st_full.get("mode") or ("insecure" if anchor == "none" else "secure")

        return anchor, mode

    def is_active(self) -> bool:
        try:
            st = self.get_status()
            status_str = (st.get("status") or "").lower()
            state = st.get("state", {})
            is_enabled = st.get("enabled", False) or (status_str == "enabled") or state.get("enabled", False)
            is_revoked = st.get("revoked", False) or state.get("revoked", False)
            is_personalized = st.get("personalized", False) or state.get("personalized", False)
            return is_enabled and is_personalized and not is_revoked
        except Exception as e:
            logger.warning("is_active() check failed: %s", e)

            return False

    def models(self) -> List[Dict[str, Any]]:
        return request(
            "GET",
            self.api_base,
            f"/agents/{self.agent_id}/models",
            headers=self._headers(),
            verify_ssl=self.verify_ssl,
        )

    def certificates(self) -> List[Dict[str, Any]]:
        try:
            return request(
                "GET",
                self.api_base,
                "/certificates",
                headers=self._headers(),
                params={"agent_id": self.agent_id},
                verify_ssl=self.verify_ssl,
            )
        except Exception:
            return []

    # ---------- actions ----------
    def set_status(self, status: str) -> Dict[str, Any]:
        return request(
            "POST",
            self.api_base,
            f"/agents/{self.agent_id}/status",
            headers=self._headers(),
            json_body={"status": status},
            verify_ssl=self.verify_ssl,
        )

    def revoke_certificates(self, reason: str = "unspecified") -> Dict[str, Any]:
        out = {"ok": True, "revoked": 0, "errors": []}
        for c in self.certificates():
            serial = c.get("serial") or c.get("Serial")
            if not serial:
                continue
            try:
                r = request(
                    "POST",
                    self.api_base,
                    f"/certificates/{serial}/revoke",
                    headers=self._headers(),
                    json_body={"reason": reason},
                    verify_ssl=self.verify_ssl,
                )
                if r.get("ok"):
                    out["revoked"] += 1
            except Exception as e:
                out["ok"] = False
                out["errors"].append(str(e))
        return out

    # ---------- personalization (challenge-aware) ----------
    def personalize(self, anchor: str, evidence: Optional[dict] = None) -> Dict[str, Any]:
        anchor = (anchor or "").strip().lower()
        if anchor not in ("tpm", "tee", "dsim", "hsm", "none"):
            raise ValueError("anchor must be one of: 'tpm', 'tee', 'dsim', 'hsm', 'none'")

        chal = request(
            "POST",
            self.api_base,
            f"/agents/{self.agent_id}/challenge",
            headers=self._headers(),
            verify_ssl=self.verify_ssl,
        )
        nonce_b64 = chal.get("nonce_b64")
        logger.debug("[TA] Got challenge nonce %s", nonce_b64)

        # --- evidence collection
        if anchor == "tpm":
            ev = self._collect_tpm_evidence(nonce_b64) if evidence is None else evidence
        elif anchor == "tee":
            ev = self._collect_tee_evidence(nonce_b64) if evidence is None else evidence
        elif anchor == "dsim":
            ev = self._collect_dsim_evidence(nonce_b64) if evidence is None else evidence
        elif anchor == "hsm":
            ev = self._collect_hsm_evidence(nonce_b64) if evidence is None else evidence
        else:  # "none"
            ev = {"dev": True, "nonce_b64": nonce_b64, "note": "No anchor (software only)"}

        # --- Add CSR if using EJBCA
        if self._using_ejbca_backend():
            csr_pem = self._generate_csr(self.agent_id)
            ev["csr_pem"] = csr_pem

        body = {"anchor": {"type": anchor, "evidence": ev or {}}, "installed": True}
        resp = request(
            "POST",
            self.api_base,
            f"/agents/{self.agent_id}/personalize",
            headers=self._headers(),
            json_body=body,
            verify_ssl=self.verify_ssl,
        )

        inst = resp.get("agent")
        if inst:
            logger.debug(
                "[TA] Personalize response agent type=%s id=%s did=%s",
                inst.get("type"),
                inst.get("public_id") or inst.get("_id"),
                inst.get("did"),
            )
        if inst and inst.get("type") == "INSTANCE":
            new_id = (
                inst.get("did")
                or inst.get("public_id")
                or (str(inst.get("_id")) if inst.get("_id") is not None else None)
                or resp.get("agent_id")
            )
            if new_id:
                old_id = self.agent_id
                self.agent_id = new_id
                (self.storage_dir / "agent_id").write_text(self.agent_id)
                self._invalidate_runtime_cache()
                self._record_identity(old_id)
                self._record_identity(new_id)
                logger.debug("[TA] ✅ Switched to Agent Instance %s", self.agent_id)
            else:
                logger.warning("[TA] ⚠️ No valid id found in personalize response")

        return resp



    # ---- TPM evidence (Linux + tpm2-tools) -----------------------------------
    def _collect_tpm_evidence(self, nonce_b64: str) -> dict:
        """
        Collect TPM2_Quote evidence using tpm2-tools.

        Returns:
          - ak_pub_der_b64 : base64(DER SubjectPublicKeyInfo)
          - quote_b64      : base64(TPMS_ATTEST bytes)
          - sig_b64        : base64(ECDSA signature in DER format)
          - pcrs           : optional { bank, selection[], values{idx:hex} }
        """
        import shutil, subprocess, binascii
        from pathlib import Path
        from tpm2_pytss import TPMT_SIGNATURE
        from cryptography.hazmat.primitives.asymmetric.utils import encode_dss_signature

        force_real = (os.getenv("PERSONALIZE_ANCHOR") in ("tpm", "tee"))
        evidence = {"nonce_b64": nonce_b64}
        if _dev_allow_insecure() and not force_real:
            evidence.update(
                {
                    "dev": True,
                    "ak_pub_der_b64": _b64(b"dev-ak-pub"),
                    "quote_b64": _b64(b"dev-quote"),
                    "sig_b64": _b64(b"dev-sig"),
                }
            )
            return evidence

        # --- helper to run tools & always log stderr on failure ---
        def run(cmd, **kw):
            kw.setdefault("check", True)
            r = subprocess.run(cmd, capture_output=True, text=True, **kw)
            if r.returncode != 0:
                raise RuntimeError(
                    f"{' '.join(cmd)} failed:\n"
                    f"STDERR: {r.stderr or '<empty>'}\n"
                    f"STDOUT: {r.stdout or '<empty>'}"
                )
            return r

        # ensure tools available
        for tool in ("tpm2_createek", "tpm2_createak", "tpm2_readpublic", "tpm2_quote", "tpm2_pcrread"):
            if not shutil.which(tool):
                raise RuntimeError(f"{tool} not found on PATH (install tpm2-tools)")

        home = Path(os.path.expanduser("~"))
        tdir = home / ".ephapsys" / "tpm"
        tdir.mkdir(parents=True, exist_ok=True)
        ek_ctx, ak_ctx = tdir / "ek.ctx", tdir / "ak.ctx"
        ak_pub_der, quote_bin, sig_file = tdir / "ak_pub.der", tdir / "quote.bin", tdir / "sig.bin"

        # nonce as bare hex
        nonce_hex = base64.b64decode(nonce_b64).hex()

        # 1) EK
        if not ek_ctx.exists():
            run(["tpm2_createek", "-G", "rsa", "-c", str(ek_ctx)])

        # 2) AK under EK
        def create_ak():
            try:
                ak_ctx.unlink()
            except Exception:
                pass
            run(
                [
                    "tpm2_createak",
                    "-C",
                    str(ek_ctx),
                    "-G",
                    "ecc",
                    "-g",
                    "sha256",
                    "-s",
                    "ecdsa",
                    "-c",
                    str(ak_ctx),
                    "-u",
                    str(tdir / "ak.pub"),
                ]
            )

        if not ak_ctx.exists():
            create_ak()

        # 3) AK DER SPKI
        run(["tpm2_readpublic", "-c", str(ak_ctx), "-o", str(ak_pub_der), "-f", "der"])
        ak_b64 = base64.b64encode(ak_pub_der.read_bytes()).decode("ascii")

        # 4) Quote with PCR fallback
        def do_quote_once() -> str:
            override = os.getenv("TPM_PCR_SPEC")
            pcr_candidates = [override] if override else ["sha256:0", "sha1:0", "sha256:0,1,7"]
            last_err = None
            for spec in pcr_candidates:
                if not spec:
                    continue
                try:
                    run(
                        [
                            "tpm2_quote",
                            "-c",
                            str(ak_ctx),
                            "-l",
                            spec,
                            "-q",
                            nonce_hex,
                            "-m",
                            str(quote_bin),
                            "-s",
                            str(sig_file),
                        ]
                    )
                    return spec
                except RuntimeError as e:
                    last_err = e
            raise RuntimeError(f"tpm2_quote failed for {pcr_candidates}:\n{last_err}")

        try:
            _ = do_quote_once()
        except RuntimeError:
            create_ak()
            run(["tpm2_readpublic", "-c", str(ak_ctx), "-o", str(ak_pub_der), "-f", "der"])
            ak_b64 = base64.b64encode(ak_pub_der.read_bytes()).decode("ascii")
            _ = do_quote_once()

        quote_b64 = base64.b64encode((tdir / "quote.bin").read_bytes()).decode("ascii")

        # 5) Convert TPMT_SIGNATURE → DER ECDSA
        sig_tss = (tdir / "sig.bin").read_bytes()
        tsig = TPMT_SIGNATURE.unmarshal(sig_tss)
        if isinstance(tsig, tuple):  # v1 API returns (obj, rest)
            tsig = tsig[0]

        def _param_to_int(param):
            if hasattr(param, "buffer"):  # TPM2B_ECC_PARAMETER
                return int.from_bytes(bytes(param.buffer), "big")
            try:
                return int(param)
            except Exception:
                return int.from_bytes(bytes(param), "big")

        r = _param_to_int(tsig.signature.ecdsa.signatureR)
        s = _param_to_int(tsig.signature.ecdsa.signatureS)

        from cryptography.hazmat.primitives.asymmetric.utils import encode_dss_signature
        der_sig = encode_dss_signature(r, s)
        sig_b64 = base64.b64encode(der_sig).decode()

        evidence.update({"ak_pub_der_b64": ak_b64, "quote_b64": quote_b64, "sig_b64": sig_b64})


        # --- SIE KEM keypair (P-256) for HPKE unwrap ---
        from cryptography.hazmat.primitives import serialization
        from cryptography.hazmat.primitives.asymmetric import ec

        priv_p, pub_p = self._kem_key_paths()
        if not priv_p.exists() or not pub_p.exists():
            prv = ec.generate_private_key(ec.SECP256R1())
            pub = prv.public_key()
            priv_p.write_bytes(
                prv.private_bytes(
                    encoding=serialization.Encoding.PEM,
                    format=serialization.PrivateFormat.PKCS8,
                    encryption_algorithm=serialization.NoEncryption(),
                )
            )
            pub_p.write_bytes(
                pub.public_bytes(
                    encoding=serialization.Encoding.PEM,
                    format=serialization.PublicFormat.SubjectPublicKeyInfo,
                )
            )
        evidence["kem_pub_pem"] = pub_p.read_text()


        return evidence

    # ---- TEE evidence --------------------------------------------------------
    def _collect_tee_evidence(self, nonce_b64: str) -> dict:
        system = platform.system().lower()
        logger.debug("[TEE] Collecting evidence for system=%s", system)

        if system == "windows":
            return self._collect_windows_tpm_evidence(nonce_b64)
        elif system == "linux":
            if self._has_sgx():
                return self._collect_sgx_evidence(nonce_b64)
            elif self._has_snp():
                return self._collect_snp_evidence(nonce_b64)
            else:
                raise RuntimeError("[TEE] No SGX/TDX or SNP device detected")
        elif system == "darwin":
            return self._collect_secure_enclave_evidence(nonce_b64)
        else:
            raise RuntimeError(f"[TEE] Unsupported OS: {system}")

    # ---- TPM evidence on Windows (stub) --------------------------------------
    def _collect_windows_tpm_evidence(self, nonce_b64: str) -> dict:
        logger.debug("[TEE][Windows] Using TPM evidence")
        # NOTE: This path requires proper tpm2-pytss ESAPI usage. Stub for now.
        raise NotImplementedError("Windows TPM evidence collection not implemented in SDK")

    def _has_sgx(self) -> bool:
        return os.path.exists("/dev/sgx_enclave") or os.path.exists("/dev/tdx-guest")

    def _collect_sgx_evidence(self, nonce_b64: str) -> dict:
        """
        Generate enclave keypair and get DCAP quote binding pubkey + nonce.
        Requires Intel SGX DCAP SDK.
        """
        logger.debug("[TEE][Linux] Using SGX DCAP evidence")
        nonce = base64.b64decode(nonce_b64)

        # Generate enclave keypair (this must happen inside enclave in real deployment)
        priv = ec.generate_private_key(ec.SECP256R1())
        pub = priv.public_key()
        pub_pem = pub.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        ).decode()

        # Hash(nonce || pubkey) → REPORTDATA
        hasher = hashes.Hash(hashes.SHA256())
        hasher.update(nonce)
        hasher.update(
            pub.public_bytes(
                encoding=serialization.Encoding.DER,
                format=serialization.PublicFormat.SubjectPublicKeyInfo,
            )
        )
        report_data = hasher.finalize()

        try:
            quote = subprocess.check_output(
                ["/usr/bin/sgx_get_quote", base64.b64encode(report_data).decode()],
                timeout=5,
            )
        except Exception as e:
            logger.error("[TEE][Linux] SGX failed: %s", e)
            raise

        return {
            "sgx_tdx_quote_b64": base64.b64encode(quote).decode(),
            "pubkey_pem": pub_pem,
            "nonce_b64": nonce_b64,
        }

    def _has_snp(self) -> bool:
        return os.path.exists("/sys/kernel/security/sev/guest/attestation_report")

    def _collect_snp_evidence(self, nonce_b64: str) -> dict:
        """
        Generate guest keypair, bind to SNP report via report_data.
        Requires AMD SNP guest driver and access to attestation_report.
        """
        logger.debug("[TEE][Linux] Using SNP evidence")
        nonce = base64.b64decode(nonce_b64)

        # Generate guest keypair
        priv = ec.generate_private_key(ec.SECP256R1())
        pub = priv.public_key()
        pub_pem = pub.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        ).decode()

        # Hash(nonce || pubkey) for report_data
        hasher = hashes.Hash(hashes.SHA256())
        hasher.update(nonce)
        hasher.update(
            pub.public_bytes(
                encoding=serialization.Encoding.DER,
                format=serialization.PublicFormat.SubjectPublicKeyInfo,
            )
        )
        report_data = hasher.finalize()

        try:
            # Normally you'd pass report_data to /dev/sev for attestation
            with open("/sys/kernel/security/sev/guest/attestation_report", "rb") as f:
                report = f.read()

            vcek_chain_pem = "-----BEGIN CERTIFICATE-----\n...AMD VCEK cert...\n-----END CERTIFICATE-----"
            return {
                "sev_snp_report_b64": base64.b64encode(report).decode(),
                "vcek_chain_pem": vcek_chain_pem,
                "pubkey_pem": pub_pem,
                "nonce_b64": nonce_b64,
            }
        except Exception as e:
            logger.error("[TEE][Linux] SNP failed: %s", e)
            raise

    def _collect_secure_enclave_evidence(self, nonce_b64: str) -> dict:
        logger.debug("[TEE][macOS] Using Secure Enclave evidence")
        nonce = base64.b64decode(nonce_b64)

        # ---- Dev/test fallback ----
        if _dev_allow_insecure():
            priv = ec.generate_private_key(ec.SECP256R1())
            pub = priv.public_key()

            pub_pem = pub.public_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PublicFormat.SubjectPublicKeyInfo,
            ).decode()

            sig = priv.sign(nonce, ec.ECDSA(hashes.SHA256()))

            # Build a self-signed cert so backend can parse the chain shape
            subject = issuer = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "Ephapsys Dev SE")])
            cert = (
                x509.CertificateBuilder()
                .subject_name(subject)
                .issuer_name(issuer)
                .public_key(pub)
                .serial_number(x509.random_serial_number())
                .not_valid_before(x509.datetime.datetime.utcnow())
                .not_valid_after(x509.datetime.datetime.utcnow() + x509.datetime.timedelta(days=30))
                .sign(priv, hashes.SHA256())
            )
            cert_pem = cert.public_bytes(serialization.Encoding.PEM).decode()

            return {
                "apple_secure_enclave_pubkey_pem": pub_pem,
                "apple_secure_enclave_cert_chain": cert_pem,
                "sig_b64": base64.b64encode(sig).decode(),
                "nonce_b64": nonce_b64,
            }

        # Production SE attestation would require PyObjC and proper entitlements.
        raise NotImplementedError("Secure Enclave attestation not wired for production in SDK")

    # ---- dSIM/eUICC evidence -------------------------------------------------
    def _collect_dsim_evidence(self, nonce_b64: str) -> dict:
        evidence: dict = {"nonce_b64": nonce_b64}
        if _dev_allow_insecure():
            evidence.update(
                {
                    "dev": True,
                    "eid": "EID-DEV-0000",
                    "iccid_last4": "9999",
                    "sig_b64": _b64(b"dev-dsim-sig"),
                    "key_cert_pem": "-----BEGIN CERTIFICATE-----\nDEV\n-----END CERTIFICATE-----\n",
                }
            )
        return evidence

    def _collect_hsm_evidence(self, nonce_b64: str) -> dict:
        """
        Collect evidence from an external Hardware Security Module.

        Supported workflows:
          1. Cloud KMS / Cloud HSM (GCP) via google-cloud-kms when HSM_KMS_KEY is set.
          2. Helper executable via HSM_HELPER (PKCS#11, vendor SDK, etc.).
          3. Static JSON via HSM_EVIDENCE_PATH (mainly for tests).
        """

        helper = os.getenv("HSM_HELPER")
        evidence_path = os.getenv("HSM_EVIDENCE_PATH")
        kms_key = os.getenv("HSM_KMS_KEY")
        kms_endpoint = os.getenv("HSM_KMS_ENDPOINT")
        kms_credentials = os.getenv("HSM_KMS_CREDENTIALS")
        slot_hint = os.getenv("HSM_SLOT")
        key_label_hint = os.getenv("HSM_KEY_LABEL")
        evidence: Optional[dict] = None

        if helper:
            cmd = shlex.split(helper)
            try:
                proc = subprocess.run(
                    cmd + [nonce_b64],
                    check=True,
                    capture_output=True,
                    text=True,
                )
            except subprocess.CalledProcessError as exc:
                logger.error("[HSM] Helper command failed: %s", exc)
                raise RuntimeError(f"HSM helper failed (exit {exc.returncode})") from exc
            out = proc.stdout.strip()
            if not out:
                raise RuntimeError("HSM helper returned empty output")
            try:
                evidence = json.loads(out)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"HSM helper did not return valid JSON: {exc}") from exc
        elif kms_key:
            evidence = self._collect_hsm_evidence_from_kms(
                nonce_b64,
                kms_key,
                endpoint=kms_endpoint,
                credentials_path=kms_credentials,
                slot_hint=slot_hint,
                key_label_hint=key_label_hint,
            )
        elif evidence_path:
            path = pathlib.Path(evidence_path).expanduser()
            if not path.exists():
                raise RuntimeError(f"HSM_EVIDENCE_PATH not found: {path}")
            with open(path, "r", encoding="utf-8") as f:
                try:
                    evidence = json.load(f)
                except json.JSONDecodeError as exc:
                    raise RuntimeError(f"Invalid JSON in {path}: {exc}") from exc

        if evidence is None:
            if _dev_allow_insecure():
                logger.warning("[HSM] Using insecure dev stub evidence")
                return {
                    "dev": True,
                    "nonce_b64": nonce_b64,
                    "sig_b64": _b64(b"dev-hsm-sig"),
                    "note": "Insecure HSM stub evidence",
                }
            raise RuntimeError(
                "HSM personalization requires either 'evidence' argument, "
                "HSM_HELPER, HSM_KMS_KEY, or HSM_EVIDENCE_PATH"
            )

        if not isinstance(evidence, dict):
            raise ValueError("HSM evidence must be a JSON object")

        evidence.setdefault("nonce_b64", nonce_b64)

        missing = [k for k in ("sig_b64",) if k not in evidence]
        if missing:
            raise ValueError(f"HSM evidence missing required field(s): {', '.join(missing)}")

        return evidence

    def _collect_hsm_evidence_from_kms(
        self,
        nonce_b64: str,
        kms_key: str,
        *,
        endpoint: Optional[str],
        credentials_path: Optional[str],
        slot_hint: Optional[str],
        key_label_hint: Optional[str],
    ) -> dict:
        """
        Collect evidence using Google Cloud KMS / Cloud HSM.

        Requires google-cloud-kms (already part of the SDK's dependencies).
        """
        try:
            from google.cloud import kms_v1
            from google.oauth2 import service_account
        except ImportError as exc:
            raise RuntimeError(
                "google-cloud-kms is required for HSM anchors configured via HSM_KMS_KEY"
            ) from exc

        client_kwargs: Dict[str, Any] = {}
        if endpoint:
            client_kwargs["client_options"] = {"api_endpoint": endpoint}
        if credentials_path:
            cred_path = os.path.expanduser(credentials_path)
            if not os.path.exists(cred_path):
                raise RuntimeError(f"HSM_KMS_CREDENTIALS not found: {cred_path}")
            client_kwargs["credentials"] = service_account.Credentials.from_service_account_file(cred_path)

        client = kms_v1.KeyManagementServiceClient(**client_kwargs)

        key_name = self._normalize_kms_key_name(client, kms_key)
        nonce = base64.b64decode(nonce_b64)
        digest = hashlib.sha256(nonce).digest()

        digest_msg = kms_v1.Digest(sha256=digest)
        resp = client.asymmetric_sign(request={"name": key_name, "digest": digest_msg})
        sig_b64 = base64.b64encode(resp.signature).decode("ascii")

        pub = client.get_public_key(request={"name": key_name})
        pubkey_pem = pub.pem

        version = client.get_crypto_key_version(request={"name": key_name})
        components = self._parse_kms_resource_name(key_name)

        slot_value = slot_hint or (
            f"{components.get('project')}/{components.get('location')}/{components.get('key_ring')}"
            if components
            else None
        )
        key_label_value = key_label_hint or (
            f"{components.get('crypto_key')}@{components.get('version')}"
            if components
            else None
        )

        evidence: Dict[str, Any] = {
            "nonce_b64": nonce_b64,
            "sig_b64": sig_b64,
            "pubkey_pem": pubkey_pem,
            "provider": "gcp-kms",
            "kms_key_name": key_name,
        }
        if slot_value:
            evidence["slot"] = slot_value
        if key_label_value:
            evidence["key_label"] = key_label_value
        if version.algorithm:
            evidence["kms_algorithm"] = version.algorithm.name
        if version.protection_level:
            evidence["kms_protection_level"] = version.protection_level.name

        return evidence

    @staticmethod
    def _parse_kms_resource_name(name: str) -> Dict[str, str]:
        pattern = (
            r"^projects/(?P<project>[^/]+)/locations/(?P<location>[^/]+)/"
            r"keyRings/(?P<key_ring>[^/]+)/cryptoKeys/(?P<crypto_key>[^/]+)/"
            r"(?:cryptoKeyVersions/(?P<version>[^/]+))?$"
        )
        match = re.match(pattern, name)
        return match.groupdict() if match else {}

    def _normalize_kms_key_name(self, client: "kms_v1.KeyManagementServiceClient", key_name: str) -> str:  # type: ignore[name-defined]
        if "/cryptoKeyVersions/" in key_name:
            return key_name

        crypto_key = client.get_crypto_key(request={"name": key_name})
        if not crypto_key.primary or not crypto_key.primary.name:
            raise RuntimeError(
                "KMS key has no primary cryptoKeyVersion; specify HSM_KMS_KEY with /cryptoKeyVersions/<n>"
            )
        return crypto_key.primary.name

    # ---------- runtime preparation ----------
    def _cache_dir(self) -> pathlib.Path:
        return _mkdir(self.storage_dir / "cache" / self.agent_id)

    def _artifact_dst(self, model_id: str, filename: str) -> pathlib.Path:
        """
        Local path for a cached artifact.
        Example: .ephapsys_state/cache/<agent_id>/<model_id>/<filename>
        """
        return _mkdir(self._cache_dir() / model_id) / filename

    def _download(self, src: str, dst: pathlib.Path) -> str:
        """
        Download or copy an artifact to the local cache.
        Returns the final path.
        """
        if not src:
            raise RuntimeError("Empty artifact URL/path")

        _mkdir(dst.parent)
        if _is_http(src):
            import urllib.request
            with urllib.request.urlopen(src) as r, open(dst, "wb") as f:
                shutil.copyfileobj(r, f)
        else:
            if not os.path.exists(src):
                raise RuntimeError(f"Artifact source not found: {src}")
            if str(dst) != src:
                shutil.copyfile(src, dst)
        return str(dst)

    def _download_all_artifacts(self, model_id: str, artifacts: Dict[str, Dict[str, str]]) -> Dict[str, str]:
        """
        Download all artifacts for a given model entry into cache.
        Returns {name: local_path}.
        """
        local_dir = _mkdir(self._cache_dir() / model_id / "artifacts")
        local_paths: Dict[str, str] = {}

        failures: List[Tuple[str, str, Exception]] = []

        for name, meta in artifacts.items():
            url = meta.get("url") or meta.get("storage_path")
            if not url:
                logger.warning("[SDK][Artifacts] ⚠️ No URL for %s:%s", model_id, name)
                continue
            try:
                dst = local_dir / os.path.basename(url)
                logger.debug("[SDK][Artifacts] → downloading %s:%s from %s", model_id, name, url)
                resolved = self._download(url, dst)
                local_paths[name] = resolved
                logger.debug("[SDK][Artifacts] ✅ Downloaded %s:%s → %s", model_id, name, resolved)
            except Exception as e:
                if name in getattr(self, "_optional_artifacts", set()):
                    logger.warning("[SDK][Artifacts] optional %s:%s unavailable (%s); skipping", model_id, name, e)
                    continue
                logger.error("[SDK][Artifacts] ❌ Failed %s:%s from %s: %s", model_id, name, url, e)
                failures.append((name, url, e))

        if failures:
            first_name, first_url, first_exc = failures[0]
            raise RuntimeError(
                f"Required artifact download failed ({model_id}:{first_name} ← {first_url}): {first_exc}"
            ) from first_exc

        return local_paths


    def _fetch_manifest(self, retries: int = 5, delay: float = 1.0) -> Optional[Dict[str, Any]]:
        """Fetch manifest with retries and template/legacy ID fallbacks."""

        def _candidate_ids() -> List[Tuple[str, bool]]:
            candidates: List[Tuple[str, bool]] = []
            template_ids: Set[str] = set()

            if self._initial_agent_id:
                template_ids.add(str(self._initial_agent_id))

            try:
                status_doc = self.get_status()
                agent_doc = status_doc.get("agent") or {}
                tpl_id = agent_doc.get("template_id") or agent_doc.get("template")
                if tpl_id:
                    template_ids.add(str(tpl_id))
            except Exception as e:
                logger.debug("[SDK] manifest fallback status fetch failed: %s", e)

            seen: Set[str] = set()
            for aid in getattr(self, "_identity_history", []):
                if not aid or aid in seen:
                    continue
                allow = aid in template_ids
                candidates.append((aid, allow))
                seen.add(aid)

            for tpl in template_ids:
                if tpl not in seen:
                    candidates.append((tpl, True))
            return candidates

        last_error: Optional[Exception] = None

        for aid, allow_uncert in _candidate_ids():
            for attempt in range(1, max(1, retries) + 1):
                try:
                    logger.debug("[SDK] Fetching manifest for agent %s (attempt %d/%d)", aid, attempt, retries)
                    path = f"/agents/{aid}/manifest/download"
                    if allow_uncert:
                        path = f"{path}?allow_uncertified=true"
                    man = request(
                        "GET",
                        self.api_base,
                        path,
                        headers=self._headers(),
                        verify_ssl=self.verify_ssl,
                    )
                    if isinstance(man, dict) and "models" in man:
                        for idx, model in enumerate(man["models"]):
                            arts = model.get("artifacts", {})
                            logger.debug(
                                "[SDK]   - model[%d] id=%s artifacts=%s",
                                idx,
                                model.get("id"),
                                list(arts.keys()),
                            )
                        return man
                    logger.warning("[SDK] ⚠️ Manifest missing 'models' key for agent %s: %s", aid, man)
                except Exception as e:
                    last_error = e
                    logger.debug("[SDK] Manifest fetch error for %s: %s", aid, e)
                    if attempt < retries:
                        time.sleep(delay)
            # If we reach here, try next candidate ID (e.g., template)

        if last_error:
            print(f"[SDK] ❌ Manifest download failed: {last_error}")
        return None



     # ---- Backend helpers ----
    def _using_ejbca_backend(self) -> bool:
        """
        Return True if PKI backend is set to 'ejbca'.
        Used to decide whether to include CSR during personalize().
        """
        return os.getenv("PKI_BACKEND", "").lower() == "ejbca"



    def _generate_csr(self, agent_id: str) -> str:
        """
        Generate a CSR PEM for this agent (EC P-256 keypair).
        NOTE: In production, replace with TPM/TEE/SE anchored key generation.
        """
        key = ec.generate_private_key(ec.SECP256R1())
        # Optionally persist the key securely; skipped for now.
        subject = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, f"agent:{agent_id}")])
        csr = x509.CertificateSigningRequestBuilder().subject_name(subject).sign(key, hashes.SHA256())
        return csr.public_bytes(serialization.Encoding.PEM).decode("utf-8")


    # def _kem_key_paths(self):
    #     keystore = self._cache_dir() / "kem"
    #     _mkdir(keystore)
    #     return keystore / "kem_priv.pem", keystore / "kem_pub.pem"

    # def _load_kem_priv(self) -> str:
    #     priv_p, _ = self._kem_key_paths()
    #     return priv_p.read_text()


    def _kem_key_paths(self):
        # Stable device-wide keystore (survives DID/agent_id switch)
        keystore = _mkdir(self.storage_dir / "kem")
        return keystore / "kem_priv.pem", keystore / "kem_pub.pem"

    def _load_kem_priv(self) -> str:
        priv_p, pub_p = self._kem_key_paths()
        if priv_p.exists():
            return priv_p.read_text()

        # ---- Migration: look for old keys under any agent-scoped cache
        # and for alternate filenames (key_priv.pem/key_pub.pem), then move here.
        cache_root = self.storage_dir / "cache"
        try:
            for agent_dir in (cache_root.glob("*/kem")):
                # Prefer kem_priv.pem but accept key_priv.pem
                candidates = [
                    agent_dir / "kem_priv.pem",
                    agent_dir / "key_priv.pem",
                ]
                for cand in candidates:
                    if cand.exists():
                        # match pub, too (best effort)
                        pub_old = agent_dir / ("kem_pub.pem" if cand.name == "kem_priv.pem" else "key_pub.pem")
                        # ensure keystore exists
                        _mkdir(priv_p.parent)
                        # move/rename
                        shutil.move(str(cand), str(priv_p))
                        if pub_old.exists():
                            shutil.move(str(pub_old), str(pub_p))
                        return priv_p.read_text()
        except Exception:
            pass

        # If we get here, the key hasn't been generated yet; let evidence path create it.
        raise FileNotFoundError(f"KEM private key not found at {priv_p}")

    def prepare_runtime(self, force: bool = False) -> Dict[str, Dict[str, Any]]:
        """
        Prepare runtime for all models (cached per agent instance):
          - fetch manifest from backend
          - download artifacts into cache
          - register each model by kind (one per kind)
        Returns: {kind: {"model_path": <dir>, "artifacts": {...}}}
        pass force=True to refresh the cached runtimes.
        """
        runtimes = self._ensure_runtime(force=force)
        return {kind: dict(paths) for kind, paths in runtimes.items()}

    def _ensure_runtime(self, force: bool = False) -> Dict[str, Dict[str, Any]]:
        """Populate runtime cache if missing or refresh requested."""
        if force or self._runtime_cache is None:
            runtimes = self._build_runtime()
            self._runtime_cache = runtimes
        return self._runtime_cache

    def _invalidate_runtime_cache(self) -> None:
        """Clear cached runtimes (e.g., after agent_id switches)."""
        self._runtime_cache = None

    def _record_identity(self, agent_id: Optional[str]) -> None:
        """Track recently used agent identifiers for manifest fallback."""
        if not agent_id:
            return
        cleaned = str(agent_id)
        self._identity_history = [cleaned] + [aid for aid in self._identity_history if aid != cleaned]

    @contextmanager
    def _suppress_transformers_warnings(self):
        """Temporarily silence transformers logging/warnings (e.g., Gemma eager hint)."""
        try:
            from transformers.utils import logging as hf_logging
            prev_level = hf_logging.get_verbosity()
            hf_logging.set_verbosity_error()
        except Exception:
            hf_logging = None
            prev_level = None
        buffer = io.StringIO()
        with warnings.catch_warnings(), redirect_stdout(buffer), redirect_stderr(buffer):
            warnings.simplefilter("ignore")
            try:
                yield
            finally:
                if hf_logging is not None and prev_level is not None:
                    hf_logging.set_verbosity(prev_level)
                leftover = buffer.getvalue().strip()
                if leftover:
                    logger.debug("[SDK][Transformers] %s", leftover)

    def _adapt_state_dir(self) -> pathlib.Path:
        agent_safe = self.agent_id.replace(":", "_")
        path = self.storage_dir / "adapt" / agent_safe
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _adapt_state_path(self, kind: str) -> pathlib.Path:
        return self._adapt_state_dir() / f"{kind}.json"

    def _load_adapt_state(self, kind: str) -> Dict[str, Any]:
        try:
            path = self._adapt_state_path(kind)
            if path.exists():
                return json.loads(path.read_text())
        except Exception as exc:
            logger.debug("[SDK][Adapt] Failed to load state for %s: %s", kind, exc)
        return {"lambda_scale": 1.0}

    def _save_adapt_state(self, kind: str, state: Dict[str, Any]) -> None:
        try:
            path = self._adapt_state_path(kind)
            path.write_text(json.dumps(state, indent=2))
        except Exception as exc:
            logger.warning("[SDK][Adapt] Failed to persist state for %s: %s", kind, exc)

    def _build_runtime(self) -> Dict[str, Dict[str, Any]]:
        manifest = self._fetch_manifest()
        if not (manifest and isinstance(manifest.get("models"), list) and manifest["models"]):
            raise RuntimeError("Manifest not available or empty")

        runtimes: Dict[str, Dict[str, Any]] = {}

        for entry in manifest["models"]:
            mid = entry.get("id") or "model0"
            kind = entry.get("kind") or "unknown"

            # ✅ enforce one model per kind
            if kind in runtimes:
                raise RuntimeError(f"Duplicate model kind '{kind}' in manifest")

            arts = entry.get("artifact_urls", {})
            model_dir = self._cache_dir() / mid
            os.makedirs(model_dir, exist_ok=True)

            local_paths: Dict[str, Any] = {}
            for name, meta in arts.items():
                url = meta.get("url") or meta.get("storage_path")
                sha = (meta.get("sha256") or "").removeprefix("sha256:")
                if not url:
                    continue
                dst = model_dir / os.path.basename(url)
                if not dst.exists():
                    logger.debug("[SDK] Downloading %s:%s → %s", mid, name, dst)
                    self._download(url, dst)
                if sha:
                    calc = sha256_file(str(dst))
                    if calc.lower() != sha.lower():
                        raise RuntimeError(f"Digest mismatch for {name} (got {calc}, expected {sha})")
                local_paths[name] = str(dst)

            # ✅ Handle ECM securely (via SIEManager)
            if "cipher_ecm_uri" in entry:
                sie = SIEManager(
                    base_url=self.api_base,
                    agent_id_or_did=self.agent_id,
                    state_dir=str(model_dir),
                    api_key=self.api_key,
                    verify_ssl=self.verify_ssl,
                    privkey_loader=lambda: self._load_kem_priv(),
                )
                try:
                    ecm_bytes = sie.ensure_ecm_cached_and_get_bytes(entry)
                    local_paths["ecm_bytes"] = ecm_bytes
                    logger.debug("[SDK][ECM] Secure ECM loaded for %s (%s)", kind, mid)

                except Exception as e:
                    print(f"[SDK][ECM] ⚠️ Secure ECM load failed for {kind}: {e}")
                    logger.warning("[SDK][ECM] ⚠️ Secure ECM load failed for %s: %s", kind, e)
            elif entry.get("ecm_uri"):
                ecm_url = entry["ecm_uri"]
                dst = model_dir / "ecm.pt"
                try:
                    logger.debug("[SDK][ECM] downloading plaintext ECM for %s (%s) from %s", kind, mid, ecm_url)
                    self._download(ecm_url, dst)
                    local_paths["ecm_path"] = str(dst)
                    logger.debug("[SDK][ECM] Secure ECM loaded for %s (%s)", kind, mid)
                except Exception as e:
                    raise RuntimeError(
                        f"ECM download failed for model {mid} ({kind}) from {ecm_url}: {e}"
                    ) from e

            # ✅ register model runtime
            local_paths["model_path"] = str(model_dir)
            local_paths["kind"] = kind
            if entry.get("config"):
                cfg = dict(entry["config"] or {})
            else:
                cfg = {}

            guard_cfg = cfg.get("output_guardrails")
            if guard_cfg is None:
                cfg["output_guardrails"] = {
                    **DEFAULT_OUTPUT_GUARDRAILS,
                    "categories": list(DEFAULT_OUTPUT_GUARDRAILS.get("categories", [])),
                }
            else:
                merged = {**DEFAULT_OUTPUT_GUARDRAILS, **{k: v for k, v in guard_cfg.items() if v is not None}}
                cats = merged.get("categories") or []
                merged["categories"] = [str(c).lower() for c in cats if str(c).strip()]
                cfg["output_guardrails"] = merged

            local_paths["config"] = cfg
            if entry.get("ephaptic"):
                local_paths["ephaptic"] = entry["ephaptic"]
            local_paths["model_id"] = entry.get("id") or mid
            local_paths["_ecm_cache_name"] = f"{mid}.ecm"
            local_paths["_adapt_state"] = self._load_adapt_state(kind)
            runtimes[kind] = local_paths

            logger.debug("[SDK] ✅ Prepared %s model %s with %d artifacts", kind, mid, len(local_paths))

        # Wire aux dependencies into TTS runtime if present
        if "tts" in runtimes:
            tts_rt = runtimes["tts"]
            # link vocoder model path if present
            if "vocoder" in runtimes and "vocoder_path" not in tts_rt:
                tts_rt["vocoder_path"] = runtimes["vocoder"].get("model_path")
            # speaker embeddings: from config uri or from a speaker model artifact
            cfg = tts_rt.get("config") or {}
            speaker_uri = cfg.get("speaker_embeddings_uri") or cfg.get("speaker_embeddings_url")
            if speaker_uri:
                try:
                    dst = pathlib.Path(tts_rt["model_path"]) / "speaker_embeddings.pt"
                    if not dst.exists():
                        self._download(speaker_uri, dst)
                    tts_rt["speaker_embeddings_path"] = str(dst)
                except Exception as exc:
                    logger.warning("[SDK][TTS] failed to download speaker embeddings: %s", exc)
            if "speaker" in runtimes and "speaker_embeddings_path" not in tts_rt:
                speaker_dir = pathlib.Path(runtimes["speaker"]["model_path"])
                # best-effort find embeddings file
                for cand in ("speaker_embeddings.pt", "speaker_embeddings.npy"):
                    match = speaker_dir / cand
                    if match.exists():
                        tts_rt["speaker_embeddings_path"] = str(match)
                        break

        return runtimes


    def run(self, input_data: Any, model_kind: str) -> Any:
        """
        Secure inference entrypoint.
        Loads the model runtime (with ephaptic coupling applied if present)
        and dispatches to the correct inference path for the given model_kind.
        """
        # Enforce basic attestation/session posture: agent must be enabled, personalized, and not revoked.
        status_doc = self.get_status()
        state = status_doc.get("state") or {}
        status_val = (status_doc.get("status") or "").lower()
        if status_val == "revoked" or state.get("revoked"):
            raise RuntimeError("Agent revoked; inference blocked")
        if status_val != "enabled":
            raise RuntimeError(f"Agent status is '{status_val or 'unknown'}'; inference blocked")
        if not state.get("personalized"):
            raise RuntimeError("Agent not personalized; inference blocked")
        attestation_digest = (status_doc.get("certificate") or {}).get("attestation_digest")
        self._attestation_digest = attestation_digest
        # Per-agent token cap if provided
        policy = status_doc.get("policy") or {}
        try:
            cap = int(policy.get("max_tokens_per_request") or DEFAULT_MAX_TOKEN_LENGTH)
            if cap > 0:
                self._max_tokens_cap = max(128, min(cap, DEFAULT_MAX_TOKEN_LENGTH * 4))
        except Exception:
            self._max_tokens_cap = DEFAULT_MAX_TOKEN_LENGTH
        # Agent-level schema for structured outputs
        schema = policy.get("output_schema")
        if isinstance(schema, dict):
            self._output_schema = schema
        self._minimal_logging = bool(policy.get("minimal_logging"))

        runtimes = self._ensure_runtime()
        kind = (model_kind or "").strip().lower()

        if kind not in runtimes:
            raise RuntimeError(
                f"No runtime prepared for kind '{kind}'. Available kinds: {list(runtimes.keys())}"
            )
        rt = runtimes[kind]
        _validate_io(kind, "input", input_data, max_bytes=DEFAULT_MAX_INPUT_BYTES, av_scanner=self._av_scanner)

        token_count = None
        start = time.time()

        if kind == "language":
            result, token_count = self._run_language(rt, input_data)
        elif kind == "vision":
            mode = (rt.get("config") or {}).get("mode", "").lower()
            if mode == "image_gen":
                result = self._run_vision_generate(rt, input_data)
            else:
                result = self._run_vision(rt, input_data)
        elif kind == "tts":
            result = self._run_tts(rt, input_data)
        elif kind == "stt":
            result = self._run_stt(rt, input_data)
        elif kind == "embedding":
            result = self._run_embedding(rt, input_data)
        elif kind == "multimodal":
            mode = (rt.get("config") or {}).get("mode", "").lower()
            if mode == "generate":
                result = self._run_multimodal_generate(rt, input_data)
            else:
                result = self._run_multimodal(rt, input_data)
        elif kind == "audio":
            task = (rt.get("config") or {}).get("task", "").lower()
            if task == "music_gen":
                result = self._run_audio_musicgen(rt, input_data)
            else:
                result = self._run_audio(rt, input_data)
        elif kind == "tabular":
            result = self._run_tabular(rt, input_data)
        elif kind == "timeseries":
            result = self._run_timeseries(rt, input_data)
        elif kind == "vocoder":
            # Vocoder is primarily used as aux for TTS; direct run returns its model path.
            result = rt.get("model_path")
        elif kind == "rl":
            result =  self._run_rl(rt, input_data)
        else:
            raise ValueError(f"Unsupported model_kind: {model_kind}")

        latency_ms = int((time.time() - start) * 1000)

        feedback = {"latency_ms": latency_ms}
        if token_count is not None:
            feedback["token_count"] = token_count
        self._update_adaptation(kind, rt, feedback)
        _validate_io(kind, "output", result, max_bytes=DEFAULT_MAX_INPUT_BYTES, av_scanner=None)
        # Schema validation for structured outputs if provided at agent level
        _validate_schema(result, self._output_schema)

        # Quick Telemetry (TODO: add latency or result size in telemetry so /stats/summary could later show average latency, tokens/sec,)
        geo_lat, geo_lon = self._extract_geo(input_data)
        if (geo_lat is None or geo_lon is None) and self._geo:
            geo_lat, geo_lon = self._geo

        try:
            payload = {
                "event": "inference",
                "agent_id": self.agent_id,
                "latency_ms": latency_ms,
                "model_kind": kind,
                "attestation_digest": attestation_digest,
            }
            if not self._minimal_logging and token_count is not None:
                payload["token_count"] = token_count
            if not self._minimal_logging and geo_lat is not None and geo_lon is not None:
                payload["latitude"] = geo_lat
                payload["longitude"] = geo_lon
            if RESIDENCY_TAG:
                payload["residency_tag"] = RESIDENCY_TAG
            request("POST", self.api_base, "/telemetry", headers=self._headers(),
                    json_body=payload, verify_ssl=self.verify_ssl)
        except Exception as e:
            logger.warning("⚠️ Telemetry logging failed: %s", e)

        # Audit: record policy outcome with attestation reference
        _record_policy_audit(
            request,
            self.api_base,
            self._headers(),
            self.verify_ssl,
            agent_id=self.agent_id,
            model_kind=kind,
            decision="ok",
            attestation_digest=attestation_digest,
            applied=None,
        )


        # return reference request response
        return result



    # ---------------------- Kind-specific helpers ----------------------

    def _device(self):
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"

    def _extract_geo(self, input_data: Any) -> Tuple[Optional[float], Optional[float]]:
        lat = lon = None
        try:
            if isinstance(input_data, dict):
                geo = None
                if "geo" in input_data and isinstance(input_data["geo"], dict):
                    geo = input_data["geo"]
                elif "location" in input_data and isinstance(input_data["location"], dict):
                    geo = input_data["location"]
                if geo:
                    lat = geo.get("lat") or geo.get("latitude")
                    lon = geo.get("lon") or geo.get("lng") or geo.get("longitude")
                else:
                    lat = input_data.get("latitude")
                    lon = input_data.get("longitude") or input_data.get("lng")
            if lat is not None and lon is not None:
                return float(lat), float(lon)
        except Exception:
            pass
        return None, None

    def _resolve_ecm_target(self, model: Any, target_path: Optional[str]):
        if target_path:
            node = model
            for attr in str(target_path).split("."):
                node = getattr(node, attr, None)
                if node is None:
                    break
            if node is not None:
                return node

        for attr in ("transformer", "encoder", "generator", "model", "backbone"):
            node = getattr(model, attr, None)
            if node is not None:
                return node
        return model

    def _load_model_state_dict(self, model_dir: str) -> Optional[Dict[str, Any]]:
        path = pathlib.Path(model_dir)
        try:
            import torch
        except ImportError:
            logger.warning("[SDK][Language] torch not available; cannot load model state dict")
            return None

        try:
            from safetensors.torch import load_file as load_safetensors
        except ImportError:
            load_safetensors = None

        def _load_safetensors(path_like: pathlib.Path) -> Optional[Dict[str, Any]]:
            if load_safetensors is None:
                logger.warning(
                    "[SDK][Language] safetensors not installed; cannot load %s", path_like
                )
                return None
            try:
                return load_safetensors(str(path_like))
            except Exception as exc:
                logger.warning("[SDK][Language] Failed to load %s: %s", path_like, exc)
                return None

        single_safe = path / "model.safetensors"
        if single_safe.exists():
            state = _load_safetensors(single_safe)
            if state:
                return state

        index_path = path / "model.safetensors.index.json"
        if index_path.exists():
            if load_safetensors is None:
                logger.warning(
                    "[SDK][Language] safetensors not installed; cannot read sharded weights %s",
                    index_path,
                )
            else:
                try:
                    index_data = json.loads(index_path.read_text())
                    weight_map = index_data.get("weight_map", {})
                    state: Dict[str, Any] = {}
                    loaded_shards: Set[str] = set()
                    for shard in weight_map.values():
                        if shard in loaded_shards:
                            continue
                        shard_path = path / shard
                        if not shard_path.exists():
                            logger.warning("[SDK][Language] Missing shard file %s", shard_path)
                            continue
                        shard_state = _load_safetensors(shard_path)
                        if shard_state:
                            state.update(shard_state)
                            loaded_shards.add(shard)
                    if state:
                        return state
                except Exception as exc:
                    logger.warning(
                        "[SDK][Language] Failed to load sharded state dict from %s: %s",
                        index_path,
                        exc,
                    )

        bin_candidates = [
            path / "pytorch_model.bin",
            path / "model.bin",
        ]
        for cand in bin_candidates:
            if cand.exists():
                try:
                    return torch.load(str(cand), map_location="cpu")
                except Exception as exc:
                    logger.warning("[SDK][Language] Failed to load %s: %s", cand, exc)
                    break

        if load_safetensors:
            for cand in sorted(path.glob("*.safetensors")):
                state = _load_safetensors(cand)
                if state:
                    return state

        for cand in sorted(path.glob("*.bin")):
            try:
                return torch.load(str(cand), map_location="cpu")
            except Exception:
                continue

        return None

    def _apply_ecm_if_available(self, model: Any, runtime: Dict[str, Any], install_only: bool = False) -> None:
        kind = runtime.get("kind") or "unknown"
        eph_cfg = runtime.get("ephaptic") or {}
        cfg = runtime.get("config") or {}
        if not eph_cfg and isinstance(cfg, dict):
            eph_cfg = cfg.get("ephaptic") or {}
        if not eph_cfg:
            eph_cfg = {}

        try:
            import torch
        except ImportError:
            logger.warning("[SDK][ECM] Torch not available; skipping ECM injection")
            return

        adapt_state = runtime.get("_adapt_state")
        if not isinstance(adapt_state, dict):
            adapt_state = {"lambda_scale": 1.0}
            runtime["_adapt_state"] = adapt_state

        variant = str(eph_cfg.get("variant") or "multiplicative").lower()
        phi = str(eph_cfg.get("phi") or "gelu")
        ecm_init = str(eph_cfg.get("ecm_init") or "identity")

        try:
            epsilon = float(eph_cfg.get("epsilon") if eph_cfg.get("epsilon") is not None else 1.0)
        except Exception:
            epsilon = 1.0
        try:
            lambda0 = float(eph_cfg.get("lambda0") if eph_cfg.get("lambda0") is not None else 0.01)
        except Exception:
            lambda0 = 0.01

        target = self._resolve_ecm_target(model, eph_cfg.get("target") or eph_cfg.get("module"))
        if target is None:
            logger.warning("[SDK][ECM] Unable to resolve ECM target; skipping injection")
            return

        if not getattr(target, "_ephaptic_hook_installed", False):
            hidden_dim = eph_cfg.get("hidden_dim")
            try:
                from .ecm import inject_ecm
                inject_ecm(
                    target,
                    epsilon=epsilon,
                    lambda_init_mag=lambda0,
                    phi=phi,
                    ecm_init=ecm_init,
                    variant=variant,
                    hidden_dim=int(hidden_dim) if hidden_dim is not None else None,
                )
                target._ephaptic_hook_installed = True
            except Exception as e:
                logger.warning("[SDK][ECM] ECM hook injection failed: %s", e)
                return

        if install_only:
            return

        ecm_tensor = None
        try:
            if runtime.get("ecm_bytes"):
                raw = runtime["ecm_bytes"]
                if isinstance(raw, memoryview):
                    raw = raw.tobytes()
                buffer = io.BytesIO(raw if isinstance(raw, (bytes, bytearray)) else bytes(raw))
                ecm_tensor = torch.load(buffer, map_location=self._device())
            elif runtime.get("ecm_path") and os.path.exists(runtime["ecm_path"]):
                ecm_tensor = torch.load(runtime["ecm_path"], map_location=self._device())
        except Exception as e:
            logger.warning("[SDK][ECM] Failed to load ECM tensor: %s", e)
            return

        if ecm_tensor is None:
            logger.debug("[SDK][ECM] No ECM tensor available; skipping parameter load")
            return

        Lambda_param = getattr(target, "lambda_ecm", None)
        if Lambda_param is None:
            logger.warning("[SDK][ECM] lambda_ecm parameter missing after injection")
            return

        try:
            shaped = ecm_tensor.to(Lambda_param.device, dtype=Lambda_param.dtype)
            if shaped.shape != Lambda_param.shape:
                shaped = shaped.reshape(Lambda_param.shape)
            scale = float(adapt_state.get("lambda_scale", 1.0))
            if scale != 1.0:
                shaped = shaped * scale
            with torch.no_grad():
                Lambda_param.copy_(shaped)
            logger.debug("[SDK][ECM] Applied ECM tensor shape=%s variant=%s epsilon=%.4f", tuple(shaped.shape), variant, epsilon)
        except Exception as e:
            logger.warning("[SDK][ECM] Failed to apply ECM tensor: %s", e)
            return

        try:
            state = runtime.get("_adapt_state") or {}
            if state.get("pending_digest"):
                digest = self._persist_adapted_ecm(kind, runtime, shaped)
                if digest:
                    state["last_digest"] = digest
                    state["pending_digest"] = False
                    self._save_adapt_state(kind, state)
                    self._report_adaptation(kind, runtime, digest)
        except Exception as exc:
            logger.debug("[SDK][Adapt] Post-apply persist skipped: %s", exc)

    def _update_adaptation(self, kind: str, runtime: Dict[str, Any], feedback: Dict[str, Any]) -> None:
        eph = runtime.get("ephaptic") or {}
        adapt_cfg = eph.get("adaptability")
        if not adapt_cfg:
            return
        if adapt_cfg is True:
            adapt_cfg = {}

        latency_ms = feedback.get("latency_ms")
        token_count = feedback.get("token_count")
        state = runtime.get("_adapt_state")
        if not isinstance(state, dict):
            state = self._load_adapt_state(kind)
            runtime["_adapt_state"] = state

        target_latency = float(adapt_cfg.get("target_latency_ms", 600.0))
        epsilon_lr = float(adapt_cfg.get("epsilon_lr", 0.05))
        epsilon_min = float(adapt_cfg.get("epsilon_min", 0.05))
        epsilon_max = float(adapt_cfg.get("epsilon_max", 3.0))
        lambda_lr = float(adapt_cfg.get("lambda_lr", 0.02))
        lambda_min = float(adapt_cfg.get("lambda_min", 0.25))
        lambda_max = float(adapt_cfg.get("lambda_max", 4.0))

        epsilon = float(state.get("epsilon", eph.get("epsilon", 1.0) or 1.0))
        lambda_scale = float(state.get("lambda_scale", 1.0))
        prev_epsilon = epsilon
        prev_lambda = lambda_scale

        if latency_ms is not None:
            error = (latency_ms - target_latency) / max(target_latency, 1.0)
            epsilon *= (1.0 - epsilon_lr * error)
            lambda_scale *= (1.0 - lambda_lr * error)

        if token_count is not None:
            target_tokens = float(adapt_cfg.get("target_tokens", 256))
            token_error = (token_count - target_tokens) / max(target_tokens, 1.0)
            epsilon *= (1.0 + epsilon_lr * 0.5 * token_error)

        epsilon = max(min(epsilon, epsilon_max), epsilon_min)
        lambda_scale = max(min(lambda_scale, lambda_max), lambda_min)

        changed = (abs(epsilon - prev_epsilon) > 1e-6) or (abs(lambda_scale - prev_lambda) > 1e-6)

        eph["epsilon"] = epsilon
        state["epsilon"] = epsilon
        state["lambda_scale"] = lambda_scale
        state["updated_at"] = int(time.time())

        if adapt_cfg:
            state["adapt_config"] = adapt_cfg

        if changed:
            state["pending_digest"] = True
            self._save_adapt_state(kind, state)

    def _persist_adapted_ecm(self, kind: str, runtime: Dict[str, Any], tensor: Any) -> Optional[str]:
        try:
            import torch
        except ImportError:
            logger.debug("[SDK][Adapt] Torch unavailable; cannot persist adapted ECM.")
            return None

        try:
            cpu_tensor = tensor.detach().to("cpu")
        except Exception as exc:
            logger.debug("[SDK][Adapt] Failed to detach tensor for %s: %s", kind, exc)
            return None

        buf = io.BytesIO()
        try:
            torch.save(cpu_tensor, buf)
        except Exception as exc:
            logger.warning("[SDK][Adapt] Failed to serialize adapted ECM for %s: %s", kind, exc)
            return None

        plaintext = buf.getvalue()
        digest = "sha256:" + hashlib.sha256(plaintext).hexdigest()

        cache_name = runtime.get("_ecm_cache_name") or f"{runtime.get('model_id','ecm')}.ecm"
        model_dir = pathlib.Path(runtime.get("model_path") or ".")

        if runtime.get("ecm_path"):
            try:
                with open(runtime["ecm_path"], "wb") as f:
                    f.write(plaintext)
            except Exception as exc:
                logger.warning("[SDK][Adapt] Failed to write plaintext ECM for %s: %s", kind, exc)
        else:
            try:
                storage.write_encrypted(str(model_dir), cache_name, plaintext)
            except Exception as exc:
                logger.warning("[SDK][Adapt] Failed to encrypt adapted ECM for %s: %s", kind, exc)

        runtime["ecm_bytes"] = plaintext
        return digest

    def _report_adaptation(self, kind: str, runtime: Dict[str, Any], digest: str) -> None:
        state = runtime.get("_adapt_state") or {}
        payload = {
            "event": "adaptation",
            "agent_id": self.agent_id,
            "model_kind": kind,
            "digest": digest,
            "epsilon": state.get("epsilon"),
            "lambda_scale": state.get("lambda_scale"),
            "updated_at": state.get("updated_at"),
        }
        try:
            request("POST", self.api_base, "/telemetry", headers=self._headers(), json_body=payload, verify_ssl=self.verify_ssl)
        except Exception as exc:
            logger.debug("[SDK][Adapt] Telemetry send failed: %s", exc)
        runtime["_adapt_state"] = state

    def _run_language(self, runtime: Dict[str, Any], prompt: str) -> tuple[str, int]:
        """
        Runs a language model with input/output policy enforcement baked in.
        Returns (decoded_text, token_count).
        """
        model_path = runtime.get("model_path")
        if not model_path:
            raise RuntimeError("Language runtime missing model_path")

        # --- Enforce input policies for language ---
        enforced_input, in_policies = self.enforce_policies_model_kind(prompt, "language", "input")
        if not enforced_input:
            logger.warning("⚠️ Input blocked by policies: %s", in_policies)
            return "[BLOCKED BY POLICIES]", 0

        try:
            from transformers import (
                AutoTokenizer,
                AutoModelForCausalLM,
                AutoModelForSeq2SeqLM,
                AutoConfig,
            )
            import torch
        except ImportError:
            raise RuntimeError("Transformers not installed. `pip install transformers`")

        logger.debug("[SDK][Language] Loading model from %s", model_path)

        with self._suppress_transformers_warnings():
            tok = AutoTokenizer.from_pretrained(model_path)
            cfg = AutoConfig.from_pretrained(model_path)
        logger.debug("[SDK][Language] Detected model_type=%s", cfg.model_type)

        # Some checkpoints (e.g., Gemma3) strongly recommend eager attention
        model_id = (runtime.get("id") or runtime.get("model_id") or "").lower()
        eager_models = (cfg.model_type or "").lower() in {"gemma", "gemma2", "gemma3"}
        eager_models = eager_models or ("gemma" in cfg.__class__.__name__.lower())
        eager_models = eager_models or ("gemma" in model_id)
        if eager_models and hasattr(cfg, "attn_implementation"):
            attn_impl = getattr(cfg, "attn_implementation", None)
            if not attn_impl or str(attn_impl).lower() == "sdpa":
                setattr(cfg, "attn_implementation", "eager")
                logger.debug("[SDK][Language] Forcing attn_implementation='eager' for %s", cfg.model_type)

        if cfg.model_type in ("t5", "mt5", "bart", "mbart", "pegasus", "prophetnet", "marian"):
            logger.debug("[SDK][Language] → Using AutoModelForSeq2SeqLM")
            model_cls = AutoModelForSeq2SeqLM
        else:
            logger.debug("[SDK][Language] → Using AutoModelForCausalLM")
            model_cls = AutoModelForCausalLM

        state_dict = self._load_model_state_dict(model_path)
        if state_dict:
            with self._suppress_transformers_warnings():
                model = model_cls.from_config(cfg)
            # Install ECM hooks/parameters before weights are loaded so state_dict restores λ without warnings
            self._apply_ecm_if_available(model, runtime, install_only=True)
            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
            if missing_keys:
                logger.debug("[SDK][Language] Missing keys after load: %s", missing_keys)
            if unexpected_keys:
                logger.debug("[SDK][Language] Unexpected keys after load: %s", unexpected_keys)
            model.to(self._device())
        else:
            logger.warning(
                "[SDK][Language] Could not locate state dict under %s; falling back to from_pretrained",
                model_path,
            )
            with self._suppress_transformers_warnings():
                model = model_cls.from_pretrained(model_path, config=cfg).to(self._device())

        self._apply_ecm_if_available(model, runtime)

        inputs = tok(str(enforced_input), return_tensors="pt").to(self._device())
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=256)

        decoded = tok.decode(outputs[0], skip_special_tokens=True)
        token_count = outputs.shape[1]  # number of tokens generated

        logger.debug("[SDK][Language] Generated raw: %s%s",
                     decoded[:80], "..." if len(decoded) > 80 else "")

        # --- Enforce output policies for language ---
        enforced_output, out_policies = self.enforce_policies_model_kind(decoded, "language", "output")
        if not enforced_output:
            logger.warning("⚠️ Output blocked by policies: %s", out_policies)
            return "[BLOCKED BY POLICIES]", token_count

        guard_output, _ = self._apply_guardrails_with_logging(runtime, enforced_output, "language")
        if guard_output is None:
            return "[BLOCKED BY GUARDRAILS]", token_count

        if in_policies:
            logger.info("Input policies applied: %s", in_policies)
        if out_policies:
            logger.info("Output policies applied: %s", out_policies)
        return guard_output, token_count


    # --- Vision (image classification) ---
    def _run_vision(self, runtime: Dict[str, Any], image_input: Any) -> str:
        """
        Runs a vision model with policy enforcement.
        """
        model_path = runtime.get("model_path")
        if not model_path:
            raise RuntimeError("Vision runtime missing model_path")
        # --- Enforce input policies for vision ---
        enforced_input, in_policies = self.enforce_policies_model_kind(image_input, "vision", "input")
        if not enforced_input:
            logger.warning("⚠️ Vision input blocked by policies: %s", in_policies)
            return "[BLOCKED BY POLICIES]"
        image_input = enforced_input

        try:
            from PIL import Image
            from transformers import AutoImageProcessor, AutoModelForImageClassification
            import torch
        except ImportError:
            raise RuntimeError("Transformers/Pillow not installed. `pip install transformers pillow`")

        if not hasattr(image_input, "convert"):
            image_input = Image.open(image_input).convert("RGB")

        processor = AutoImageProcessor.from_pretrained(model_path)
        model = AutoModelForImageClassification.from_pretrained(model_path)
        self._apply_ecm_if_available(model, runtime)
        model = model.to(self._device())
        inputs = processor(images=image_input, return_tensors="pt").to(self._device())
        with torch.no_grad():
            logits = model(**inputs).logits
            probs = logits.softmax(-1)
            label_id = int(probs.argmax(-1).item())
        result = model.config.id2label.get(label_id, str(label_id))

        # --- Enforce output policies for vision ---
        enforced_output, out_policies = self.enforce_policies_model_kind(result, "vision", "output")
        if enforced_output is None:
            logger.warning("⚠️ Vision output blocked by policies: %s", out_policies)
            return "[BLOCKED BY POLICIES]"

        guard_output, _ = self._apply_guardrails_with_logging(runtime, enforced_output, "vision")
        if guard_output is None:
            return "[BLOCKED BY GUARDRAILS]"

        if in_policies:
            logger.info("Vision input policies applied: %s", in_policies)
        if out_policies:
            logger.info("Vision output policies applied: %s", out_policies)
        return guard_output

    def _run_vision_generate(self, runtime: Dict[str, Any], text_prompt: str):
        """
        Text-to-image generation (Stable Diffusion-style) with policy enforcement.
        Expects config.mode == "image_gen" and SD-style artifacts under model_path.
        """
        cfg = runtime.get("config") or {}
        model_path = runtime.get("model_path")
        if not model_path:
            raise RuntimeError("Vision image_gen runtime missing model_path")
        enforced_input, in_policies = self.enforce_policies_model_kind(text_prompt, "vision", "input")
        if enforced_input is None:
            logger.warning("⚠️ Vision-gen input blocked by policies: %s", in_policies)
            return "[BLOCKED BY POLICIES]"
        text_prompt = enforced_input

        try:
            from diffusers import StableDiffusionPipeline, DDIMScheduler
            import torch
        except ImportError:
            raise RuntimeError("Install diffusers/torch for image generation: `pip install diffusers torch torchvision`")

        # Validate expected components exist
        required = ["unet", "vae", "text_encoder"]
        missing = [comp for comp in required if not (pathlib.Path(model_path) / comp).exists()]
        if missing:
            raise RuntimeError(f"Image generation requires components {missing} under {model_path}")

        try:
            pipe = StableDiffusionPipeline.from_pretrained(
                model_path,
                torch_dtype=torch.float16 if self._device() == "cuda" else torch.float32,
                safety_checker=None,
                local_files_only=False,
            )
        except Exception as exc:
            raise RuntimeError(f"Failed to load diffusion pipeline from {model_path}: {exc}")

        try:
            self._apply_ecm_if_available(pipe, runtime)
        except Exception as exc:
            logger.debug("[SDK][ECM] Vision-gen hook skipped: %s", exc)

        pipe = pipe.to(self._device())
        sched_cfg = cfg.get("scheduler_config")
        if sched_cfg:
            try:
                pipe.scheduler = DDIMScheduler.from_pretrained(model_path, subfolder=sched_cfg)
            except Exception:
                pass

        height = int(cfg.get("height") or 512)
        width = int(cfg.get("width") or 512)
        guidance = float(cfg.get("guidance_scale") or 7.5)
        steps = int(cfg.get("num_inference_steps") or 30)

        image = pipe(str(text_prompt), height=height, width=width, guidance_scale=guidance, num_inference_steps=steps).images[0]

        enforced_output, out_policies = self.enforce_policies_model_kind(image, "vision", "output")
        if enforced_output is None:
            logger.warning("⚠️ Vision-gen output blocked by policies: %s", out_policies)
            return "[BLOCKED BY POLICIES]"

        return enforced_output

    def _run_tts(self, runtime: Dict[str, Any], text: str):
        """
        Runs a TTS model with policy enforcement.
        """
        model_path = runtime.get("model_path")
        if not model_path:
            raise RuntimeError("TTS runtime missing model_path")
        enforced_input, in_policies = self.enforce_policies_model_kind(text, "tts", "input")
        if enforced_input is None:
            logger.warning("⚠️ TTS input blocked by policies: %s", in_policies)
            return None
        text = enforced_input

        try:
            import numpy as np
            import torch
            from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
        except ImportError:
            raise RuntimeError("Transformers/torch not installed. `pip install transformers torch soundfile`")

        processor = SpeechT5Processor.from_pretrained(model_path)
        model = SpeechT5ForTextToSpeech.from_pretrained(model_path)
        self._apply_ecm_if_available(model, runtime)
        model = model.to(self._device())
        inputs = processor(text=str(text), return_tensors="pt").to(self._device())
        # Some transformers versions require speaker_embeddings; provide a default if absent
        speaker_embeddings = runtime.get("speaker_embeddings")
        if speaker_embeddings is None:
            speaker_embeddings_path = runtime.get("speaker_embeddings_path")
            speaker_embeddings = self._load_speaker_embeddings(model_path, model, speaker_embeddings_path)
        with torch.no_grad():
            vocoder = self._load_tts_vocoder(runtime.get("vocoder_path") or model_path, SpeechT5HifiGan)
            waveform = model.generate_speech(
                inputs["input_ids"],
                speaker_embeddings=speaker_embeddings,
                vocoder=vocoder,
            )
        result = waveform.cpu().numpy()

        enforced_output, out_policies = self.enforce_policies_model_kind(result, "tts", "output")
        if enforced_output is None:
            logger.warning("⚠️ TTS output blocked by policies: %s", out_policies)
            return None

        guard_output, _ = self._apply_guardrails_with_logging(runtime, enforced_output, "tts")
        if guard_output is None:
            return None

        return guard_output


    def _load_speaker_embeddings(self, model_path: str, model, override_path: Optional[str] = None) -> "torch.Tensor":
        """
        Attempt to load speaker embeddings from the model directory; otherwise return zeros.
        """
        import torch
        import numpy as np

        candidates = []
        if override_path:
            candidates.append(override_path)
        candidates += ["speaker_embeddings.pt", "speaker_embeddings.npy"]
        for fname in candidates:
            for path in pathlib.Path(model_path).rglob(os.path.basename(fname)):
                try:
                    if path.suffix == ".pt":
                        emb = torch.load(path, map_location=self._device())
                    else:
                        emb = torch.from_numpy(np.load(path))
                    if emb is not None:
                        return emb.to(self._device())
                except Exception:
                    continue

        dim = getattr(model.config, "speaker_embeddings_dim", 512)
        return torch.zeros(
            (1, dim),
            device=self._device(),
            dtype=getattr(model.dtype, "value", None) or torch.float32,
        )


    def _load_tts_vocoder(self, model_path: str, VocoderCls):
        """
        Try to load a HiFiGan vocoder colocated with the TTS model.
        """
        try:
            vocoder_path = None
            for cand in ("vocoder", "hifigan", model_path):
                cand_path = pathlib.Path(model_path) / cand if cand != model_path else pathlib.Path(model_path)
                if cand_path.exists():
                    vocoder_path = cand_path
                    break
            if vocoder_path is None:
                return None
            vocoder = VocoderCls.from_pretrained(str(vocoder_path))
            return vocoder.to(self._device())
        except Exception:
            return None


    def _run_stt(self, runtime: Dict[str, Any], audio_input: Any) -> str:
        """
        Runs an STT model with policy enforcement.
        """
        model_path = runtime.get("model_path")
        if not model_path:
            raise RuntimeError("STT runtime missing model_path")
        enforced_input, in_policies = self.enforce_policies_model_kind(audio_input, "stt", "input")
        if enforced_input is None:
            logger.warning("⚠️ STT input blocked by policies: %s", in_policies)
            return "[BLOCKED BY POLICIES]"
        audio_input = enforced_input

        try:
            import numpy as np
            import torch
            from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
            import librosa
        except ImportError:
            raise RuntimeError("Install deps: `pip install transformers torch librosa`")

        processor = Wav2Vec2Processor.from_pretrained(model_path)
        model = Wav2Vec2ForCTC.from_pretrained(model_path)
        self._apply_ecm_if_available(model, runtime)
        model = model.to(self._device())

        if isinstance(audio_input, str):
            waveform, sr = librosa.load(audio_input, sr=16000)
        else:
            waveform = audio_input.numpy() if hasattr(audio_input, "numpy") else np.asarray(audio_input)
            sr = 16000

        input_values = processor(waveform, sampling_rate=sr, return_tensors="pt").input_values.to(self._device())
        with torch.no_grad():
            logits = model(input_values).logits
            pred_ids = logits.argmax(dim=-1)
        result = processor.decode(pred_ids[0])

        enforced_output, out_policies = self.enforce_policies_model_kind(result, "stt", "output")
        if not enforced_output:
            logger.warning("⚠️ STT output blocked by policies: %s", out_policies)
            return "[BLOCKED BY POLICIES]"
        guard_output, _ = self._apply_guardrails_with_logging(runtime, enforced_output, "stt")
        if guard_output is None:
            return "[BLOCKED BY GUARDRAILS]"

        if in_policies:
            logger.info("STT input policies applied: %s", in_policies)
        if out_policies:
            logger.info("STT output policies applied: %s", out_policies)
        return guard_output


    # --- Embedding (SentenceTransformers) ---
    def _run_embedding(self, runtime: Dict[str, Any], sentences: Union[str, List[str]]):
        """
        Runs an embedding model with policy enforcement.
        """
        model_path = runtime.get("model_path")
        if not model_path:
            raise RuntimeError("Embedding runtime missing model_path")
        enforced_input, in_policies = self.enforce_policies_model_kind(sentences, "embedding", "input")
        if not enforced_input:
            logger.warning("⚠️ Embedding input blocked by policies: %s", in_policies)
            return "[BLOCKED BY POLICIES]"
        sentences = enforced_input

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise RuntimeError("Install sentence-transformers: `pip install sentence-transformers`")

        texts = [sentences] if isinstance(sentences, str) else [str(s) for s in sentences]

        model = SentenceTransformer(model_path, device=self._device())
        self._apply_ecm_if_available(model, runtime)
        vecs = model.encode(texts, convert_to_numpy=True)

        enforced_output, out_policies = self.enforce_policies_model_kind(vecs, "embedding", "output")
        if not enforced_output:
            logger.warning("⚠️ Embedding output blocked by policies: %s", out_policies)
            return "[BLOCKED BY POLICIES]"

        return enforced_output.tolist() if len(texts) > 1 else enforced_output[0].tolist()


    # --- Multimodal (CLIP text↔image similarity) ---
    def _run_multimodal(self, runtime: Dict[str, Any], payload: Dict[str, Any]):
        """
        Runs a multimodal CLIP model with policy enforcement.
        payload = {"image": <path or PIL.Image>, "text": "caption"}
        """
        model_path = runtime.get("model_path")
        if not model_path:
            raise RuntimeError("Multimodal runtime missing model_path")
        enforced_input, in_policies = self.enforce_policies_model_kind(payload, "multimodal", "input")
        if not enforced_input:
            logger.warning("⚠️ Multimodal input blocked by policies: %s", in_policies)
            return "[BLOCKED BY POLICIES]"
        payload = enforced_input

        try:
            import torch
            from PIL import Image
            from transformers import CLIPProcessor, CLIPModel
        except ImportError:
            raise RuntimeError("Install CLIP deps: `pip install transformers pillow torch`")

        image, text = payload["image"], payload["text"]
        if not hasattr(image, "convert"):
            image = Image.open(image).convert("RGB")

        processor = CLIPProcessor.from_pretrained(model_path)
        model = CLIPModel.from_pretrained(model_path)
        self._apply_ecm_if_available(model, runtime)
        model = model.to(self._device())
        inputs = processor(text=[str(text)], images=image, return_tensors="pt", padding=True).to(self._device())

        with torch.no_grad():
            outputs = model(**inputs)
            image_embeds = outputs.image_embeds / outputs.image_embeds.norm(p=2, dim=-1, keepdim=True)
            text_embeds  = outputs.text_embeds  / outputs.text_embeds.norm(p=2, dim=-1, keepdim=True)
            sim = (image_embeds @ text_embeds.T).squeeze().item()

        result = {"similarity": float(sim)}

        enforced_output, out_policies = self.enforce_policies_model_kind(result, "multimodal", "output")
        if not enforced_output:
            logger.warning("⚠️ Multimodal output blocked by policies: %s", out_policies)
            return "[BLOCKED BY POLICIES]"

        return enforced_output


    def _run_multimodal_generate(self, runtime: Dict[str, Any], payload: Dict[str, Any]):
        """
        Placeholder multimodal generation: currently not implemented.
        """
        raise RuntimeError("Multimodal generation is not supported yet; provide a generation-capable model or use similarity mode.")


    # --- Audio (audio event classification) ---
    def _run_audio(self, runtime: Dict[str, Any], audio_input: Any):
        """
        Runs an audio classification pipeline with policy enforcement.
        """
        model_path = runtime.get("model_path")
        if not model_path:
            raise RuntimeError("Audio runtime missing model_path")
        enforced_input, in_policies = self.enforce_policies_model_kind(audio_input, "audio", "input")
        if not enforced_input:
            logger.warning("⚠️ Audio input blocked by policies: %s", in_policies)
            return "[BLOCKED BY POLICIES]"
        audio_input = enforced_input

        try:
            import numpy as np
            import librosa
            from transformers import pipeline
        except ImportError:
            raise RuntimeError("Install deps: `pip install transformers librosa`")

        classifier = pipeline("audio-classification", model=model_path, device=0 if self._device()=="cuda" else -1)
        try:
            model = getattr(classifier, "model", None)
            if model is not None:
                self._apply_ecm_if_available(model, runtime)
        except Exception as exc:
            logger.debug("[SDK][ECM] Audio pipeline hook skipped: %s", exc)
        if isinstance(audio_input, str):
            result = classifier(audio_input)
        else:
            import soundfile as sf
            tmp = "_tmp_infer.wav"
            sf.write(tmp, audio_input, 16000)
            result = classifier(tmp)
            try: os.remove(tmp)
            except Exception: pass

        enforced_output, out_policies = self.enforce_policies_model_kind(result, "audio", "output")
        if not enforced_output:
            logger.warning("⚠️ Audio output blocked by policies: %s", out_policies)
            return "[BLOCKED BY POLICIES]"

        return enforced_output

    def _run_audio_musicgen(self, runtime: Dict[str, Any], prompt: str):
        """
        Music/text-to-audio generation using a MusicGen-style checkpoint.
        Expects config.task == \"music_gen\" and artifact_urls.model present.
        """
        cfg = runtime.get("config") or {}
        model_path = runtime.get("model_path")
        if not model_path:
            raise RuntimeError("Audio music_gen runtime missing model_path")
        enforced_input, in_policies = self.enforce_policies_model_kind(prompt, "audio", "input")
        if enforced_input is None:
            logger.warning("⚠️ Audio-gen input blocked by policies: %s", in_policies)
            return "[BLOCKED BY POLICIES]"
        prompt = enforced_input

        try:
            from audiocraft.models import MusicGen
            import torch
        except ImportError:
            raise RuntimeError("Install audiocraft for music generation: `pip install audiocraft`")

        if not os.path.exists(model_path):
            raise RuntimeError(f"MusicGen model path not found: {model_path}")
        try:
            model = MusicGen.get_pretrained(model_path, device=self._device())
        except Exception as exc:
            raise RuntimeError(f"Failed to load music gen model from {model_path}: {exc}")

        sample_rate = int(cfg.get("sample_rate") or 32000)
        duration = float(cfg.get("duration") or 8.0)
        try:
            self._apply_ecm_if_available(model, runtime)
        except Exception as exc:
            logger.debug("[SDK][ECM] Audio-gen hook skipped: %s", exc)

        with torch.no_grad():
            audio = model.generate([str(prompt)], progress=False, use_sampling=True, duration=duration, sample_rate=sample_rate)[0]
        audio_np = audio.cpu().numpy()

        enforced_output, out_policies = self.enforce_policies_model_kind(audio_np, "audio", "output")
        if enforced_output is None:
            logger.warning("⚠️ Audio-gen output blocked by policies: %s", out_policies)
            return "[BLOCKED BY POLICIES]"

        return enforced_output


    # --- Tabular (scikit-learn pipeline) ---
    def _run_tabular(self, runtime: Dict[str, Any], X: Any):
        """
        Runs a scikit-learn pipeline with policy enforcement.
        """
        model_path = runtime.get("model_path")
        if not model_path:
            raise RuntimeError("Tabular runtime missing model_path")
        enforced_input, in_policies = self.enforce_policies_model_kind(X, "tabular", "input")
        if not enforced_input:
            logger.warning("⚠️ Tabular input blocked by policies: %s", in_policies)
            return "[BLOCKED BY POLICIES]"
        X = enforced_input

        try:
            import joblib, glob
        except ImportError:
            raise RuntimeError("Install joblib: `pip install joblib`")

        path = model_path
        if os.path.isdir(model_path):
            cand = glob.glob(os.path.join(model_path, "*.joblib"))
            if not cand:
                raise RuntimeError("No .joblib model found for tabular runtime.")
            path = cand[0]

        model = joblib.load(path)
        result = model.predict(X).tolist()

        enforced_output, out_policies = self.enforce_policies_model_kind(result, "tabular", "output")
        if not enforced_output:
            logger.warning("⚠️ Tabular output blocked by policies: %s", out_policies)
            return "[BLOCKED BY POLICIES]"

        return enforced_output

    # --- Time series (TorchScript forecaster) ---
    def _run_timeseries(self, runtime: Dict[str, Any], seq: Any):
        """
        Runs a TorchScript time series forecaster with policy enforcement.
        """
        model_path = runtime.get("model_path")
        if not model_path:
            raise RuntimeError("Time series runtime missing model_path")
        # --- Enforce input policies ---
        enforced_input, in_policies = self.enforce_policies_model_kind(seq, "timeseries", "input")
        if not enforced_input:
            logger.warning("⚠️ Time series input blocked by policies: %s", in_policies)
            return "[BLOCKED BY POLICIES]"
        seq = enforced_input

        try:
            import torch, numpy as np, glob
        except ImportError:
            raise RuntimeError("Install torch/numpy")

        ts_path = None
        if os.path.isdir(model_path):
            cand = glob.glob(os.path.join(model_path, "forecast.pt")) or glob.glob(os.path.join(model_path, "*.pt"))
            if not cand:
                raise RuntimeError("No TorchScript .pt found for timeseries runtime.")
            ts_path = cand[0]
        else:
            ts_path = model_path

        model = torch.jit.load(ts_path, map_location=self._device())
        try:
            self._apply_ecm_if_available(model, runtime)
        except Exception as exc:
            logger.debug("[SDK][ECM] Time series hook skipped: %s", exc)
        x = torch.tensor(seq, dtype=torch.float32, device=self._device()).unsqueeze(0)  # [1, T]
        with torch.no_grad():
            y = model(x)  # [1, H]
        result = y.squeeze(0).detach().cpu().numpy().tolist()

        # --- Enforce output policies ---
        enforced_output, out_policies = self.enforce_policies_model_kind(result, "timeseries", "output")
        if not enforced_output:
            logger.warning("⚠️ Time series output blocked by policies: %s", out_policies)
            return "[BLOCKED BY POLICIES]"

        if in_policies:
            logger.info("Time series input policies applied: %s", in_policies)
        if out_policies:
            logger.info("Time series output policies applied: %s", out_policies)

        return enforced_output


    # --- RL (stable-baselines3 policy) ---
    def _run_rl(self, runtime: Dict[str, Any], payload: Dict[str, Any]):
        """
        Runs an RL model with policy enforcement.
        """
        model_path = runtime.get("model_path")
        if not model_path:
            raise RuntimeError("RL runtime missing model_path")
        enforced_input, in_policies = self.enforce_policies_model_kind(payload, "rl", "input")
        if not enforced_input:
            logger.warning("⚠️ RL input blocked by policies: %s", in_policies)
            return "[BLOCKED BY POLICIES]"
        payload = enforced_input

        try:
            import numpy as np, glob
            from stable_baselines3 import PPO, A2C, DQN, SAC, TD3
        except ImportError:
            raise RuntimeError("Install stable-baselines3: `pip install stable-baselines3[extra]`")

        if os.path.isdir(model_path):
            cand = glob.glob(os.path.join(model_path, "*.zip"))
            if not cand:
                raise RuntimeError("No sb3 .zip policy found for RL runtime.")
            policy_path = cand[0]
        else:
            policy_path = model_path

        model = None
        for cls in (PPO, A2C, DQN, SAC, TD3):
            try:
                model = cls.load(policy_path, device=self._device())
                break
            except Exception:
                continue
        if model is None:
            raise RuntimeError("Unsupported/unknown SB3 policy format")

        try:
            self._apply_ecm_if_available(model.policy, runtime)
        except Exception as exc:
            logger.debug("[SDK][ECM] RL hook skipped: %s", exc)

        obs = payload.get("obs")
        if obs is None:
            raise ValueError("RL run expects payload {'obs': ...}")

        obs_np = np.asarray(obs, dtype=np.float32)
        action, _ = model.predict(obs_np, deterministic=True)

        enforced_output, out_policies = self.enforce_policies_model_kind(action, "rl", "output")
        if not enforced_output:
            logger.warning("⚠️ RL output blocked by policies: %s", out_policies)
            return "[BLOCKED BY POLICIES]"

        return enforced_output.tolist() if hasattr(enforced_output, "tolist") else enforced_output


    # # A2A SUPPORT
    # def send_message(self, to_agent_id: str, tool: str, args: dict) -> dict:
    #     """
    #     Create, sign, and send an A2A message to another agent.
    #     - Signs with this agent’s DID/cert
    #     - Encrypts payload
    #     Returns message metadata (id, ts).
    #     """

    # def fetch_messages(self) -> list[dict]:
    #     """
    #     Retrieve incoming messages for this agent from the backend.
    #     Each message must be verified (signature, provenance).
    #     """

    # def verify_message(self, message: dict) -> bool:
    #     """
    #     Verify authenticity (dPKI, DID, revocation status).
    #     Returns True if valid.
    #     """

    # def process_message(self, message: dict) -> dict:
    #     """
    #     Execute the requested tool if message is valid.
    #     - Example message:
    #       {
    #         "from": "agent_A",
    #         "to": "agent_B",
    #         "tool": "language.respond",
    #         "args": {"text": "hello"},
    #         "sig": "..."
    #       }
    #     - Calls run_tool(message["tool"], message["args"])
    #     - Returns signed response
    #     """

    # # MCP SUPPORT
    # def list_tools(self) -> list[dict]:
    #     """
    #     Return the list of available tools this agent supports.
    #     Each tool maps to a model_kind or extension.
    #     Example:
    #     [
    #         {"name": "stt.transcribe", "description": "Convert audio to text"},
    #         {"name": "vision.classify", "description": "Label an image frame"},
    #         {"name": "language.respond", "description": "Generate a text response"}
    #     ]
    #     """

    # def serve_mcp(self, host: str = "0.0.0.0", port: int = 8081):
    #     """
    #     Run an MCP server endpoint.
    #     Exposes list_tools() and run_tool() for external clients.
    #     MCP clients can then call these tools.
    #     """

    # def run_tool(self, tool_name: str, arguments: dict) -> dict:
    #     """
    #     Execute a registered tool by name with arguments.
    #     Wraps agent.run(input, model_kind=...) under the hood.
    #     Returns structured JSON.
    #     """


# ---------------------- SIE helper classes ----------------------
import urllib.request

# ... keep existing imports above ...
from .crypto import hpke
from . import storage

class SecureInferenceError(RuntimeError):
    pass

def _http_get_bytes(url: str) -> bytes:
    import urllib.request
    with urllib.request.urlopen(url) as r:
        return r.read()

def _sha256_hex(b: bytes) -> str:
    import hashlib
    return hashlib.sha256(b).hexdigest()

class SIEManager:
    """
    Secure ECM loader with encrypted-at-rest cache:
      - status gate (revocation → jam)
      - ciphertext fetch + digest check
      - CEK unwrap (HPKE) in hardware/soft → AES-GCM decrypt to RAM
      - write encrypted cache (TPM-sealed DEK), never plaintext to disk
      - return plaintext ECM bytes in memory only
    """
    def __init__(
        self,
        *,
        base_url: str,
        agent_id_or_did: str,
        state_dir: str,
        api_key: Optional[str],
        verify_ssl: bool = True,
        privkey_pem: Optional[str] = None,                   # dev
        privkey_loader: Optional[Callable[[], str]] = None,  # dev
        tpm_ecdh: Optional[Callable[[bytes], bytes]] = None, # prod
    ):
        self.base_url   = base_url.rstrip("/")
        self.agent_id   = agent_id_or_did
        self.state_dir  = state_dir
        self.verify_ssl = verify_ssl
        self.api_key = get_api_key(
            api_key,
            base_url=base_url,
            agent_instance_id=agent_id_or_did,
            verify_ssl=verify_ssl,
        )
        self.privkey_pem     = privkey_pem
        self.privkey_loader  = privkey_loader
        self.tpm_ecdh        = tpm_ecdh
        os.makedirs(self.state_dir, exist_ok=True)

    def _headers(self) -> dict:
        return {"Authorization": f"Bearer {self.api_key}", "Accept": "application/json"}

    def _status(self) -> dict:
        # Authenticated call; backend route already exists in agents.py
        return request(
            "GET",
            self.base_url,
            f"/agents/{self.agent_id}/status",
            headers=self._headers(),
            verify_ssl=self.verify_ssl,
        )

    def ensure_ecm_cached_and_get_bytes(self, entry: dict) -> bytes:
        """
        Returns ECM plaintext bytes from RAM. If encrypted cache exists, read+decrypt it.
        Otherwise: fetch cipher → verify → unwrap CEK → decrypt to RAM → write encrypted cache → return bytes.
        Never writes plaintext to disk.
        """
        # 0) gate on status (jam on disable/revoke)
        st = self._status()
        if not st.get("ok"):
            raise SecureInferenceError("Status check failed")
        if not st.get("enabled") or st.get("revoked"):
            raise SecureInferenceError(f"Agent not enabled (status={st.get('status')})")

        # 1) validate manifest fields
        cache_name = f"{entry.get('id','ecm')}.ecm"
        c_uri  = entry.get("cipher_ecm_uri")
        c_dgst = (entry.get("cipher_ecm_digest") or "").removeprefix("sha256:")
        wrapped = entry.get("sie_wrapped_cek_b64")

        if not c_uri:
            raise SecureInferenceError("Secure ECM not available (cipher_ecm_uri missing)")
        if wrapped is None:
            # With SIE: we require a wrapped CEK (HPKE/ECIES). If you want TPM-sealed DEK-only mode, add a separate branch.
            raise SecureInferenceError("Missing wrapped CEK (sie_wrapped_cek_b64)")

        # 2) fast-path: encrypted cache hit
        if storage.has_encrypted(self.state_dir, cache_name):
            # return storage.read_encrypted(self.state_dirprepare_runtime, cache_name)
            return storage.read_encrypted(self.state_dir, cache_name)

        # 3) fetch ciphertext + verify digest
        ct = _http_get_bytes(c_uri)
        if c_dgst:
            got = _sha256_hex(ct)
            if got != c_dgst:
                raise SecureInferenceError(f"cipher_ecm_digest mismatch (got {got}, want {c_dgst})")

        # 4) unwrap CEK (HPKE/ECIES)
        try:
            cek = hpke.unwrap(
                wrapped,
                privkey_pem=self.privkey_pem,
                privkey_loader=self.privkey_loader,
                ecdh_with_ephemeral=self.tpm_ecdh,
            )
        except Exception as e:
            raise SecureInferenceError(f"SIE unwrap failed: {e}")

        # 5) decrypt to RAM; blob layout: [12-byte nonce][ciphertext+tag]
        if len(ct) < 13:
            raise SecureInferenceError("cipher_ecm too short")
        nonce, body = ct[:12], ct[12:]
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
        try:
            pt = AESGCM(cek).decrypt(nonce, body, None)
        except Exception as e:
            raise SecureInferenceError(f"SIE decrypt failed: {e}")

        # 6) store encrypted-at-rest copy (TPM-sealed DEK), never plaintext
        storage.write_encrypted(self.state_dir, cache_name, pt)

        return pt
