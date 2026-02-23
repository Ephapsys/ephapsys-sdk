# SPDX-License-Identifier: Apache-2.0
"""Ephapsys CLI entrypoint (single source of truth)."""
import os, sys, json, argparse, requests, time
from pathlib import Path
from typing import Optional

from tqdm import tqdm

# --- Version handling (from pyproject.toml / wheel metadata) ---
try:
    from importlib.metadata import version, PackageNotFoundError
except ImportError:  # Python < 3.8
    from importlib_metadata import version, PackageNotFoundError

PACKAGE_NAME = "ephapsys"
try:
    __version__ = version(PACKAGE_NAME)
except PackageNotFoundError:
    __version__ = "0.0.0"

from . import TrustedAgent, ModulatorClient
from .manifest import export_agent_manifest
from .session import login as sdk_login, auth_headers, API_URL
from .auth import get_api_key as resolve_api_key


# ---------------- Utilities ----------------
def _env(name: str, default: Optional[str]=None) -> Optional[str]:
    return os.getenv(name, default)

# FIXME: BASE URL SHOULD BE HARDCODED AND NOT RETRIEVED AS AN .ENV
def _get_base_url(args): 
    return getattr(args, "base_url", None) or _env("AOC_BASE_URL", _env("AOC_API_URL", "http://localhost:7001"))

def _get_api_key(args): 
    explicit = getattr(args, "api_key", None)
    if explicit:
        return explicit
    return resolve_api_key(
        None,
        base_url=_get_base_url(args),
        agent_instance_id=getattr(args, "agent_id", None),
        verify_ssl=_env("AOC_VERIFY_SSL", "1") != "0",
    )

def print_table(items, headers):
    if not items:
        print("No items found.")
        return
    widths = {col: max(len(col), max(len(str(row.get(col, ""))) for row in items)) for col in headers}
    header_row = "  ".join(col.ljust(widths[col]) for col in headers)
    print(header_row)
    print("-" * len(header_row))
    for row in items:
        line = "  ".join(str(row.get(col, "")).ljust(widths[col]) for col in headers)
        print(line)

# ---------------- Login ----------------
def do_login(args):
    sdk_login(args.username, args.password)
    return 0

# ---------------- Agent commands ----------------
def do_agent_verify(args):
    a = TrustedAgent(
        agent_id=args.agent_id,
        api_base=_get_base_url(args),
        api_key=_get_api_key(args)
    )
    ok, report = a.verify()
    print(json.dumps({"ok": ok, "report": report}, indent=2))
    return 0 if ok else 2

def do_agent_enable(args):
    a = TrustedAgent(
        agent_id=args.agent_id,
        api_base=_get_base_url(args),
        api_key=_get_api_key(args)
    )
    resp = a.set_status("enabled")
    print(json.dumps(resp, indent=2)); return 0

def do_agent_disable(args):
    a = TrustedAgent(
        agent_id=args.agent_id,
        api_base=_get_base_url(args),
        api_key=_get_api_key(args)
    )
    resp = a.set_status("disabled")
    print(json.dumps(resp, indent=2)); return 0

def do_agent_revoke(args):
    a = TrustedAgent(
        agent_id=args.agent_id,
        api_base=_get_base_url(args),
        api_key=_get_api_key(args)
    )
    resp = a.revoke_certificates(reason=args.reason or "unspecified")
    print(json.dumps(resp, indent=2)); return 0

def do_agent_export_manifest(args):
    models_json = json.loads(args.models)
    cert = json.loads(args.certificate) if args.certificate else {}
    policy = json.loads(args.policy) if args.policy else {}
    res = export_agent_manifest(
        agent_id=args.agent_id,
        label=args.label,
        org_id=args.org_id,
        version=args.version,
        models=models_json,
        policy=policy,
        certificate=cert,
        allowed_hosts=args.allowed_hosts or [],
        out_path=args.out
    )
    print(json.dumps(res, indent=2)); return 0

def do_agent_list(args):
    resp = requests.get(f"{API_URL}/cli/agents/list", headers=auth_headers())
    if resp.status_code != 200:
        print(f"[error] {resp.text}", file=sys.stderr)
        return 1
    data = resp.json()
    if args.json:
        print(json.dumps(data, indent=2))
    else:
        if data.get("ok"):
            items = data.get("items", [])
            print_table(items, headers=["agent_id", "label", "status"])
        else:
            print(json.dumps(data, indent=2))
    return 0

def _load_json_value(raw: Optional[str]) -> Optional[object]:
    if raw is None:
        return None
    try:
        return json.loads(raw)
    except Exception as exc:
        raise SystemExit(f"[error] invalid JSON payload: {exc}")

def _load_json_file(path: str) -> object:
    try:
        return json.loads(Path(path).read_text())
    except Exception as exc:
        raise SystemExit(f"[error] failed to read {path}: {exc}")

def do_agent_create_template(args):
    base = _get_base_url(args)
    api_key = _get_api_key(args)
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    if not args.models and not args.models_file:
        raise SystemExit("[error] --models or --models-file is required")

    if args.models_file:
        models = _load_json_file(args.models_file)
    else:
        models = _load_json_value(args.models)
    if not isinstance(models, list):
        raise SystemExit("[error] models must be a JSON array of entries")

    policy_obj = _load_json_value(args.policy) if args.policy else None
    if policy_obj is not None and not isinstance(policy_obj, dict):
        raise SystemExit("[error] policy must be a JSON object")

    body = {
        "label": args.label,
        "type": "TEMPLATE",
        "models": models,
    }
    if args.description:
        body["description"] = args.description
    if policy_obj is not None:
        body["policy"] = policy_obj

    resp = requests.post(f"{base}/agents", headers=headers, json=body)
    if resp.status_code not in (200, 201):
        print(f"[error] {resp.status_code} {resp.text}", file=sys.stderr)
        return 1
    print(json.dumps(resp.json(), indent=2))
    return 0

# ---------------- Modulation commands ----------------
def do_mod_start(args):
    mod = ModulatorClient(_get_base_url(args), _get_api_key(args))
    search = json.loads(args.search_space) if args.search_space else {}
    kpi = json.loads(args.kpi) if args.kpi else {}
    resp = mod.start_job(
        model_template_id=args.model_template_id,
        variant=args.variant,
        search_space=search,
        kpi=kpi,
        mode=args.mode,
        approved_params=json.loads(args.approved_params) if args.approved_params else None
    )
    print(json.dumps(resp, indent=2)); return 0

def do_mod_metrics(args):
    mod = ModulatorClient(_get_base_url(args), _get_api_key(args))
    metrics = json.loads(args.metrics)
    resp = mod.report_metrics(args.job_id, metrics)
    print(json.dumps(resp or {"ok": True}, indent=2)); return 0

def do_mod_next(args):
    mod = ModulatorClient(_get_base_url(args), _get_api_key(args))
    last = json.loads(args.last_metrics) if args.last_metrics else None
    resp = mod.next(args.job_id, last)
    print(json.dumps(resp, indent=2)); return 0

def do_mod_complete(args):
    mod = ModulatorClient(_get_base_url(args), _get_api_key(args))
    arts = json.loads(args.artifacts) if args.artifacts else {}
    resp = mod.complete_job(args.job_id, artifact_urls=arts, ecm_digest=args.ecm_digest)
    print(json.dumps(resp, indent=2)); return 0

# ---------------- Certificates ----------------
# def do_certs_issue_model_ecm(args):
#     pki = PkiClient(_get_base_url(args), _get_api_key(args))
#     extra = json.loads(args.extra) if args.extra else None
#     resp = pki.issue_model_ecm_cert(
#         org_id=args.org_id,
#         model_template_id=args.model_template_id,
#         rms_hash=args.rms_hash,
#         job_id=args.job_id,
#         extra=extra,
#     )
#     print(json.dumps(resp, indent=2)); return 0

# ---------------- Model commands (REST) ----------------
def do_model_register(args):
    # ---- Provider-specific validation ----
    if args.provider == "huggingface" and not args.hf_token:
        print("[error] HuggingFace provider requires --hf-token for private repos", file=sys.stderr)
        # We can't know if it's public/private at CLI level, so we warn but allow.
        # Optional: uncomment below to enforce
        # return 1

    if args.provider == "kaggle":
        if not args.kaggle_username or not args.kaggle_key:
            print("[error] Kaggle provider requires both --kaggle-username and --kaggle-key", file=sys.stderr)
            return 1

    if args.provider == "vertex":
        if not args.vertex_project:
            print("[error] Vertex provider requires --vertex-project", file=sys.stderr)
            return 1

    # ---- Force confirmation ----
    if args.force:
        confirm = input("⚠️  This will delete and re-register the model(s) for your org. Continue? [y/N]: ")
        if confirm.strip().lower() not in ("y", "yes"):
            print("[info] Operation cancelled.")
            return 0

    # ---- Build body ----
    body = {
        "provider": args.provider,
        "ids": args.ids,
        "auto_register": args.auto_register,
        "force": args.force,
    }
    
    if args.repo_id:
        body["repo_id"] = args.repo_id
    if args.revision:
        body["revision"] = args.revision
    if args.hf_token:
        body["hf_token"] = args.hf_token
    if args.kaggle_username:
        body["kaggle_username"] = args.kaggle_username
    if args.kaggle_key:
        body["kaggle_key"] = args.kaggle_key
    if args.vertex_project:
        body["vertex_project"] = args.vertex_project
    if args.vertex_region:
        body["vertex_region"] = args.vertex_region
    if args.name:
        body["name"] = args.name
    if args.status:
        body["status"] = args.status
    if args.version:
        body["version"] = args.version
    if args.model_kind:
        body["model_kind"] = args.model_kind

    # ---- Execute ----
    print(f"[info] registering {len(args.ids)} model(s) with provider={args.provider} ...")

    resp = requests.post(f"{API_URL}/cli/models/register",
                         json=body, headers=auth_headers())
    if resp.status_code != 200:
        print(f"[error] {resp.text}", file=sys.stderr)
        return 1

    result = resp.json()
    print("[info] registration completed.")
    print(json.dumps(result, indent=2))
    return 0


def do_model_list(args):
    resp = requests.get(f"{API_URL}/cli/models/list", headers=auth_headers())
    if resp.status_code != 200:
        print(f"[error] {resp.text}", file=sys.stderr)
        return 1
    data = resp.json()
    if args.json:
        print(json.dumps(data, indent=2))
    else:
        if data.get("ok"):
            items = data.get("items", [])
            print_table(items, headers=["_id", "name", "provider", "status", "type", "kind"])
        else:
            print(json.dumps(data, indent=2))
    return 0


def do_model_remove(args):
    url = f"{API_URL}/cli/models/remove/{args.provider}/{args.id}"
    resp = requests.delete(url, headers=auth_headers())
    if resp.status_code != 200:
        print(f"[error] {resp.text}", file=sys.stderr)
        return 1
    print(json.dumps(resp.json(), indent=2))
    return 0

# ---------------- Parser Builder ----------------
def build_parser():
    p = argparse.ArgumentParser(
        prog="ephapsys",
        description="Ephapsys CLI – manage agents, models, modulation jobs, and certificates"
    )
    p.add_argument("--base-url", help="AOC base URL (default from AOC_BASE_URL)")
    p.add_argument("--api-key", help="Bootstrap/runtime token")
    p.add_argument("-v", "--version", action="version", version=f"%(prog)s {__version__}")

    sub = p.add_subparsers(dest="cmd", required=True)

    # ---- Login
    login_parser = sub.add_parser("login", help="Authenticate to Ephapsys CLI backend")
    login_parser.add_argument("--username", help="Username (if omitted, prompted)")
    login_parser.add_argument("--password", help="Password (if omitted, prompted)")
    login_parser.set_defaults(func=do_login)

    # ---- Agent commands
    agent = sub.add_parser("agent", help="Agent operations")
    asub = agent.add_subparsers(dest="sub", required=True)

    a_verify = asub.add_parser("verify", help="Verify agent certificate")
    a_verify.add_argument("--agent-id", required=True)
    a_verify.set_defaults(func=do_agent_verify)

    a_enable = asub.add_parser("enable", help="Enable an agent")
    a_enable.add_argument("--agent-id", required=True)
    a_enable.set_defaults(func=do_agent_enable)

    a_disable = asub.add_parser("disable", help="Disable an agent")
    a_disable.add_argument("--agent-id", required=True)
    a_disable.set_defaults(func=do_agent_disable)

    a_revoke = asub.add_parser("revoke", help="Revoke an agent certificate")
    a_revoke.add_argument("--agent-id", required=True)
    a_revoke.add_argument("--reason")
    a_revoke.set_defaults(func=do_agent_revoke)

    a_exp = asub.add_parser("export-manifest", help="Write a non-secret .agent.json manifest")
    a_exp.add_argument("--agent-id", required=True)
    a_exp.add_argument("--label", required=True)
    a_exp.add_argument("--org-id", required=True)
    a_exp.add_argument("--version", required=True)
    a_exp.add_argument("--models", required=True, help="JSON array of model entries")
    a_exp.add_argument("--policy", help="JSON object (optional)")
    a_exp.add_argument("--certificate", help="JSON object (optional)")
    a_exp.add_argument("--allowed-hosts", nargs="*", help="List of host identifiers")
    a_exp.add_argument("--out", required=True, help="Path to write .agent.json")
    a_exp.set_defaults(func=do_agent_export_manifest)

    a_list = asub.add_parser("list", help="List all registered agents in your org")
    a_list.add_argument("--json", action="store_true", help="Output raw JSON instead of table")
    a_list.set_defaults(func=do_agent_list)

    a_create = asub.add_parser("create-template", help="Create an agent template with model policies")
    a_create.add_argument("--label", required=True, help="Template label/name")
    a_create.add_argument("--description", help="Optional description")
    group = a_create.add_mutually_exclusive_group(required=True)
    group.add_argument("--models", help="JSON array of model entries [{id,config:{type,policies}}]")
    group.add_argument("--models-file", help="Path to JSON file containing models array")
    a_create.add_argument("--policy", help="Optional JSON object of agent-level policies")
    a_create.set_defaults(func=do_agent_create_template)

    # ---- Model commands
    model = sub.add_parser("model", help="Model operations")
    msub = model.add_subparsers(dest="sub", required=True)

    mreg = msub.add_parser("register", help="Register one or more models")
    mreg.add_argument("--provider", required=True, choices=["huggingface", "kaggle", "vertex"],
                      help="Model provider")
    mreg.add_argument("--ids", nargs="+", required=True, help="Model IDs to register")
    mreg.add_argument("--repo-id", help="Optional repo ID override")
    mreg.add_argument("--revision", help="Revision/branch (default: main)")

    # HuggingFace
    mreg.add_argument("--hf-token", help="HuggingFace access token (required for private repos)")

    # Kaggle
    mreg.add_argument("--kaggle-username", help="Kaggle username (required for Kaggle datasets)")
    mreg.add_argument("--kaggle-key", help="Kaggle API key (required for Kaggle datasets)")

    # Vertex
    mreg.add_argument("--vertex-project", help="GCP project ID for Vertex AI")
    mreg.add_argument("--vertex-region", default="us-central1", help="Vertex AI region (default: us-central1)")

    # Metadata + control
    mreg.add_argument("--name", help="Friendly display name")
    mreg.add_argument("--status", help="Status (default: registered)")
    mreg.add_argument("--version", help="Model version (default: 1.0)")
    mreg.add_argument("--model-kind", help="Logical kind (language, vision, ...)")
    mreg.add_argument("--auto-register", action="store_true",
                  help="Automatically register after caching (issue provenance cert)")
    mreg.add_argument("--force", action="store_true",
                  help="Force overwrite existing model (delete + re-download)")
    mreg.set_defaults(func=do_model_register)

    mlist = msub.add_parser("list", help="List all registered models")
    mlist.add_argument("--json", action="store_true", help="Output raw JSON instead of table")
    mlist.set_defaults(func=do_model_list)

    mrm = msub.add_parser("remove", help="Remove a registered model")
    mrm.add_argument("--provider", required=True)
    mrm.add_argument("--id", required=True)
    mrm.set_defaults(func=do_model_remove)

    # ---- Modulation commands
    mod = sub.add_parser("mod", help="Modulation job commands")
    msub = mod.add_subparsers(dest="sub", required=True)

    ms = msub.add_parser("start", help="Start a modulation job")
    ms.add_argument("--model-template-id", required=True)
    ms.add_argument("--variant", required=True, choices=["additive","multiplicative","ec-ann","ec-mul"])
    ms.add_argument("--search-space", help="JSON dict of hyperparameter search space")
    ms.add_argument("--kpi", help="JSON dict objective")
    ms.add_argument("--mode", choices=["auto","manual"], default="auto")
    ms.add_argument("--approved-params", help="JSON dict (manual mode)")
    ms.set_defaults(func=do_mod_start)

    mm = msub.add_parser("metrics", help="Report modulation metrics")
    mm.add_argument("--job-id", required=True)
    mm.add_argument("--metrics", required=True, help="JSON array of metric points")
    mm.set_defaults(func=do_mod_metrics)

    mn = msub.add_parser("next", help="Request next modulation step")
    mn.add_argument("--job-id", required=True)
    mn.add_argument("--last-metrics", help="JSON array (optional)")
    mn.set_defaults(func=do_mod_next)

    mc = msub.add_parser("complete", help="Complete a modulation job")
    mc.add_argument("--job-id", required=True)
    mc.add_argument("--artifacts", help="JSON dict (urls/paths)")
    mc.add_argument("--ecm-digest", help="sha256:... digest")
    mc.set_defaults(func=do_mod_complete)

    # ---- Certificates
    # certs = sub.add_parser("certs", help="Certificate management")
    # csub = certs.add_subparsers(dest="sub", required=True)

    # cim = csub.add_parser("issue-model-ecm", help="Issue a model ECM certificate binding rms_hash")
    # cim.add_argument("--org-id", required=True)
    # cim.add_argument("--model-template-id", required=True)
    # cim.add_argument("--rms_hash", required=True)
    # cim.add_argument("--job-id")
    # cim.add_argument("--extra", help="JSON dict (optional)")
    # cim.set_defaults(func=do_certs_issue_model_ecm)

    return p


import shutil

def print_banner():
    GOLD  = "\033[38;5;220m"   # Gold accent (works everywhere on macOS Terminal)
    RESET = "\033[0m"

    # Banner text
    art = """
███████ ██████  ██   ██  █████  ██████  ███████ ██    ██ ███████ 
██      ██   ██ ██   ██ ██   ██ ██   ██ ██       ██  ██  ██      
█████   ██████  ███████ ███████ ██████  ███████   ████   ███████ 
██      ██      ██   ██ ██   ██ ██           ██    ██         ██ 
███████ ██      ██   ██ ██   ██ ██      ███████    ██    ███████ 
                                                                 
    """

    # Center and print
    width = shutil.get_terminal_size((80, 20)).columns
    centered_art = "\n".join(line.center(width) for line in art.splitlines())
    subtitle = f"Ephapsys CLI v{__version__} — Trusted AI Agents Platform"
    centered_subtitle = subtitle.center(width)

    print(f"{GOLD}{centered_art}\n{centered_subtitle}{RESET}\n")



# ---------------- Entry ----------------
def main():
    parser = build_parser()

    # show banner only if no args or "login" command
    show_banner = (
        len(sys.argv) == 1 or
        (len(sys.argv) > 1 and sys.argv[1] == "login")
    )

    if show_banner and sys.stdout.isatty():
        print_banner()

    if len(sys.argv) == 1:
        parser.print_help()
        return 0

    args = parser.parse_args()

    if hasattr(args, "func"):
        try:
            return args.func(args)
        except Exception as e:
            print(f"[error] {e}", file=sys.stderr)
            return 1

    parser.print_help()
    return 0



if __name__ == "__main__":
    sys.exit(main())
