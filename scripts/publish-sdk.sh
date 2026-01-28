#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
publish-sdk.sh [--verbose] --dev|--stag|--prod

  --dev         Rebuild the wheel locally and reinstall (delegates to cleanup_sdk.sh).
  --stag        Build fresh artifacts and upload to TestPyPI.
  --prod        Build fresh artifacts and upload to PyPI (prompts for confirmation).
  --verbose     Print full command output (otherwise pip installs are quieter).

Set PUBLISH_FORCE=1 to skip the production confirmation prompt.
Set PUBLISH_VERSION to force a specific version; otherwise we auto-bump the patch
to avoid collisions with what's already on TestPyPI/PyPI.
Legacy aliases: --staging (TestPyPI) and --production (PyPI) are still accepted.
USAGE
}

VERBOSE=0
ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --verbose) VERBOSE=1; shift ;;
    *) ARGS+=("$1"); shift ;;
  esac
done

if [[ ${#ARGS[@]} -ne 1 ]]; then
  usage
  exit 1
fi

MODE="${ARGS[0]}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

if [[ -n "${NO_COLOR:-}" ]]; then
  CYAN=""; GREEN=""; YELLOW=""; RESET=""
else
  CYAN="\033[36m"; GREEN="\033[32m"; YELLOW="\033[33m"; RESET="\033[0m"
fi
info() { printf "${CYAN}➜ %s${RESET}\n" "$*"; }
ok()   { printf "${GREEN}✔ %s${RESET}\n" "$*"; }
warn() { printf "${YELLOW}⚠ %s${RESET}\n" "$*"; }

resolve_sdk_dir() {
  if [[ -n "${SDK_DIR:-}" ]]; then
    echo "$SDK_DIR"
    return
  fi

  if [[ -d "$REPO_DIR/sdk/python" ]]; then
    echo "$REPO_DIR/sdk/python"
    return
  fi

  echo "SDK directory not found. Set SDK_DIR to the python SDK path." >&2
  exit 1
}

SDK_DIR="$(resolve_sdk_dir)"
PIP_FLAGS=()
if [[ "$VERBOSE" -eq 0 ]]; then
  export PIP_DISABLE_PIP_VERSION_CHECK=1
  PIP_FLAGS+=("-q")
fi

ensure_toml_writer() {
  python3 - <<'PY'
try:
    import tomli_w  # type: ignore
except ModuleNotFoundError:
    raise SystemExit(1)
PY
  if [[ $? -ne 0 ]]; then
    python3 -m pip "${PIP_FLAGS[@]}" install tomli_w >/dev/null 2>&1 || python3 -m pip install tomli_w
  fi
}

get_sdk_version() {
  SDK_DIR_ENV="$SDK_DIR" python3 - <<'PY'
import os
import pathlib
try:
    import tomllib as toml
except ModuleNotFoundError:
    import tomli as toml

sdk_dir = os.environ["SDK_DIR_ENV"]
pyproject = pathlib.Path(sdk_dir, "pyproject.toml").read_text()
data = toml.loads(pyproject)
print(data.get("project", {}).get("version", "unknown"))
PY
}

write_sdk_version() {
  local new_ver="$1"
  info "Setting version to $new_ver"
  ensure_toml_writer
  SDK_DIR_ENV="$SDK_DIR" NEW_VER="$new_ver" python3 - <<'PY'
import os
import pathlib
try:
    import tomllib as toml
except ModuleNotFoundError:
    import tomli as toml
import tomli_w

sdk_dir = os.environ["SDK_DIR_ENV"]
new_ver = os.environ["NEW_VER"]
path = pathlib.Path(sdk_dir, "pyproject.toml")
data = toml.loads(path.read_text())
data.setdefault("project", {})["version"] = new_ver
path.write_text(tomli_w.dumps(data))
print(f"🔢 Set version to {new_ver} in {path}")
PY
}

auto_bump_version() {
  local repo="$1"  # "testpypi" or "pypi"
  local override="${PUBLISH_VERSION:-}"
  local current
  current="$(get_sdk_version)"

  # If user provided a version, respect it.
  if [[ -n "$override" ]]; then
    write_sdk_version "$override"
    echo "$override"
    return
  fi

  # Try to fetch remote version; fall back to local if unreachable.
  local remote_url
  if [[ "$repo" == "testpypi" ]]; then
    remote_url="https://test.pypi.org/pypi/ephapsys/json"
  else
    remote_url="https://pypi.org/pypi/ephapsys/json"
  fi

  local remote
  remote="$(REMOTE_URL="$remote_url" python3 - <<'PY'
import json, os, sys
from urllib.request import urlopen

url = os.environ["REMOTE_URL"]
try:
    with urlopen(url, timeout=5) as resp:
        data = json.load(resp)
    print(data.get("info", {}).get("version", ""))
except Exception:
    print("")
    sys.exit(0)
PY
)"

  # Simple semantic-ish compare and bump patch.
  bump_patch() {
    python3 - <<'PY'
import os, sys
v = os.environ["VER"]
parts = v.split(".")
if not parts:
    print("0.0.1")
    sys.exit(0)
try:
    nums = [int(p) for p in parts]
except ValueError:
    # fallback
    print(v + ".1")
    sys.exit(0)
nums[-1] += 1
print(".".join(str(n) for n in nums))
PY
  }

  version_gte() {
    python3 - <<'PY'
import os, sys
a = os.environ["A"]
b = os.environ["B"]
def parse(v):
    try:
        return [int(x) for x in v.split(".")]
    except ValueError:
        return [0]

pa, pb = parse(a), parse(b)
for x, y in zip(pa, pb):
    if x > y:
        print("1")
        sys.exit(0)
    if x < y:
        print("0")
        sys.exit(0)
print("1" if len(pa) >= len(pb) else "0")
PY
  }

  local target="$current"
  info "Remote version (${repo}): ${remote:-none}; local pyproject: ${current:-unknown}"
  if [[ -n "$remote" ]]; then
    if [[ "$(A="$remote" B="$current" version_gte)" == "1" ]]; then
      target="$(VER="$remote" bump_patch)"
    else
      target="$(VER="$current" bump_patch)"
    fi
  else
    target="$(VER="$current" bump_patch)"
  fi

  write_sdk_version "$target"
  echo "$target"
}

ensure_twine_credentials() {
  local section="$1"
  local repo_url="$2"
  local signup_url="$3"
  local guide_url="$4"

  if [[ -n "${TWINE_API_KEY:-}" ]]; then
    export TWINE_USERNAME="${TWINE_USERNAME:-__token__}"
    export TWINE_PASSWORD="${TWINE_API_KEY}"
    return
  fi

  if [[ -n "${TWINE_PASSWORD:-}" && -z "${TWINE_USERNAME:-}" ]]; then
    export TWINE_USERNAME="__token__"
  fi

  if [[ -n "${TWINE_USERNAME:-}" && -n "${TWINE_PASSWORD:-}" ]]; then
    return
  fi

  local pypirc="$HOME/.pypirc"
  if [[ -f "$pypirc" ]] && grep -q "^\[$section\]" "$pypirc"; then
    return
  fi

  echo "Twine credentials for '$section' were not found."
  if command -v open >/dev/null 2>&1; then
    open "$signup_url"
    open "$guide_url"
  elif command -v xdg-open >/dev/null 2>&1; then
    xdg-open "$signup_url"
    xdg-open "$guide_url"
  else
    echo "Visit:"
    echo "  $signup_url"
    echo "  $guide_url"
  fi

  read -rp "Enter username for $section (use __token__ for API tokens): " user
  while [[ -z "$user" ]]; do
    read -rp "Username cannot be empty. Enter username for $section: " user
  done
  read -rsp "Enter password/token for $section: " pass
  echo

  PY_SECTION="$section" PY_REPO="$repo_url" PY_USERNAME="$user" PY_PASSWORD="$pass" python3 - <<'PY'
import configparser, os, pathlib
section = os.environ["PY_SECTION"]
repo = os.environ["PY_REPO"]
username = os.environ["PY_USERNAME"]
password = os.environ["PY_PASSWORD"]
path = pathlib.Path(os.path.expanduser("~/.pypirc"))
config = configparser.RawConfigParser()
config.read(path)
if not config.has_section("distutils"):
    config.add_section("distutils")
servers = config.get("distutils", "index-servers", fallback="")
entries = [line.strip() for line in servers.splitlines() if line.strip()]
if section not in entries:
    entries.append(section)
config.set("distutils", "index-servers", "\n    " + "\n    ".join(entries))
if not config.has_section(section):
    config.add_section(section)
config.set(section, "repository", repo)
config.set(section, "username", username)
config.set(section, "password", password)
with path.open("w") as fh:
    config.write(fh)
PY

  chmod 600 "$pypirc"
  echo "Saved credentials for $section to $pypirc"
}

run_cleanup() {
  bash "$SCRIPT_DIR/cleanup-sdk.sh"
}

prepare_dist() {
  pushd "$SDK_DIR" >/dev/null
  rm -rf build dist
  python3 -m pip "${PIP_FLAGS[@]}" install -U pip setuptools wheel build twine
  if [[ "$VERBOSE" -eq 0 ]]; then
    local log
    log="$(mktemp "/tmp/publish-sdk-build.XXXXXX.log")"
    if ! PYTHONWARNINGS=ignore::DeprecationWarning python3 -m build >"$log" 2>&1; then
      warn "Build failed. See $log"
      cat "$log"
      exit 1
    fi
    if ! PYTHONWARNINGS=ignore::DeprecationWarning python3 -m twine check dist/* >"$log" 2>&1; then
      warn "twine check failed. See $log"
      cat "$log"
      exit 1
    fi
    rm -f "$log"
  else
    python3 -m build
    python3 -m twine check dist/*
  fi
  popd >/dev/null
}

upload_dist() {
  local repo_flag="$1"
  pushd "$SDK_DIR" >/dev/null
  python3 -m twine upload $repo_flag dist/*
  popd >/dev/null
}

confirm_production() {
  if [[ "${PUBLISH_FORCE:-0}" == "1" ]]; then
    return
  fi
  read -rp "About to upload to PyPI. Type 'release' to continue: " reply
  if [[ "$reply" != "release" ]]; then
    echo "Aborted."
    exit 1
  fi
}

case "$MODE" in
  --dev)
    info "Running local rebuild..."
    run_cleanup
    version="$(get_sdk_version)"
    ok "Local rebuild complete for ephapsys v${version}. Update NEXT_PUBLIC_SDK_VERSION if needed."
    ;;
  --stag|--staging)
    info "Publishing to TestPyPI..."
    ensure_twine_credentials \
      "testpypi" \
      "https://test.pypi.org/legacy/" \
      "https://test.pypi.org/account/register/" \
      "https://packaging.python.org/en/latest/guides/using-testpypi/"
    bumped="$(auto_bump_version testpypi)"
    prepare_dist
    upload_dist "--repository testpypi"
    version="$(get_sdk_version)"
    ok "Published ephapsys v${version} to TestPyPI."
    info "View: https://test.pypi.org/project/ephapsys/${version}/"
    info "Remember to bump NEXT_PUBLIC_SDK_VERSION to ${version}."
    ;;
  --prod|--production)
    info "Publishing to PyPI..."
    confirm_production
    ensure_twine_credentials \
      "pypi" \
      "https://upload.pypi.org/legacy/" \
      "https://pypi.org/account/register/" \
      "https://packaging.python.org/en/latest/guides/publishing-package/"
    bumped="$(auto_bump_version pypi)"
    prepare_dist
    upload_dist ""
    version="$(get_sdk_version)"
    ok "Published ephapsys v${version} to PyPI."
    info "View: https://pypi.org/project/ephapsys/${version}/"
    info "Update NEXT_PUBLIC_SDK_VERSION to ${version} in frontend env files."
    ;;
  *)
    usage
    exit 1
    ;;
esac
