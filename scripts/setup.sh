#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
setup.sh --local|--testpypi|--pypi [options]

Install Ephapsys SDK into a dedicated virtualenv for repeatable testing.

Modes:
  --local       Install from the local repo working tree (editable mode)
  --testpypi    Install from TestPyPI
  --pypi        Install from PyPI

Options:
  --venv PATH       Virtualenv path. Default: ./venvs/sdk
  --version VER     Pin a specific package version for --testpypi or --pypi
  --extras LIST     Extra groups, e.g. modulation or modulation,audio,vision
  --force-reinstall Reinstall even if already present
  --verbose         Show full pip output
  -h, --help        Show this help

Examples:
  ./setup.sh --local
  ./setup.sh --testpypi
  ./setup.sh --testpypi --version 0.2.21
  ./setup.sh --pypi --version 0.2.20 --extras modulation

Notes:
  - This script does not activate the virtualenv in your current shell.
  - After it completes, run: source <venv>/bin/activate
EOF
}

MODE=""
VENV_PATH=""
VERSION=""
EXTRAS=""
FORCE_REINSTALL=0
VERBOSE=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --local)
      MODE="local"
      shift
      ;;
    --testpypi)
      MODE="testpypi"
      shift
      ;;
    --pypi)
      MODE="pypi"
      shift
      ;;
    --venv)
      VENV_PATH="$2"
      shift 2
      ;;
    --version)
      VERSION="$2"
      shift 2
      ;;
    --extras)
      EXTRAS="$2"
      shift 2
      ;;
    --force-reinstall)
      FORCE_REINSTALL=1
      shift
      ;;
    --verbose)
      VERBOSE=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "[ERROR] Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ -z "$MODE" ]]; then
  usage
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
SDK_DIR="$REPO_DIR/sdk/python"
DEFAULT_VENV="$SCRIPT_DIR/venvs/sdk"
VENV_PATH="${VENV_PATH:-$DEFAULT_VENV}"
PYTHON_BIN="$VENV_PATH/bin/python"
PIP_FLAGS=()

if [[ "$VERBOSE" -eq 0 ]]; then
  PIP_FLAGS+=("-q")
  export PIP_DISABLE_PIP_VERSION_CHECK=1
fi

if [[ ! -d "$VENV_PATH" ]]; then
  echo "[INFO] Creating virtualenv at $VENV_PATH"
  python3 -m venv "$VENV_PATH"
fi

echo "[INFO] Using virtualenv: $VENV_PATH"
"$PYTHON_BIN" -m pip "${PIP_FLAGS[@]}" install -U pip setuptools wheel

PIP_INSTALL_ARGS=()
if [[ "$FORCE_REINSTALL" -eq 1 ]]; then
  PIP_INSTALL_ARGS+=("--force-reinstall")
fi

run_pip_install() {
  if [[ "$FORCE_REINSTALL" -eq 1 ]]; then
    "$PYTHON_BIN" -m pip "${PIP_FLAGS[@]}" install --force-reinstall "$@"
  else
    "$PYTHON_BIN" -m pip "${PIP_FLAGS[@]}" install "$@"
  fi
}

resolve_latest_version() {
  local repo="$1"
  local url=""
  case "$repo" in
    testpypi)
      url="https://test.pypi.org/pypi/ephapsys/json"
      ;;
    pypi)
      url="https://pypi.org/pypi/ephapsys/json"
      ;;
    *)
      echo "[ERROR] Unknown package index: $repo" >&2
      exit 1
      ;;
  esac

  if command -v curl >/dev/null 2>&1; then
    local payload
    payload="$(curl -fsSL "$url")"
    JSON_PAYLOAD="$payload" python3 - <<'PY'
import json
import os

try:
    data = json.loads(os.environ["JSON_PAYLOAD"])
except Exception as exc:
    raise SystemExit(f"[ERROR] Failed to parse package metadata JSON: {exc}")

version = (data.get("info") or {}).get("version")
if not version:
    raise SystemExit("[ERROR] No version metadata returned by package index")
print(version)
PY
    return
  fi

  REPO_URL="$url" python3 - <<'PY'
import json
import os
import sys
from urllib.request import urlopen

url = os.environ["REPO_URL"]
try:
    with urlopen(url, timeout=10) as resp:
        data = json.load(resp)
except Exception as exc:
    raise SystemExit(f"[ERROR] Failed to resolve latest version from {url}: {exc}")

version = (data.get("info") or {}).get("version")
if not version:
    raise SystemExit(f"[ERROR] No version metadata returned from {url}")
print(version)
PY
}

build_package_spec() {
  local version="$1"
  if [[ -n "$EXTRAS" ]]; then
    if [[ -n "$version" ]]; then
      printf 'ephapsys[%s]==%s\n' "$EXTRAS" "$version"
    else
      printf 'ephapsys[%s]\n' "$EXTRAS"
    fi
  else
    if [[ -n "$version" ]]; then
      printf 'ephapsys==%s\n' "$version"
    else
      printf 'ephapsys\n'
    fi
  fi
}

case "$MODE" in
  local)
    TARGET="$SDK_DIR"
    if [[ ! -d "$TARGET" ]]; then
      echo "[ERROR] SDK directory not found at $TARGET" >&2
      exit 1
    fi
    if [[ -n "$EXTRAS" ]]; then
      TARGET="${TARGET}[${EXTRAS}]"
    fi
    echo "[INFO] Installing local SDK from $SDK_DIR"
    run_pip_install -e "$TARGET"
    ;;
  testpypi)
    if [[ -z "$VERSION" ]]; then
      VERSION="$(resolve_latest_version testpypi)"
      echo "[INFO] Resolved latest TestPyPI version: $VERSION"
    fi
    PACKAGE="$(build_package_spec "$VERSION")"
    echo "[INFO] Installing $PACKAGE from TestPyPI"
    run_pip_install \
      -i https://test.pypi.org/simple/ \
      --extra-index-url https://pypi.org/simple \
      --no-cache-dir \
      "$PACKAGE"
    ;;
  pypi)
    if [[ -z "$VERSION" ]]; then
      VERSION="$(resolve_latest_version pypi)"
      echo "[INFO] Resolved latest PyPI version: $VERSION"
    fi
    PACKAGE="$(build_package_spec "$VERSION")"
    echo "[INFO] Installing $PACKAGE from PyPI"
    run_pip_install --no-cache-dir "$PACKAGE"
    ;;
esac

echo "[INFO] Verifying install"
"$PYTHON_BIN" - <<'PY'
import sys
try:
    import importlib.metadata as m
    print("python =", sys.executable)
    print("ephapsys =", m.version("ephapsys"))
except Exception as exc:
    raise SystemExit(f"[ERROR] Failed to verify ephapsys install: {exc}")
PY

echo
echo "[DONE] SDK environment ready."
echo "Activate with:"
echo "  source $VENV_PATH/bin/activate"
