#!/usr/bin/env bash
set -xe

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
SDK_DIR="${SDK_DIR:-$REPO_DIR/sdk/python}"

cd "$SDK_DIR"

rm -rf dist

python3 -m pip install -U pip setuptools wheel build
python3 -m build --wheel

python3 -m pip uninstall -y ephapsys || true
python3 -m pip install dist/ephapsys-*.whl

python3 - <<'PY'
import sys, importlib.metadata as m
print("python =", sys.executable)
print("ephapsys =", m.version("ephapsys"))
from ephapsys import TrustedAgent
print("TrustedAgent import OK")
PY
