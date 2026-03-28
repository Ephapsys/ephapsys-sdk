#!/usr/bin/env bash
set -euo pipefail

cat <<'EOF'
[INFO] Robot GCP mode deploys only the brain remotely.
[INFO] The published agent/model templates are the same ones used by local mode.
[INFO] Delegating to ./push_local.sh.
EOF

exec ./push_local.sh "$@"
