#!/usr/bin/env bash
# setup_env.sh — create this repo's own virtualenv and install its dependencies.
#
# The repo is self-contained: it owns its venv, its task modules, and its
# dispatch tooling. Nothing is sourced from a sibling checkout.
#
# Usage (identical on macOS and on Empire AI):
#     bash scripts/setup_env.sh              # create/refresh .venv
#     bash scripts/setup_env.sh --recreate   # delete .venv first
#
# Then:
#     source .venv/bin/activate

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${VENV_DIR:-$REPO_ROOT/.venv}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

if [[ "${1:-}" == "--recreate" && -d "$VENV_DIR" ]]; then
    echo "Removing existing venv: $VENV_DIR"
    rm -rf "$VENV_DIR"
fi

if [[ ! -d "$VENV_DIR" ]]; then
    echo "Creating venv at $VENV_DIR (using $PYTHON_BIN)"
    "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

"$VENV_DIR/bin/python" -m pip install --upgrade pip setuptools wheel
"$VENV_DIR/bin/python" -m pip install -r "$REPO_ROOT/requirements.txt"

echo
echo "Environment ready."
"$VENV_DIR/bin/python" - <<'PY'
import sys
print(f"  python       {sys.version.split()[0]}  ({sys.executable})")
try:
    import torch
    print(f"  torch        {torch.__version__}  cuda={torch.cuda.is_available()}")
except Exception as exc:  # torch is optional on a laptop doing analysis only
    print(f"  torch        not importable: {exc}")
try:
    import transformers
    print(f"  transformers {transformers.__version__}")
except Exception as exc:
    print(f"  transformers not importable: {exc}")
PY
echo
echo "Activate with:  source ${VENV_DIR#"$REPO_ROOT/"}/bin/activate"
