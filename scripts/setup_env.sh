#!/usr/bin/env bash
# setup_env.sh — create this repo's own virtualenv and install its dependencies.
#
# The repo is self-contained: it owns its venv, its task modules, and its
# dispatch tooling. Nothing is sourced from a sibling checkout.
#
# Usage (identical on macOS and on Empire AI):
#     bash scripts/setup_env.sh              # create/refresh .venv
#     bash scripts/setup_env.sh --recreate   # delete .venv first
#     bash scripts/setup_env.sh --s1         # .venv-s1 from requirements-s1.txt
#     bash scripts/setup_env.sh --s1 --gpu   # ... plus requirements-s1-gpu.txt
#     bash scripts/setup_env.sh --vllm       # .venv-vllm from requirements-vllm.txt
#
# Then:
#     source .venv/bin/activate

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
RECREATE=0; S1=0; GPU=0; VLLM=0
for arg in "$@"; do
    case "$arg" in
        --recreate) RECREATE=1 ;;
        --s1) S1=1 ;;
        --gpu) GPU=1 ;;
        --vllm) VLLM=1 ;;
        *) echo "unknown argument: $arg" >&2; exit 2 ;;
    esac
done
if (( GPU && !S1 )); then echo "--gpu needs --s1" >&2; exit 2; fi
if (( VLLM && S1 )); then echo "--vllm and --s1 are separate venvs" >&2; exit 2; fi
if (( VLLM )); then
    VENV_DIR="${VENV_DIR:-$REPO_ROOT/.venv-vllm}"
    REQS=("$REPO_ROOT/requirements-vllm.txt")
elif (( S1 )); then
    VENV_DIR="${VENV_DIR:-$REPO_ROOT/.venv-s1}"
    REQS=("$REPO_ROOT/requirements-s1.txt")
    (( GPU )) && REQS+=("$REPO_ROOT/requirements-s1-gpu.txt")
else
    VENV_DIR="${VENV_DIR:-$REPO_ROOT/.venv}"
    REQS=("$REPO_ROOT/requirements.txt")
fi

if (( RECREATE )) && [[ -d "$VENV_DIR" ]]; then
    echo "Removing existing venv: $VENV_DIR"
    rm -rf "$VENV_DIR"
fi

if [[ ! -d "$VENV_DIR" ]]; then
    echo "Creating venv at $VENV_DIR (using $PYTHON_BIN)"
    "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

"$VENV_DIR/bin/python" -m pip install --upgrade pip setuptools wheel
for req in "${REQS[@]}"; do
    "$VENV_DIR/bin/python" -m pip install -r "$req"
done

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
