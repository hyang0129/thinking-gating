#!/usr/bin/env bash
# remote_setup.sh — SessionStart hook for Claude Code cloud sessions
# (claude.ai/code). Builds .venv with CPU-only torch and puts it on PATH.
# No-op everywhere else: local machines and the cluster use setup_env.sh.
set -euo pipefail

[[ "${CLAUDE_CODE_REMOTE:-}" == "true" ]] || exit 0

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# numpy<2 (requirements.txt) has no wheels past Python 3.12.
PYTHON_BIN="${PYTHON_BIN:-$(command -v python3.12 || command -v python3.11 || command -v python3)}"
export PYTHON_BIN

# The container has no GPU; the CPU wheel index avoids ~3 GB of CUDA wheels.
# If the network policy blocks it, pip falls back to PyPI (slower, still works).
PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cpu \
    bash "$REPO_ROOT/scripts/setup_env.sh" >&2

if [[ -n "${CLAUDE_ENV_FILE:-}" ]]; then
    echo "export PATH=\"$REPO_ROOT/.venv/bin:\$PATH\"" >> "$CLAUDE_ENV_FILE"
fi
