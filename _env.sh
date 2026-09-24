REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY_VER="$(python3 -c 'import sys; print(f"{sys.version_info[0]}{sys.version_info[1]}")')"
if [[ "$PY_VER" == "311" ]]; then VENV_DIR="/workspace/venv"; else VENV_DIR="/workspace/venv$PY_VER"; fi
if [[ -z "${RSNA_USER:-}" && "$(dirname "$REPO_DIR")" == "/workspace/users" ]]; then
  RSNA_USER="$(basename "$REPO_DIR")"
fi
export REPO_DIR VENV_DIR RSNA_USER
