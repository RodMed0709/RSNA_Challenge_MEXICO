#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"

if ! "$VENV_DIR/bin/python" -c "import sys" 2>/dev/null; then
  echo "Creando el ambiente $VENV_DIR (Python $PY_VER)"
  rm -rf "$VENV_DIR"
  python3 -m venv --system-site-packages "$VENV_DIR"
  "$VENV_DIR/bin/python" -m pip install --upgrade pip
fi
"$VENV_DIR/bin/python" -m pip install -q -r "$REPO_DIR/requirements.txt"
"$VENV_DIR/bin/python" -m ipykernel install --user --name rsna --display-name "Python (rsna)" >/dev/null

cd "$REPO_DIR"
"$VENV_DIR/bin/nbstripout" --install
git config user.name >/dev/null || git config user.name "$RSNA_USER"
git config user.email >/dev/null || git config user.email "$RSNA_USER@rsna.local"

echo "Usuario: ${RSNA_USER:-sin definir} · ambiente: $VENV_DIR · kernel: Python (rsna)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
