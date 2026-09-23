#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="/workspace/venv"

if [[ ! -d "$VENV_DIR" ]]; then
  echo "Creando el ambiente compartido en $VENV_DIR"
  python -m venv --system-site-packages "$VENV_DIR"
  "$VENV_DIR/bin/python" -m pip install --upgrade pip
  "$VENV_DIR/bin/python" -m pip install -r "$REPO_DIR/requirements.txt"
else
  echo "El ambiente $VENV_DIR ya existe; no se reinstalan paquetes."
fi

"$VENV_DIR/bin/python" -m ipykernel install --user --name rsna --display-name "Python (rsna)"

echo
echo "GPU:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader || true
echo
echo "Disco de /workspace:"
df -h /workspace
