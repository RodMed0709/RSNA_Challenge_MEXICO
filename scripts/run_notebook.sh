#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Uso: bash scripts/run_notebook.sh <notebook.ipynb> <run_name> [args de papermill]" >&2
  exit 2
fi
if [[ -z "${RSNA_USER:-}" ]]; then
  echo "Falta RSNA_USER. Define tu nombre primero: export RSNA_USER=ana" >&2
  exit 2
fi

NOTEBOOK="$1"
RUN_NAME="$2"
shift 2

if [[ ! -f "$NOTEBOOK" ]]; then
  echo "No existe el notebook: $NOTEBOOK" >&2
  exit 2
fi
if [[ "$RUN_NAME" == *"/"* || "$RUN_NAME" == *"\\"* || "$RUN_NAME" == "." || "$RUN_NAME" == ".." ]]; then
  echo "run_name no puede contener rutas: $RUN_NAME" >&2
  exit 2
fi

RUN_DIR="/workspace/users/$RSNA_USER/runs/$RUN_NAME"
mkdir -p "$RUN_DIR"
LOG_FILE="$RUN_DIR/train.log"
OUTPUT_NOTEBOOK="$RUN_DIR/executed.ipynb"

nohup /workspace/venv/bin/python -m papermill \
  "$NOTEBOOK" "$OUTPUT_NOTEBOOK" \
  -k rsna -p RUN_NAME "$RUN_NAME" "$@" \
  >"$LOG_FILE" 2>&1 </dev/null &
PID=$!

echo "Entrenamiento iniciado. PID: $PID"
echo "Salida: $OUTPUT_NOTEBOOK"
echo "Log: $LOG_FILE"
echo "Para verlo: tail -f '$LOG_FILE'"
