#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

if [[ $# -lt 1 ]]; then
  echo "Uso: bash scripts/run_notebook.sh experiments/expNNN_tag/expNNN_tag.ipynb [-p FOLDS '[0,1,2,3,4]']" >&2
  exit 2
fi
NOTEBOOK="$(realpath "$1")"
shift
EXP="$(basename "$NOTEBOOK" .ipynb)"
OUT_DIR="$REPO_DIR/results/$EXP"
mkdir -p "$OUT_DIR"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG="$OUT_DIR/run_$STAMP.log"

cd "$(dirname "$NOTEBOOK")"
nohup "$VENV_DIR/bin/python" -m papermill "$NOTEBOOK" "$OUT_DIR/executed_$STAMP.ipynb" -k rsna --log-output "$@" \
  >"$LOG" 2>&1 </dev/null &
echo "$!" > "$OUT_DIR/pid"
echo "Corriendo $EXP (PID $!)"
echo "Avance: tail -f $LOG"
