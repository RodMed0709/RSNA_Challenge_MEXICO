#!/usr/bin/env bash
# L5v3 (sub056_3): generate 2 new LLM readings -> build L5v3 -> validate -> launch sub056_3 fold 0 on a GPU pod.
# Run from rsna-cron (has /root/.openai_key and ssh to pods). See README.md.
# Usage: bash run_l5v3.sh [--dry-run] [--launch-only] [--pod PORT HOST] [--budget USD]
set -u
ROOT=/workspace/users/rodrigo/l5v3
PY=/workspace/venv312/bin/python
LBL=/workspace/data/labels/ours
DRY=0; LAUNCH_ONLY=0; PORT=""; HOST=""; BUDGET=29
while [ $# -gt 0 ]; do case "$1" in
  --dry-run) DRY=1;; --launch-only) LAUNCH_ONLY=1;;
  --pod) PORT="$2"; HOST="$3"; shift 2;; --budget) BUDGET="$2"; shift;;
  *) echo "unknown arg $1"; exit 2;; esac; shift; done
mkdir -p "$ROOT/logs"; LOG="$ROOT/logs/run_$(date -u +%Y%m%d_%H%M%S)$([ $DRY = 1 ] && echo _dry).log"
exec > >(tee -a "$LOG") 2>&1
st(){ echo "$(date -u +%FT%TZ) $*" | tee -a "$ROOT/STATUS"; }
D=""; [ $DRY = 1 ] && D="--dry-run"
st "run_l5v3 start dry=$DRY launch_only=$LAUNCH_ONLY pod=${PORT:-none} budget=$BUDGET"
if [ $DRY = 0 ] && [ ! -f "$ROOT/GO" ]; then st "NO GO marker ($ROOT/GO): stop"; exit 1; fi
[ -f /root/.openai_key ] || { [ $DRY = 1 ] && st "WARN no /root/.openai_key here" || { st "NO KEY /root/.openai_key"; exit 1; }; }

if [ $LAUNCH_ONLY = 0 ]; then
  rm -f "$ROOT/L5V3_NO_QUOTA" "$ROOT/L5V3_OVER_BUDGET"
  ONLY=""
  for step in pilot run; do
    "$PY" "$ROOT/generar_L5v3.py" --root "$ROOT" $step $D --budget "$BUDGET" $ONLY; rc=$?
    [ $rc = 3 ] && { st "STOP insufficient quota (L5V3_NO_QUOTA)"; exit 3; }
    if [ $rc = 4 ] && [ $step = pilot ]; then
      # reduction rule (SPEC): if F+G projects over budget, run F_focus3 only (if F alone fits)
      FP=$("$PY" -c "import json;print(json.load(open('$ROOT/pilot_projection.json'))['F_focus3'])")
      if "$PY" -c "import sys;sys.exit(0 if $FP <= $BUDGET else 1)"; then
        rm -f "$ROOT/L5V3_OVER_BUDGET"; ONLY="--only F_focus3"; st "pilot over budget: reduced to F_focus3 only (proj USD $FP)"; continue
      fi
    fi
    [ $rc = 4 ] && { st "STOP over budget (L5V3_OVER_BUDGET)"; exit 4; }
    [ $rc != 0 ] && { st "STOP $step failed rc=$rc"; exit $rc; }
    st "$step ok"
  done
  if [ $DRY = 1 ]; then
    st "dry-run: would build $LBL/L5v3_<date>.csv (+ copy L5v3.csv), validate, touch L5V3_READY"
  else
    OUT="$LBL/L5v3_$(date -u +%Y%m%d).csv"
    "$PY" "$ROOT/generar_L5v3.py" --root "$ROOT" build "$OUT" || { st "STOP build failed"; exit 5; }
    cp "$OUT" "$LBL/L5v3.csv"
    (cd "$ROOT" && "$PY" validar_L5v3.py "$ROOT/data" "$OUT" "$ROOT/L5v3_validacion.json" "$LBL/L5v2_20261012.csv")
    rc=$?; [ $rc != 0 ] && { touch "$ROOT/L5V3_UNHEALTHY"; st "STOP L5v3 unhealthy rc=$rc (no training)"; exit 6; }
    touch "$ROOT/L5V3_READY"; st "L5v3 ready: $OUT (copy $LBL/L5v3.csv)"
  fi
fi

[ -z "$PORT" ] && { st "no --pod given: training not launched"; exit 0; }
[ $DRY = 0 ] && [ ! -f "$ROOT/L5V3_READY" ] && { st "L5V3_READY missing: training not launched"; exit 1; }
SSH="ssh -o BatchMode=yes -o StrictHostKeyChecking=no -o ConnectTimeout=15 -p $PORT root@$HOST"
$SSH 'pgrep -f "python -m [p]apermill" >/dev/null && exit 3; m=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1); [ "$m" -lt 1000 ] || exit 4; [ -f /root/corpus96_384/ids.npy ] || exit 5; exit 0'
rc=$?; [ $rc != 0 ] && { [ $DRY = 0 ] && touch "$ROOT/L5V3_POD_BUSY"; st "pod $PORT $HOST not free (rc=$rc): retry with --launch-only --pod ..."; exit 7; }
if [ $DRY = 1 ]; then st "dry-run: pod $PORT $HOST is free; would run: setsid nohup bash $ROOT/train_sub056_3.sh"; exit 0; fi
rm -f "$ROOT/L5V3_POD_BUSY"
$SSH "setsid nohup bash $ROOT/train_sub056_3.sh > /root/train_sub056_3.log 2>&1 < /dev/null & sleep 3; pgrep -af '[t]rain_sub056_3'"
st "launched sub056_3 on $PORT $HOST (status: /workspace/users/rodrigo/wt_sub056_3_L5v3/results/sub056_3_L5v3/chain_status)"
