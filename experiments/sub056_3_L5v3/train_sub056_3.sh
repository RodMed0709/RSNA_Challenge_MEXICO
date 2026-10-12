#!/usr/bin/env bash
# runs ON the GPU pod: smoke -> fold 0 of sub056_3 (exp056 combo with L5v3 as 5th source)
cd /workspace/users/rodrigo/wt_sub056_3_L5v3 || exit 1
export RSNA_USER=rodrigo
source _env.sh
OUT=$REPO_DIR/results/sub056_3_L5v3; mkdir -p "$OUT"
st(){ echo "$(date -u +%FT%TZ) $(hostname) $*" >> "$OUT/chain_status"; }
[ -f /workspace/data/labels/ours/L5v3.csv ] || { st "NO_L5V3_CSV"; exit 1; }
cd experiments/sub056_3_L5v3
T=$(date +%Y%m%d_%H%M%S); st "smoke $T"
"$VENV_DIR/bin/python" -m papermill sub056_3_L5v3.ipynb "$OUT/executed_smoke_$T.ipynb" -k rsna --log-output \
  -p SMOKE True > "$OUT/run_smoke_$T.log" 2>&1 || { st "SMOKE_FAIL"; exit 1; }
[ -f "$OUT/metrics.json" ] && mv "$OUT/metrics.json" "$OUT/metrics_smoke.json" || { st "SMOKE_NO_METRICS"; exit 1; }
T=$(date +%Y%m%d_%H%M%S); st "SMOKE_OK; fold0 $T"
"$VENV_DIR/bin/python" -m papermill sub056_3_L5v3.ipynb "$OUT/executed_$T.ipynb" -k rsna --log-output \
  -p SMOKE False -y 'FOLDS: [0]' > "$OUT/run_$T.log" 2>&1 || { st "F0_FAIL"; exit 1; }
st "F0_DONE"
