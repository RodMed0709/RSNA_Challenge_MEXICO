#!/usr/bin/env bash
set -Eeuo pipefail

POD_ID=${1:?pod id is required}
POD_NAME=${2:?pod name is required}
WORKER=${3:?worker a or b is required}
[[ "$WORKER" == "a" || "$WORKER" == "b" ]] || { echo "worker must be a or b" >&2; exit 2; }

REPO=/workspace/users/rodrigo/wt_stress
EXP=$REPO/experiments/exp091_stress_v1
RESULTS=$REPO/results/exp091_stress_v1
ORQ=/workspace/users/rodrigo/orq/stress
PYTHON=${PYTHON:-python}
DRY_RUN=${DRY_RUN:-0}
API_KEY_FILE=/root/.runpod_key
TOTAL_INFERENCE_MODELS=43

mkdir -p "$ORQ/done" "$RESULTS"
touch "$ORQ/status.log" "$ORQ/HANDOFF.md"

timestamp() { date -u +'%Y-%m-%dT%H:%M:%SZ'; }

status() {
    local message=$1
    printf '%s worker=%s %s\n' "$(timestamp)" "$WORKER" "$message" | tee -a "$ORQ/status.log"
    (
        flock -x 9
        printf -- '- `%s` worker %s: %s\n' "$(timestamp)" "$WORKER" "$message" >> "$ORQ/HANDOFF.md"
    ) 9>"$ORQ/handoff.lock"
}

task_spend() {
    "$PYTHON" - "$ORQ" <<'PY'
import sys, time
from pathlib import Path

root = Path(sys.argv[1])
now = time.time()
seconds = 0.0
for worker in ("a", "b"):
    path = root / f"pod_{worker}_start_epoch"
    if path.exists():
        seconds += max(0.0, now - float(path.read_text().strip()))
print(seconds / 3600.0 * 0.72)
PY
}

budget_guard() {
    local spent completed projected
    spent=$(task_spend)
    completed=$(find "$ORQ/done" -type f -name '*.done' | wc -l)
    projected=$(
        "$PYTHON" - "$spent" "$completed" "$TOTAL_INFERENCE_MODELS" <<'PY'
import sys
spent, completed, total = float(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
print(spent / completed * total if completed >= 8 else 0.0)
PY
    )
    if "$PYTHON" - "$spent" "$projected" <<'PY'
import sys
spent, projected = map(float, sys.argv[1:])
raise SystemExit(0 if spent >= 13.50 or projected > 14.00 else 1)
PY
    then
        printf '%s\n' "spent=$spent projected=$projected" > "$ORQ/budget_exceeded"
        status "STOP budget guard: spent_usd=$spent projected_usd=$projected"
        return 1
    fi
}

partial_metrics() {
    (
        flock -x 9
        cd "$EXP"
        "$PYTHON" analyze_stress.py --partial
    ) 9>"$ORQ/analyze.lock"
}

run_model() {
    local panel=$1 model=$2
    if [[ "$DRY_RUN" == 1 ]]; then
        printf '%s %s\n' "$panel" "$model"
        return
    fi
    budget_guard
    status "START panel=$panel model=$model"
    cd "$EXP"
    "$PYTHON" infer_stress.py --panel "$panel" --model "$model"
    mkdir -p "$ORQ/done/$panel"
    touch "$ORQ/done/$panel/$model.done"
    partial_metrics
    status "DONE panel=$panel model=$model spend_usd=$(task_spend)"
}

mark_stage() {
    local stage=$1
    [[ "$DRY_RUN" == 1 ]] || touch "$ORQ/stage_${stage}_${WORKER}.done"
}

wait_stage() {
    local stage=$1
    [[ "$DRY_RUN" == 1 ]] && return
    status "WAIT stage=$stage"
    while [[ ! -f "$ORQ/stage_${stage}_a.done" || ! -f "$ORQ/stage_${stage}_b.done" ]]; do
        [[ ! -e "$ORQ/worker_a.failed" && ! -e "$ORQ/worker_b.failed" ]] || return 1
        [[ ! -e "$ORQ/budget_exceeded" ]] || return 1
        sleep 20
    done
    status "BARRIER stage=$stage complete"
}

on_exit() {
    local rc=$?
    if [[ "$DRY_RUN" == 1 ]]; then
        return
    fi
    if [[ $rc -ne 0 ]]; then
        touch "$ORQ/worker_${WORKER}.failed"
        status "FAILED rc=$rc"
    else
        status "EXIT success"
    fi
    rm -f "/workspace/users/rodrigo/kaggle_queue/guard_skip/$POD_NAME"
    if [[ -s "$API_KEY_FILE" ]]; then
        curl --silent --show-error --fail --request POST \
            --url "https://rest.runpod.io/v1/pods/$POD_ID/stop" \
            --header "Authorization: Bearer $(<"$API_KEY_FILE")" >/dev/null || true
    fi
}
trap on_exit EXIT

if [[ "$WORKER" == a ]]; then
    S_TANK=(exp088_s42 exp088_s7 exp088_s123 exp088_s2024 exp089_s42 exp089_s7)
    S_RECENT=(exp090_all exp080_all exp074_all sub075_1_all)
    S_OLDER=()
    R_MODELS=(exp029 exp030 exp031 exp033 exp045 exp046 exp047 exp048 exp052)
    F_MODELS=(exp045_f1 exp045_f2 exp045_f3 exp045_f4)
else
    S_TANK=(exp085_s42 exp085_s7 exp085_s123 exp085_s2024)
    S_RECENT=(exp071_all)
    S_OLDER=(exp057_all exp056_all)
    R_MODELS=(exp053 exp054 exp056 exp062_s42 exp062_s7 exp063_s42 exp063_s7 exp072 exp075)
    F_MODELS=(exp054_f1 exp054_f2 exp054_f3 exp054_f4)
fi

status "CHAIN start pod=$POD_NAME"
for model in "${S_TANK[@]}"; do run_model S "$model"; done
mark_stage s_tanks
wait_stage s_tanks
for model in "${S_RECENT[@]}"; do run_model S "$model"; done
mark_stage s_recent
wait_stage s_recent
for model in "${S_OLDER[@]}"; do run_model S "$model"; done
mark_stage s
wait_stage s
for model in "${R_MODELS[@]}"; do run_model R "$model"; done
mark_stage r
wait_stage r
for model in "${F_MODELS[@]}"; do run_model F "$model"; done
mark_stage f
wait_stage f

if [[ "$DRY_RUN" == 1 ]]; then
    exit 0
fi

if [[ "$WORKER" == a ]]; then
    status "ANALYSIS final"
    cd "$EXP"
    "$PYTHON" analyze_stress.py --final
    "$PYTHON" analyze_stress.py --finalize-docs
    "$PYTHON" -m unittest -v test_stress.py
    cd "$REPO"
    git diff --check
    git add experiments/exp091_stress_v1 docs/EXPERIMENTOS.md
    git add -f results/exp091_stress_v1/metrics.json
    if ! git diff --cached --quiet; then
        git commit -m "feat: complete exp091 stress benchmark"
    fi
    git bundle create "$ORQ/exp091_stress_v1.bundle" exp091_stress_v1
    touch "$ORQ/final.done"
    status "FINAL bundle=$ORQ/exp091_stress_v1.bundle spend_usd=$(task_spend)"
else
    status "WAIT final artifacts"
    while [[ ! -f "$ORQ/final.done" ]]; do
        [[ ! -e "$ORQ/worker_a.failed" ]] || exit 1
        sleep 20
    done
fi
