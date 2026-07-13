#!/usr/bin/env bash
# Run fine-tuned configs C06→C10 sequentially so each gets full GPU.
set -e
PY="/home/kafi/miniforge3/envs/image/bin/python"
BASE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

run() {
    local cfg=$1
    local num="${cfg%%_*}"   # e.g. C06
    local script="$BASE/$cfg/MorphNN_${num}.py"
    echo ""
    echo "══════════════════════════════════════════════════════"
    echo " $cfg  started at $(date '+%H:%M:%S')"
    echo "══════════════════════════════════════════════════════"
    PYTHONUNBUFFERED=1 "$PY" -u "$script" 2>&1 | tee "$BASE/$cfg/run.log"
    echo " $cfg  finished at $(date '+%H:%M:%S')"
    "$PY" "$BASE/compare.py" 2>/dev/null || true
}

run C06_ft_mobilenetv2_rf
run C07_ft_mobilenetv3_rf
run C08_ft_ensemble_rf
run C09_ft_xgboost
run C10_vote_ensemble

echo ""
echo "ALL FINE-TUNED CONFIGS DONE at $(date)"
"$PY" "$BASE/compare.py"
