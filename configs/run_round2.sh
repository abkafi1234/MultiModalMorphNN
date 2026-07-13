#!/usr/bin/env bash
# Round 2: C11→C12→C13 — all use fine-tuned V3 (the winner from round 1)
set -e
PY="/home/kafi/miniforge3/envs/image/bin/python"
BASE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

run() {
    local cfg=$1
    local num="${cfg%%_*}"
    echo ""
    echo "══════════════════════════════════════════════════════"
    echo " $cfg  started at $(date '+%H:%M:%S')"
    echo "══════════════════════════════════════════════════════"
    PYTHONUNBUFFERED=1 "$PY" -u "$BASE/$cfg/MorphNN_${num}.py" 2>&1 | tee "$BASE/$cfg/run.log"
    echo " $cfg  finished at $(date '+%H:%M:%S')"
    "$PY" "$BASE/compare.py" 2>/dev/null || true
}

run C11_ft_v3_pca128_rf
run C12_ft_v3_xgboost
run C13_ft_v3_nopca_rf
run C14_ft_v3_tta
run C15_ft_v3_10epochs

echo ""
echo "ROUND 2 DONE at $(date)"
"$PY" "$BASE/compare.py"
