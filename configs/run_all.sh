#!/usr/bin/env bash
# Run all 10 MorphNN configurations sequentially.
# C01-C05 are fast (cached frozen features, ~5-15 min each).
# C06-C10 are slow (per-fold fine-tuning, ~2h each).

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="/home/kafi/miniforge3/envs/image/bin/python"

run_config() {
    local cfg=$1
    local script="$SCRIPT_DIR/$cfg/MorphNN_${cfg%%_*}.py"
    echo ""
    echo "══════════════════════════════════════════════════════"
    echo " Starting: $cfg  at $(date '+%H:%M:%S')"
    echo "══════════════════════════════════════════════════════"
    PYTHONUNBUFFERED=1 "$PY" -u "$script" 2>&1
    echo ""
    echo " Finished: $cfg  at $(date '+%H:%M:%S')"
    echo "══════════════════════════════════════════════════════"
    # Show running comparison after each config
    "$PY" "$SCRIPT_DIR/compare.py" 2>/dev/null || true
}

# ── Fast configs (frozen features) ────────────────────────────────────────────
run_config C01_pca64_rf
run_config C02_pca256_rf
run_config C03_multiscale_pca
run_config C04_extratrees
run_config C05_mlp_gpu

# ── Slow configs (fine-tuned CNN features) ────────────────────────────────────
run_config C06_ft_mobilenetv2_rf
run_config C07_ft_mobilenetv3_rf
run_config C08_ft_ensemble_rf
run_config C09_ft_xgboost

# ── Ensemble (depends on C06-C09) ─────────────────────────────────────────────
run_config C10_vote_ensemble

echo ""
echo "ALL CONFIGS DONE  $(date)"
"$PY" "$SCRIPT_DIR/compare.py"
