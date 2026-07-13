"""Aggregate all config results into a single comparison table."""
import sys, os
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parent

CONFIGS = [
    "C01_pca64_rf", "C02_pca256_rf", "C03_multiscale_pca",
    "C04_extratrees", "C05_mlp_gpu",
    "C06_ft_mobilenetv2_rf", "C07_ft_mobilenetv3_rf",
    "C08_ft_ensemble_rf", "C09_ft_xgboost", "C10_vote_ensemble",
    "C11_ft_v3_pca128_rf", "C12_ft_v3_xgboost", "C13_ft_v3_nopca_rf",
    "C14_ft_v3_tta", "C15_ft_v3_10epochs",
    "C16_ft_v3_15epochs", "C17_ft_v3_20epochs", "C18_top3_ensemble",
    "C19_ft_v3_20epochs_lr1e4", "C20_ft_effb3_15epochs", "C21_joint_morphcnn",
]

TARGET_F1 = 0.9946   # Phase 3 fine-tuned CNN holdout (to beat)

rows = []
for cfg in CONFIGS:
    p = ROOT / cfg / "results" / "summary.csv"
    if not p.exists():
        rows.append({"Config": cfg, "Status": "pending"})
        continue
    r = pd.read_csv(p).iloc[0].to_dict()
    r["Status"]   = "done"
    r["Beats_CNN"] = "YES ✓" if float(r.get("Holdout_MacroF1", 0)) > TARGET_F1 else "no"
    rows.append(r)

df = pd.DataFrame(rows)
print("\n" + "="*75)
print(f"COMPARISON TABLE  (target to beat: {TARGET_F1})")
print("="*75)
cols = ["Config", "Status", "CV_MacroF1", "CV_Std", "Holdout_MacroF1",
        "Errors", "vs_PhaseRF_baseline", "Beats_CNN"]
print(df[[c for c in cols if c in df.columns]].to_string(index=False))

done = df[df["Status"]=="done"]
if len(done):
    best = done.loc[done["Holdout_MacroF1"].astype(float).idxmax()]
    print(f"\nBest config so far: {best['Config']}  Holdout F1={best['Holdout_MacroF1']}")
    if float(best["Holdout_MacroF1"]) > TARGET_F1:
        print(f"*** CNN DEFEATED! MorphNN wins by "
              f"{float(best['Holdout_MacroF1'])-TARGET_F1:+.4f} ***")
    else:
        print(f"Gap to CNN: {float(best['Holdout_MacroF1'])-TARGET_F1:+.4f}")

df.to_csv(ROOT / "comparison_table.csv", index=False)
print(f"\nFull table → {ROOT}/comparison_table.csv")
