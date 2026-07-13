"""Generate and save all benchmark figures for MorphNN on HAM10000."""
import sys, pickle
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.metrics import confusion_matrix, f1_score, classification_report

ROOT   = Path(__file__).resolve().parents[2]
CACHE  = ROOT / "ham_cache"
OUT    = Path(__file__).resolve().parent
CLASSES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
BLUE, RED = "#2563eb", "#dc2626"

def savefig(fig, name, dpi=200):
    p = OUT / name
    fig.savefig(p, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {p}")

# ── 1. Confusion matrix — C21-HAM ─────────────────────────────────────────────
print("\n[1] Confusion matrix (C21-HAM)")
res_dir = ROOT / "ham_configs" / "C21_joint_morphcnn" / "results"
y_pred  = np.load(res_dir / "y_pred.npy")
y_test  = np.load(res_dir / "y_test.npy")
cm      = confusion_matrix(y_test, y_pred)
cm_pct  = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
ho_f1   = f1_score(y_test, y_pred, average="macro")
errors  = int((y_pred != y_test).sum())

fig, ax = plt.subplots(figsize=(8, 6.5))
sns.heatmap(cm_pct, annot=False, cmap="Blues",
            xticklabels=CLASSES, yticklabels=CLASSES,
            linewidths=0.5, linecolor="white", ax=ax)
for i in range(len(CLASSES)):
    for j in range(len(CLASSES)):
        v, p = cm[i,j], cm_pct[i,j]
        col = "white" if p > 60 else "black"
        ax.text(j+0.5, i+0.5, f"{v}\n({p:.1f}%)", ha="center", va="center",
                fontsize=8, color=col, fontweight="bold")
ax.set_xlabel("Predicted Label", fontsize=11, labelpad=8)
ax.set_ylabel("True Label", fontsize=11, labelpad=8)
ax.set_title(f"Confusion Matrix — C21 Joint MorphCNN — HAM10000\nMacro F1 = {ho_f1:.4f}  |  Errors = {errors} / {len(y_test)}",
             fontsize=10, pad=10)
plt.xticks(rotation=30, ha="right", fontsize=9)
plt.yticks(rotation=0, fontsize=9)
savefig(fig, "confusion_matrix_C21_HAM.png")

# ── 2. Compare CNN baseline vs C21 ────────────────────────────────────────────
print("\n[2] Confusion matrix comparison (Phase 3 CNN vs C21-HAM)")
with open(CACHE / "phase_3.pkl", "rb") as f:
    ph3 = pickle.load(f)
y_pred_cnn = ph3["y_pred"]
f1_cnn  = f1_score(y_test, y_pred_cnn, average="macro")
err_cnn = int((y_pred_cnn != y_test).sum())
cm_cnn  = confusion_matrix(y_test, y_pred_cnn)
cm_cnn_pct = cm_cnn.astype(float) / cm_cnn.sum(axis=1, keepdims=True) * 100

fig, axes = plt.subplots(1, 2, figsize=(16, 6.5))
for ax, cm_d, cm_p, title in [
    (axes[0], cm_cnn, cm_cnn_pct, f"CNN Baseline (HAM Phase 3)\nMacro F1 = {f1_cnn:.4f}  |  Errors = {err_cnn} / {len(y_test)}"),
    (axes[1], cm,     cm_pct,     f"C21 Joint MorphCNN (Ours)\nMacro F1 = {ho_f1:.4f}  |  Errors = {errors} / {len(y_test)}"),
]:
    sns.heatmap(cm_d, annot=False, cmap="Blues",
                xticklabels=CLASSES, yticklabels=CLASSES,
                linewidths=0.5, linecolor="white", ax=ax)
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            v,p = cm_d[i,j], cm_p[i,j]
            col = "white" if p > 60 else "black"
            ax.text(j+0.5,i+0.5,f"{v}\n({p:.1f}%)",ha="center",va="center",
                    fontsize=7.5,color=col,fontweight="bold")
    ax.set_xlabel("Predicted",fontsize=10); ax.set_ylabel("True",fontsize=10)
    ax.set_title(title, fontsize=9, pad=8)
    ax.set_xticklabels(CLASSES,rotation=30,ha="right",fontsize=8)
    ax.set_yticklabels(CLASSES,rotation=0,fontsize=8)
plt.tight_layout(pad=2)
savefig(fig, "confusion_matrix_comparison_HAM.png")

# ── 3. Per-class F1 bar chart ─────────────────────────────────────────────────
print("\n[3] Per-class F1 bar chart")
rep_cnn = classification_report(y_test, y_pred_cnn, target_names=CLASSES, output_dict=True)
rep_c21 = classification_report(y_test, y_pred,     target_names=CLASSES, output_dict=True)
f1_cnn_cls = [rep_cnn[c]["f1-score"] for c in CLASSES]
f1_c21_cls = [rep_c21[c]["f1-score"] for c in CLASSES]

x, width = np.arange(len(CLASSES)), 0.35
fig, ax = plt.subplots(figsize=(10, 5.5))
b1 = ax.bar(x-width/2, f1_cnn_cls, width, label=f"CNN Baseline (F1={f1_cnn:.4f})", color="#94a3b8", edgecolor="white")
b2 = ax.bar(x+width/2, f1_c21_cls, width, label=f"C21 Joint MorphCNN (F1={ho_f1:.4f})", color=RED, edgecolor="white")
ax.set_ylim(0, 1.05); ax.set_xticks(x); ax.set_xticklabels(CLASSES, fontsize=11)
ax.set_ylabel("F1-Score", fontsize=12)
ax.set_title("Per-Class F1: CNN Baseline vs C21 Joint MorphCNN — HAM10000", fontsize=11, pad=10)
ax.legend(fontsize=10); ax.grid(axis="y", alpha=0.3)
for bar, val in zip(b1, f1_cnn_cls):
    ax.text(bar.get_x()+bar.get_width()/2, val+0.01, f"{val:.3f}", ha="center", fontsize=8)
for bar, val in zip(b2, f1_c21_cls):
    ax.text(bar.get_x()+bar.get_width()/2, val+0.01, f"{val:.3f}", ha="center", fontsize=8, fontweight="bold")
savefig(fig, "per_class_f1_HAM.png")

# ── 4. Ablation study (phase-level) ───────────────────────────────────────────
print("\n[4] Ablation study")
with open(CACHE/"phase_1.pkl","rb") as f: ph1=pickle.load(f)
with open(CACHE/"phase_2.pkl","rb") as f: ph2=pickle.load(f)
with open(CACHE/"phase_4.pkl","rb") as f: ph4=pickle.load(f)

def best_holdout(df):
    vals = df["Holdout_F1"].dropna()
    return float(vals.max()) if len(vals) > 0 else float(df["Mean_MacroF1"].max())

abl_f1_cnn = f1_cnn
abl_f1_c21 = ho_f1

ABLATION = [
    ("Phase 1\nMorph-only RF",         best_holdout(ph1["df"])),
    ("Phase 2\nFrozen CNN + Morph",     best_holdout(ph2["df"])),
    ("Phase 4\nOriginal MorphNN",       best_holdout(ph4["df"])),
    ("Phase 3\nFine-tuned CNN alone",   abl_f1_cnn),
    ("C21 Joint MorphCNN\n(Ours)",      abl_f1_c21),
]
abl_colors = ["#93c5fd","#60a5fa","#3b82f6","#f87171","#dc2626"]

fig, ax = plt.subplots(figsize=(11, 4.5))
bars = ax.barh(range(len(ABLATION)), [a[1] for a in ABLATION],
               color=abl_colors[::-1], height=0.55, edgecolor="white")
ax.axvline(abl_f1_cnn, color="#64748b", linestyle="--", linewidth=1.5, label="CNN Baseline")
ax.set_xlim(0.50, 1.02)
ax.set_yticks(range(len(ABLATION)))
ax.set_yticklabels([a[0] for a in ABLATION][::-1], fontsize=9.5)
ax.set_xlabel("Macro F1 (Holdout)", fontsize=12)
ax.set_title("Ablation Study — HAM10000 Skin Lesion Classification\nPhase-level component contributions",
             fontsize=11, pad=10)
for bar, (_, val) in zip(bars, [a for a in ABLATION][::-1]):
    ax.text(val+0.002, bar.get_y()+bar.get_height()/2, f"{val:.4f}", va="center", fontsize=9, fontweight="bold")
ax.legend(fontsize=10); ax.grid(axis="x", alpha=0.3)
plt.tight_layout()
savefig(fig, "ablation_study_HAM.png")

# ── 5. Summary CSV ────────────────────────────────────────────────────────────
print("\n[5] Summary CSV")
summary = pd.read_csv(ROOT/"ham_configs"/"C21_joint_morphcnn"/"results"/"summary.csv")
summary.to_csv(OUT/"summary_HAM.csv", index=False)
print(f"  C21-HAM: F1={ho_f1:.4f}, errors={errors}/{len(y_test)}")
print(f"  CNN baseline: F1={f1_cnn:.4f}, errors={err_cnn}/{len(y_test)}")
beats = "YES ✓" if ho_f1 > f1_cnn else "no"
print(f"  C21 beats CNN: {beats}")

abl_df = pd.DataFrame({"Component":[a[0].replace('\n',' ') for a in ABLATION],
                        "Holdout_F1":[a[1] for a in ABLATION],
                        "Beats_CNN":["YES" if a[1]>abl_f1_cnn else "no" for a in ABLATION]})
abl_df.to_csv(OUT/"ablation_table_HAM.csv", index=False)

print(f"\nAll HAM benchmark files saved to: {OUT}")
