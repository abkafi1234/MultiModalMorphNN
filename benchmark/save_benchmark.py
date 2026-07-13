"""Generate and save all benchmark figures for the MorphNN 6-class paper."""
import sys, pickle
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.metrics import confusion_matrix, f1_score

ROOT   = Path(__file__).resolve().parent.parent
CACHE  = ROOT / "morphnn_cache"
CFGS   = ROOT / "configs"
OUT    = Path(__file__).resolve().parent
CLASSES = ["Chickenpox", "Cowpox", "HFMD", "Healthy", "Measles", "Monkeypox"]
BLUE, ORANGE, RED = "#2563eb", "#f97316", "#dc2626"

# ── helpers ───────────────────────────────────────────────────────────────────
def savefig(fig, name, dpi=200):
    p = OUT / name
    fig.savefig(p, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {p}")

# ── 1. Comparison table CSV ───────────────────────────────────────────────────
print("\n[1] Comparison table")
cmp = pd.read_csv(CFGS / "comparison_table.csv")
cmp.to_csv(OUT / "comparison_table.csv", index=False)
print(f"  saved → {OUT}/comparison_table.csv")

# ── 2. Confusion matrix — C21 ─────────────────────────────────────────────────
print("\n[2] Confusion matrix (C21 Joint MorphCNN)")
y_pred_c21 = np.load(CFGS / "C21_joint_morphcnn/results/y_pred.npy")
y_test      = np.load(CFGS / "C21_joint_morphcnn/results/y_test.npy")
cm = confusion_matrix(y_test, y_pred_c21)
cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(cm_pct, annot=False, fmt=".1f", cmap="Blues",
            xticklabels=CLASSES, yticklabels=CLASSES,
            linewidths=0.5, linecolor="white", ax=ax)
for i in range(len(CLASSES)):
    for j in range(len(CLASSES)):
        val = cm[i, j]
        pct = cm_pct[i, j]
        color = "white" if pct > 60 else "black"
        ax.text(j + 0.5, i + 0.5, f"{val}\n({pct:.1f}%)",
                ha="center", va="center", fontsize=8.5, color=color, fontweight="bold")
ax.set_xlabel("Predicted Label", fontsize=12, labelpad=8)
ax.set_ylabel("True Label", fontsize=12, labelpad=8)
ax.set_title("Confusion Matrix — C21 Joint MorphCNN (Holdout)\nMacro F1 = 0.9977  |  Errors = 2 / 1134",
             fontsize=11, pad=12)
plt.xticks(rotation=30, ha="right", fontsize=9)
plt.yticks(rotation=0, fontsize=9)
savefig(fig, "confusion_matrix_C21.png")

# Also save CNN baseline confusion matrix for comparison
print("\n[2b] Confusion matrix (CNN baseline — Phase 3 FineTuned_ResNet34)")
with open(CACHE / "phase_3.pkl", "rb") as f:
    ph3 = pickle.load(f)
y_pred_cnn = ph3["y_pred"]
y_te_cnn   = np.load(CACHE / "morph_test_y.npy")
cm_cnn = confusion_matrix(y_te_cnn, y_pred_cnn)
cm_cnn_pct = cm_cnn.astype(float) / cm_cnn.sum(axis=1, keepdims=True) * 100
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
for ax, cm_data, pct_data, title in [
    (axes[0], cm_cnn, cm_cnn_pct, f"CNN Baseline (ResNet34)\nMacro F1 = 0.9946  |  Errors = 6 / 1134"),
    (axes[1], cm, cm_pct,         "C21 Joint MorphCNN (Ours)\nMacro F1 = 0.9977  |  Errors = 2 / 1134"),
]:
    sns.heatmap(cm_data, annot=False, cmap="Blues",
                xticklabels=CLASSES, yticklabels=CLASSES,
                linewidths=0.5, linecolor="white", ax=ax)
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            v, p = cm_data[i, j], pct_data[i, j]
            col = "white" if p > 60 else "black"
            ax.text(j+0.5, i+0.5, f"{v}\n({p:.1f}%)", ha="center", va="center",
                    fontsize=7.5, color=col, fontweight="bold")
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("True", fontsize=11)
    ax.set_title(title, fontsize=10, pad=8)
    ax.set_xticklabels(CLASSES, rotation=30, ha="right", fontsize=8)
    ax.set_yticklabels(CLASSES, rotation=0, fontsize=8)
plt.tight_layout(pad=2)
savefig(fig, "confusion_matrix_comparison.png")

# ── 3. F1 progression chart ───────────────────────────────────────────────────
print("\n[3] F1 progression chart")
KEY_CONFIGS = [
    ("Morph-only RF\n(Phase 1)",          0.7457),
    ("Frozen CNN+Morph\n(Phase 2)",        0.8951),
    ("Original MorphNN\n(Phase 4)",        0.8324),
    ("Fine-tuned CNN\nalone (Phase 3)",    0.9946),
    ("MorphNN + FT V3\n5 ep (C11)",        0.9819),
    ("MorphNN + FT V3\n10 ep (C15)",       0.9845),
    ("MorphNN + FT V3\n15 ep (C16)",       0.9901),
    ("MorphNN + FT V3\n20 ep (C19)",       0.9925),
    ("MorphNN + EffNetB3\n15 ep (C20)",    0.9926),
    ("Joint MorphCNN\n(C21 — Ours)",       0.9977),
]
labels = [x[0] for x in KEY_CONFIGS]
f1s    = [x[1] for x in KEY_CONFIGS]
colors = [BLUE if i < 8 else ORANGE if i == 8 else RED for i, _ in enumerate(KEY_CONFIGS)]

fig, ax = plt.subplots(figsize=(14, 5.5))
bars = ax.bar(range(len(labels)), f1s, color=colors, width=0.65, edgecolor="white", linewidth=0.8)
ax.axhline(0.9946, color="#64748b", linestyle="--", linewidth=1.5, label="CNN Baseline (0.9946)")
ax.set_ylim(0.70, 1.005)
ax.set_xticks(range(len(labels)))
ax.set_xticklabels(labels, fontsize=8.5)
ax.set_ylabel("Macro F1 (Holdout)", fontsize=12)
ax.set_title("MorphNN Progression — Morphology + CNN Feature Fusion\nFrom Phase 1 (morph-only) to C21 Joint MorphCNN",
             fontsize=11, pad=10)
for bar, val in zip(bars, f1s):
    ax.text(bar.get_x() + bar.get_width()/2, val + 0.003,
            f"{val:.4f}", ha="center", va="bottom", fontsize=7.5, fontweight="bold")
patches = [
    mpatches.Patch(color=BLUE,   label="MorphNN variants"),
    mpatches.Patch(color=ORANGE, label="EfficientNet-B3 hybrid"),
    mpatches.Patch(color=RED,    label="Joint MorphCNN (C21 — Ours)"),
]
ax.legend(handles=patches + [plt.Line2D([0],[0], color="#64748b", linestyle="--", linewidth=1.5,
                                         label="CNN Baseline (0.9946)")],
          fontsize=9, loc="lower right")
ax.grid(axis="y", alpha=0.3)
savefig(fig, "f1_progression.png")

# ── 4. Per-class F1 bar chart ─────────────────────────────────────────────────
print("\n[4] Per-class F1 bar chart")
from sklearn.metrics import classification_report
import json

rep_cnn = classification_report(y_te_cnn, y_pred_cnn, target_names=CLASSES, output_dict=True)
rep_c21 = classification_report(y_test,   y_pred_c21, target_names=CLASSES, output_dict=True)

f1_cnn = [rep_cnn[c]["f1-score"] for c in CLASSES]
f1_c21 = [rep_c21[c]["f1-score"] for c in CLASSES]

x     = np.arange(len(CLASSES))
width = 0.35
fig, ax = plt.subplots(figsize=(10, 5.5))
b1 = ax.bar(x - width/2, f1_cnn, width, label="CNN Baseline (ResNet34)", color="#94a3b8", edgecolor="white")
b2 = ax.bar(x + width/2, f1_c21, width, label="C21 Joint MorphCNN (Ours)", color=RED, edgecolor="white")
ax.set_ylim(0.94, 1.005)
ax.set_xticks(x)
ax.set_xticklabels(CLASSES, fontsize=11)
ax.set_ylabel("F1-Score", fontsize=12)
ax.set_title("Per-Class F1: CNN Baseline vs C21 Joint MorphCNN\nHoldout test set (1,134 images)",
             fontsize=11, pad=10)
ax.legend(fontsize=10)
ax.grid(axis="y", alpha=0.3)
for bar, val in zip(b1, f1_cnn):
    ax.text(bar.get_x() + bar.get_width()/2, val - 0.003,
            f"{val:.3f}", ha="center", va="top", fontsize=8, color="white", fontweight="bold")
for bar, val in zip(b2, f1_c21):
    ax.text(bar.get_x() + bar.get_width()/2, val + 0.001,
            f"{val:.3f}", ha="center", va="bottom", fontsize=8, fontweight="bold")
savefig(fig, "per_class_f1.png")

# ── 5. Ablation study figure ──────────────────────────────────────────────────
print("\n[5] Ablation study")
ABLATION = [
    ("Phase 1\nMorph-only RF",         0.7457, "Morphological\nfeatures only"),
    ("Phase 2\nFrozen CNN + Morph",    0.8951, "+ Frozen\nImageNet CNN"),
    ("Phase 4\nOriginal MorphNN",       0.8324, "+ Hybrid\n(frozen CNN)"),
    ("Phase 3\nFine-tuned CNN alone",  0.9946, "Fine-tuned\nCNN (no morph)"),
    ("MorphNN + FT CNN\n(C16, 15 ep)", 0.9901, "+ Fine-tuned\nCNN hybrid"),
    ("MorphNN + FT CNN\n(C20, EffB3)", 0.9926, "+ Larger\nbackbone (B3)"),
    ("C21 Joint MorphCNN\n(Ours)",     0.9977, "End-to-end\njoint training"),
]
abl_labels = [a[0] for a in ABLATION]
abl_f1     = [a[1] for a in ABLATION]
abl_notes  = [a[2] for a in ABLATION]
abl_colors = ["#93c5fd","#60a5fa","#3b82f6","#f87171","#f97316","#fb923c","#dc2626"]

fig, ax = plt.subplots(figsize=(13, 5.5))
bars = ax.barh(range(len(ABLATION)), abl_f1, color=abl_colors[::-1],
               height=0.55, edgecolor="white")
ax.axvline(0.9946, color="#64748b", linestyle="--", linewidth=1.5, label="CNN Baseline")
ax.set_xlim(0.68, 1.015)
ax.set_yticks(range(len(ABLATION)))
ax.set_yticklabels(abl_labels[::-1], fontsize=9.5)
ax.set_xlabel("Macro F1 (Holdout)", fontsize=12)
ax.set_title("Ablation Study — Component Contribution to MorphNN Performance\n"
             "Each row adds a capability over the row above",
             fontsize=11, pad=10)
for i, (bar, val) in enumerate(zip(bars, abl_f1[::-1])):
    ax.text(val + 0.002, bar.get_y() + bar.get_height()/2,
            f"{val:.4f}", va="center", fontsize=9, fontweight="bold")
ax.legend(fontsize=10)
ax.grid(axis="x", alpha=0.3)
plt.tight_layout()
savefig(fig, "ablation_study.png")

# ── 6. Ablation table CSV ─────────────────────────────────────────────────────
abl_df = pd.DataFrame({
    "Component":    [a[0].replace("\n"," ") for a in ABLATION],
    "Description":  [a[2].replace("\n"," ") for a in ABLATION],
    "Holdout_F1":   abl_f1,
    "Delta_vs_prev":["-"] + [f"{abl_f1[i]-abl_f1[i-1]:+.4f}" for i in range(1,len(abl_f1))],
    "Beats_CNN":    ["YES" if f > 0.9946 else "no" for f in abl_f1],
})
abl_df.to_csv(OUT / "ablation_table.csv", index=False)
print(f"  saved → {OUT}/ablation_table.csv")

print("\nAll benchmark files saved to:", OUT)
