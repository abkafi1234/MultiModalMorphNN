"""Per-class clinical sensitivity/specificity table for MorphNN on HAM10000.
Computes: Sensitivity (Recall), Specificity, PPV (Precision), NPV, F1
from existing y_pred.npy and y_test.npy — no retraining needed.
Saves: per_class_sensitivity_HAM.csv, per_class_sensitivity_HAM.png
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

ROOT    = Path(__file__).resolve().parents[2]
CACHE   = ROOT / "ham_cache"
OUT     = Path(__file__).resolve().parent
RES_DIR = ROOT / "ham_configs" / "C21_joint_morphcnn" / "results"
CLASSES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
FULL_NAMES = {
    "akiec": "Actinic Keratosis",
    "bcc":   "Basal Cell Carc.",
    "bkl":   "Benign Keratosis",
    "df":    "Dermatofibroma",
    "mel":   "Melanoma",
    "nv":    "Melanocytic Nevi",
    "vasc":  "Vascular Lesion",
}

def savefig(fig, name, dpi=200):
    p = OUT / name
    fig.savefig(p, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {p}")


def per_class_stats(y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred)
    n = len(classes)
    rows = []
    for i, cls in enumerate(classes):
        TP = cm[i, i]
        FN = cm[i, :].sum() - TP          # missed by model
        FP = cm[:, i].sum() - TP          # false alarms
        TN = cm.sum() - TP - FN - FP
        support = int(cm[i, :].sum())
        sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        specificity = TN / (TN + FP) if (TN + FP) > 0 else 0.0
        ppv         = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        npv         = TN / (TN + FN) if (TN + FN) > 0 else 0.0
        f1          = 2*TP / (2*TP + FP + FN) if (2*TP + FP + FN) > 0 else 0.0
        rows.append({
            "Class":       cls,
            "Full Name":   FULL_NAMES[cls],
            "n (test)":    support,
            "TP": int(TP), "FP": int(FP), "FN": int(FN), "TN": int(TN),
            "Sensitivity": round(sensitivity, 4),
            "Specificity": round(specificity, 4),
            "PPV":         round(ppv, 4),
            "NPV":         round(npv, 4),
            "F1":          round(f1, 4),
        })
    return pd.DataFrame(rows)


# ── Load MorphNN results ─────────────────────────────────────────────────────
y_pred_morphnn = np.load(RES_DIR / "y_pred.npy")
y_test         = np.load(RES_DIR / "y_test.npy")

# ── Load CNN baseline (Phase 3) ──────────────────────────────────────────────
import pickle
with open(CACHE / "phase_3.pkl", "rb") as f:
    ph3 = pickle.load(f)
y_pred_cnn = ph3["y_pred"]

print(f"MorphNN  accuracy: {accuracy_score(y_test, y_pred_morphnn):.4f}")
print(f"CNN-FT   accuracy: {accuracy_score(y_test, y_pred_cnn):.4f}")

df_morphnn = per_class_stats(y_test, y_pred_morphnn, CLASSES)
df_cnn     = per_class_stats(y_test, y_pred_cnn,     CLASSES)

# ── Save CSVs ────────────────────────────────────────────────────────────────
df_morphnn.to_csv(OUT / "per_class_sensitivity_morphnn_HAM.csv", index=False)
df_cnn.to_csv(    OUT / "per_class_sensitivity_cnn_HAM.csv",     index=False)
print("\nMorphNN per-class stats:")
print(df_morphnn[["Class","n (test)","Sensitivity","Specificity","PPV","NPV","F1"]].to_string(index=False))
print("\nCNN-FT per-class stats:")
print(df_cnn[["Class","n (test)","Sensitivity","Specificity","PPV","NPV","F1"]].to_string(index=False))

# ── Figure 1: Sensitivity + Specificity side-by-side bar chart ───────────────
BLUE, RED = "#2563eb", "#dc2626"
GRAY = "#94a3b8"

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
x = np.arange(len(CLASSES))
w = 0.35

for ax, metric, title in [
    (axes[0], "Sensitivity", "Sensitivity (Recall) — HAM10000"),
    (axes[1], "Specificity", "Specificity — HAM10000"),
]:
    vals_cnn  = df_cnn[metric].values
    vals_morph = df_morphnn[metric].values
    b1 = ax.bar(x - w/2, vals_cnn,  w, label="CNN-FT Baseline", color=GRAY, edgecolor="white")
    b2 = ax.bar(x + w/2, vals_morph, w, label="MorphNN (Ours)",  color=RED,  edgecolor="white")
    ax.set_ylim(0, 1.12)
    ax.set_xticks(x)
    ax.set_xticklabels(CLASSES, fontsize=10)
    ax.set_ylabel(metric, fontsize=12)
    ax.set_title(title, fontsize=11, pad=8)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    for bar, val in zip(b1, vals_cnn):
        ax.text(bar.get_x()+bar.get_width()/2, val+0.01, f"{val:.3f}",
                ha="center", va="bottom", fontsize=7.5)
    for bar, val in zip(b2, vals_morph):
        ax.text(bar.get_x()+bar.get_width()/2, val+0.01, f"{val:.3f}",
                ha="center", va="bottom", fontsize=7.5, fontweight="bold")

plt.tight_layout(pad=2)
savefig(fig, "per_class_sensitivity_HAM.png")

# ── Figure 2: Full clinical metrics heatmap (MorphNN) ────────────────────────
metrics = ["Sensitivity", "Specificity", "PPV", "NPV", "F1"]
heat_data = df_morphnn[metrics].values.T  # shape: (5, 7)

fig, ax = plt.subplots(figsize=(11, 4.5))
im = ax.imshow(heat_data, cmap="RdYlGn", vmin=0.3, vmax=1.0, aspect="auto")
ax.set_xticks(range(len(CLASSES)))
ax.set_xticklabels([f"{c}\n(n={df_morphnn.loc[i,'n (test)']})"
                    for i, c in enumerate(CLASSES)], fontsize=9)
ax.set_yticks(range(len(metrics)))
ax.set_yticklabels(metrics, fontsize=10)
ax.set_title("MorphNN — Per-Class Clinical Metrics (HAM10000)", fontsize=11, pad=10)
plt.colorbar(im, ax=ax, fraction=0.025, pad=0.03)
for i in range(len(metrics)):
    for j in range(len(CLASSES)):
        v = heat_data[i, j]
        ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                fontsize=8.5, fontweight="bold",
                color="white" if v < 0.45 or v > 0.85 else "black")
plt.tight_layout()
savefig(fig, "clinical_metrics_heatmap_HAM.png")

# ── Print key finding for paper ───────────────────────────────────────────────
print("\n" + "="*60)
print("KEY FINDINGS FOR PAPER:")
print("="*60)
for _, row in df_morphnn.iterrows():
    delta_sens = row["Sensitivity"] - df_cnn.loc[df_cnn["Class"]==row["Class"], "Sensitivity"].values[0]
    print(f"  {row['Class']:6s}  Sensitivity={row['Sensitivity']:.4f}  "
          f"Specificity={row['Specificity']:.4f}  NPV={row['NPV']:.4f}  "
          f"Δsens={delta_sens:+.4f}")
print()

# Overall accuracy and errors
n_total = len(y_test)
err_morphnn = (y_pred_morphnn != y_test).sum()
err_cnn     = (y_pred_cnn != y_test).sum()
print(f"MorphNN errors: {err_morphnn}/{n_total}  ({100*err_morphnn/n_total:.2f}%)")
print(f"CNN-FT  errors: {err_cnn}/{n_total}   ({100*err_cnn/n_total:.2f}%)")
print(f"Error reduction: {100*(err_cnn-err_morphnn)/err_cnn:.1f}%")
print(f"\nAll files saved to: {OUT}")
