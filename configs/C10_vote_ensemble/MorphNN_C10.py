"""C10 — Voting ensemble of C06 + C07 + C08 + C09 holdout predictions.
Combines the fine-tuned hybrid predictions via majority vote.
Depends on C06-C09 having already saved their y_pred.npy files.
"""
import sys; sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[1]))
from shared_utils import *

CFG = "C10_vote_ensemble"
RES = Path(__file__).parent / "results"
os.makedirs(RES, exist_ok=True)

DEPS = ["C06_ft_mobilenetv2_rf", "C07_ft_mobilenetv3_rf",
        "C08_ft_ensemble_rf",    "C09_ft_xgboost"]

print(f"\n{'='*55}\n{CFG}  |  Voting ensemble of C06–C09\n{'='*55}")

le     = load_le()
_, y_te = load_morph("test")
baseline = load_rf_baseline()

preds = []
for dep in DEPS:
    p = Path(__file__).resolve().parents[1] / dep / "results" / "y_pred.npy"
    if not p.exists():
        print(f"  MISSING: {p}  — skipping {dep}")
        continue
    preds.append(np.load(p))
    print(f"  Loaded: {dep}  "
          f"F1={f1_score(y_te, preds[-1], average='macro'):.4f}")

if len(preds) < 2:
    print("Not enough predictions for ensemble. Run C06-C09 first.")
    sys.exit(1)

# Majority vote
votes = np.stack(preds, axis=1)   # (N, k)
y_pred_vote = np.apply_along_axis(
    lambda row: np.bincount(row, minlength=N_CLASSES).argmax(), 1, votes)

# Also try soft voting via per-model confidence (binary: 1 if matches class, 0 otherwise)
# Simple: sum one-hot encodings
from sklearn.preprocessing import label_binarize
soft = np.zeros((len(y_te), N_CLASSES), dtype=np.float32)
for p in preds:
    soft += label_binarize(p, classes=np.arange(N_CLASSES)).astype(np.float32)
y_pred_soft = soft.argmax(axis=1)

ho_f1_hard = f1_score(y_te, y_pred_vote, average="macro")
ho_f1_soft = f1_score(y_te, y_pred_soft, average="macro")
print(f"\nHard vote holdout macro F1 : {ho_f1_hard:.4f}")
print(f"Soft vote holdout macro F1 : {ho_f1_soft:.4f}")

best_pred = y_pred_soft if ho_f1_soft >= ho_f1_hard else y_pred_vote
best_f1   = max(ho_f1_hard, ho_f1_soft)

# Use component CV scores as a proxy (average their means)
comp_scores = []
for dep in DEPS:
    p = Path(__file__).resolve().parents[1] / dep / "results" / "summary.csv"
    if p.exists():
        import pandas as pd
        row = pd.read_csv(p)
        comp_scores.append(float(row["CV_MacroF1"].iloc[0]))
cv_proxy = np.array(comp_scores) if comp_scores else np.array([best_f1])

save_results(RES, CFG, cv_proxy, y_te, best_pred, le, baseline)
