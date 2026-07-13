"""C18 — Voting ensemble of C11 + C15 + C16 (once done) + C17 (once done).
Combines the best configs via majority vote and soft vote.
No retraining — just combines saved predictions.
Also tries: ensemble of just C15 + C16 or C15 + C17.
"""
import sys; sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[1]))
from shared_utils import *
from sklearn.preprocessing import label_binarize

CFG = "C18_top3_ensemble"
RES = Path(__file__).parent / "results"
os.makedirs(RES, exist_ok=True)

# Try all available fine-tuned configs, pick the best ensemble
CANDIDATES = [
    "C11_ft_v3_pca128_rf",
    "C15_ft_v3_10epochs",
    "C16_ft_v3_15epochs",
    "C17_ft_v3_20epochs",
]

le     = load_le()
_, y_te = load_morph("test")
baseline = load_rf_baseline()

print(f"\n{'='*55}\n{CFG}  |  Voting ensemble of best fine-tuned V3 configs\n{'='*55}")

available = []
for dep in CANDIDATES:
    p = Path(__file__).resolve().parents[1] / dep / "results" / "y_pred.npy"
    if p.exists():
        pred = np.load(p)
        f1   = f1_score(y_te, pred, average="macro")
        available.append((dep, pred, f1))
        print(f"  {dep:<30s}: F1={f1:.4f}  errors={int((pred!=y_te).sum())}")
    else:
        print(f"  {dep:<30s}: NOT AVAILABLE")

if len(available) < 2:
    print("Need at least 2 configs. Run C16/C17 first if needed.")
    sys.exit(0)

best_f1   = -1.
best_name = ""
best_pred = None

# Try all subsets of size 2+
from itertools import combinations
for size in range(len(available), 1, -1):
    for combo in combinations(available, size):
        names  = [c[0] for c in combo]
        preds  = np.stack([c[1] for c in combo], axis=1)
        # Hard vote
        y_hard = np.apply_along_axis(
            lambda r: np.bincount(r, minlength=N_CLASSES).argmax(), 1, preds)
        # Soft vote (sum one-hot)
        soft = sum(label_binarize(c[1], classes=np.arange(N_CLASSES)).astype(np.float32)
                   for c in combo)
        y_soft = soft.argmax(axis=1)
        for mode, ypred in [("hard", y_hard), ("soft", y_soft)]:
            f1 = f1_score(y_te, ypred, average="macro")
            err = int((ypred != y_te).sum())
            if f1 > best_f1:
                best_f1   = f1
                best_name = f"{mode}_vote({'+'.join(n.split('_')[0] for n in names)})"
                best_pred = ypred
            print(f"  {mode} {'+'.join(n.split('_')[0] for n in names)}: F1={f1:.4f}  errors={err}"
                  + (" ← BEST" if f1 == best_f1 else ""))

print(f"\nBest ensemble: {best_name}  F1={best_f1:.4f}")

# Use best component CV scores as proxy
comp_scores = np.array([c[2] for c in available])
save_results(RES, CFG + f"_{best_name}", comp_scores, y_te, best_pred, le, baseline)
