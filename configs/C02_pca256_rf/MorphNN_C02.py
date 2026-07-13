"""C02 — Expand CNN-PCA to 256 dims (73.3% variance). Frozen MobileNetV2. RF.
Even more texture detail than C01. Tests how much the bottleneck matters.
"""
import sys; sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[1]))
from shared_utils import *
from sklearn.ensemble import RandomForestClassifier

CFG = "C02_pca256_rf"
RES = Path(__file__).parent / "results"
N_PCA = 256

def rf(): return RandomForestClassifier(n_estimators=500, class_weight="balanced",
                                         random_state=SEED, n_jobs=-1)

print(f"\n{'='*55}\n{CFG}  |  16 morph + {N_PCA} CNN-PCA (frozen MobileNetV2) + RF\n{'='*55}")
t0 = time.perf_counter()

X_morph_tr, y_tr = load_morph("train")
X_morph_te, y_te = load_morph("test")
X_cnn_tr,   _    = load_cnn("MobileNetV2", "train")
X_cnn_te,   _    = load_cnn("MobileNetV2", "test")
le               = load_le()
baseline         = load_rf_baseline()

cv_scores, y_pred = hybrid_cv_holdout(
    X_morph_tr, X_cnn_tr, y_tr,
    X_morph_te, X_cnn_te, y_te,
    n_pca=N_PCA, clf_factory=rf, n_repeats=CV_REPEATS)

save_results(RES, CFG, cv_scores, y_te, y_pred, le, baseline)
print(f"Wall time: {time.perf_counter()-t0:.1f}s")
