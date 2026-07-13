"""C03 — Multi-backbone texture fusion: top-3 backbones, PCA-32 each → 96 CNN dims.
MobileNetV2 + MobileNetV3 + EfficientNet-B0 (top-3 from Phase 4 CV).
16 morph + 96 CNN = 112-dim hybrid. RF.
"""
import sys; sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[1]))
from shared_utils import *
from sklearn.ensemble import RandomForestClassifier

CFG = "C03_multiscale_pca"
RES = Path(__file__).parent / "results"
BACKBONES = ["MobileNetV2", "MobileNetV3", "EfficientNet-B0"]
N_PCA_EACH = 32   # 3 × 32 = 96 CNN dims

def rf(): return RandomForestClassifier(n_estimators=500, class_weight="balanced",
                                         random_state=SEED, n_jobs=-1)

print(f"\n{'='*55}\n{CFG}  |  16 morph + {len(BACKBONES)}×{N_PCA_EACH} CNN-PCA + RF\n{'='*55}")
t0 = time.perf_counter()

X_morph_tr, y_tr = load_morph("train")
X_morph_te, y_te = load_morph("test")
le               = load_le()
baseline         = load_rf_baseline()

# Build multi-backbone concatenated embeddings
def multi_cnn(split):
    parts = []
    for bb in BACKBONES:
        X, _ = load_cnn(bb, split)
        parts.append(X)
    return np.hstack(parts)   # (N, 1280*3)

X_cnn_tr = multi_cnn("train")   # (N, 3840)
X_cnn_te = multi_cnn("test")

cv_scores, y_pred = hybrid_cv_holdout(
    X_morph_tr, X_cnn_tr, y_tr,
    X_morph_te, X_cnn_te, y_te,
    n_pca=N_PCA_EACH * len(BACKBONES), clf_factory=rf, n_repeats=CV_REPEATS)

save_results(RES, CFG, cv_scores, y_te, y_pred, le, baseline)
print(f"Wall time: {time.perf_counter()-t0:.1f}s")
