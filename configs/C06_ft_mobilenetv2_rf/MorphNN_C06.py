"""C06 — Fine-tuned MobileNetV2 features + 16 morph + RF.

KEY HYPOTHESIS: Frozen ImageNet features capture generic texture;
fine-tuned features capture lesion-specific texture. This is the
primary reason Phase 3 CNN beats Phase 4 hybrid — the CNN adapts
its representations to the task, ours doesn't.

Per-fold fine-tuning inside CV (no leakage). 5 epochs × 5 folds × 3 repeats = 75 fine-tuning runs.
"""
import sys; sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[1]))
from shared_utils import *
from sklearn.ensemble import RandomForestClassifier

CFG      = "C06_ft_mobilenetv2_rf"
RES      = Path(__file__).parent / "results"
BACKBONE = "MobileNetV2"
N_PCA    = 64

def rf(): return RandomForestClassifier(n_estimators=500, class_weight="balanced",
                                         random_state=SEED, n_jobs=-1)

print(f"\n{'='*55}\n{CFG}  |  16 morph + {N_PCA} FT-CNN-PCA ({BACKBONE}) + RF\n{'='*55}")
print(f"Per-fold fine-tuning: {CV_SPLITS} folds × {CV_REP_CNN} repeats × {FT_EPOCHS} epochs = "
      f"{CV_SPLITS*CV_REP_CNN*FT_EPOCHS} total epochs")
t0 = time.perf_counter()

X_morph_tr, y_tr = load_morph("train")
X_morph_te, y_te = load_morph("test")
le               = load_le()
baseline         = load_rf_baseline()

print("\nLoading + augmenting training images (needed for fine-tuning) ...")
flat_tr_imgs, flat_tr_y = load_augmented_flat(target=2000)
flat_te_imgs, flat_te_y = load_flat_images("test")
flat_tr_imgs = np.array(flat_tr_imgs, dtype=np.uint8)   # (N, H, W, 3)
flat_te_imgs = np.array(flat_te_imgs, dtype=np.uint8)
print(f"  train: {flat_tr_imgs.shape}, test: {flat_te_imgs.shape}")

# ── CV with per-fold fine-tuning ───────────────────────────────────────────────
cv_scores = []
for rep in range(CV_REP_CNN):
    skf = StratifiedKFold(n_splits=CV_SPLITS, shuffle=True, random_state=SEED+rep)
    for fold, (tr_i, vl_i) in enumerate(skf.split(X_morph_tr, y_tr)):
        t1 = time.perf_counter()
        # Fine-tune on fold's training images
        feat_ext, pool = finetune_and_extract(
            list(flat_tr_imgs[tr_i]), y_tr[tr_i].tolist(), BACKBONE, FT_EPOCHS)
        # Extract embeddings
        Xc_tr_ft = batch_embed_imgs(feat_ext, pool, list(flat_tr_imgs[tr_i]))
        Xc_vl_ft = batch_embed_imgs(feat_ext, pool, list(flat_tr_imgs[vl_i]))
        # Hybrid
        pca = PCA(n_components=N_PCA, random_state=SEED)
        sc  = StandardScaler()
        Xf_tr = sc.fit_transform(np.hstack([X_morph_tr[tr_i], pca.fit_transform(Xc_tr_ft)]))
        Xf_vl = sc.transform(       np.hstack([X_morph_tr[vl_i], pca.transform(Xc_vl_ft)]))
        clf = rf(); clf.fit(Xf_tr, y_tr[tr_i])
        f1  = f1_score(y_tr[vl_i], clf.predict(Xf_vl), average="macro")
        cv_scores.append(f1)
        print(f"  Rep {rep+1}/{CV_REP_CNN} Fold {fold+1}/{CV_SPLITS}  F1={f1:.4f}  "
              f"({time.perf_counter()-t1:.0f}s)  mean so far: {np.mean(cv_scores):.4f}")
        del feat_ext, pool   # free GPU memory

# ── Holdout: fine-tune on ALL training data ────────────────────────────────────
print("\nFine-tuning on full training set for holdout ...")
feat_ext_h, pool_h = finetune_and_extract(
    list(flat_tr_imgs), y_tr.tolist(), BACKBONE, FT_EPOCHS)
Xc_tr_h = batch_embed_imgs(feat_ext_h, pool_h, list(flat_tr_imgs))
Xc_te_h = batch_embed_imgs(feat_ext_h, pool_h, list(flat_te_imgs))

pca_h = PCA(n_components=N_PCA, random_state=SEED); sc_h = StandardScaler()
Xf_tr_h = sc_h.fit_transform(np.hstack([X_morph_tr, pca_h.fit_transform(Xc_tr_h)]))
Xf_te_h = sc_h.transform(       np.hstack([X_morph_te, pca_h.transform(Xc_te_h)]))
clf_h = rf(); clf_h.fit(Xf_tr_h, y_tr)
y_pred = clf_h.predict(Xf_te_h)

save_results(RES, CFG, np.array(cv_scores), y_te, y_pred, le, baseline)
print(f"Total wall time: {time.perf_counter()-t0:.1f}s")
