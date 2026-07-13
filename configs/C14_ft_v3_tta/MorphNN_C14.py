"""C14 — Fine-tuned MobileNetV3 + PCA-128 + RF + Test-Time Augmentation (TTA).

Same architecture as C11 (best so far: 0.9819, 16 errors) but at inference time
we create K augmented copies of each test image, extract embeddings for all,
average the embeddings, THEN apply PCA + RF.

TTA reduces embedding variance on ambiguous images (e.g. Chickenpox/Monkeypox
border cases, early Measles vs Healthy). No retraining required — just richer
test-time representation.
"""
import sys; sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[1]))
from shared_utils import *
from sklearn.ensemble import RandomForestClassifier

CFG      = "C14_ft_v3_tta"
RES      = Path(__file__).parent / "results"
BACKBONE = "MobileNetV3"
N_PCA    = 128
TTA_K    = 12   # augmented copies per test image

def rf(): return RandomForestClassifier(n_estimators=500, class_weight="balanced",
                                         random_state=SEED, n_jobs=-1)

def tta_embed(feat_ext, pool, imgs_bgr, k=TTA_K, batch_size=64):
    """Embed each image k times with random augmentation, return mean embedding."""
    rng = np.random.default_rng(SEED)
    all_means = []
    for img in imgs_bgr:
        augmented = [img] + [augment_image(img, rng) for _ in range(k - 1)]
        embs = batch_embed_imgs(feat_ext, pool, augmented, batch_size=batch_size)
        all_means.append(embs.mean(axis=0))
    return np.stack(all_means)

print(f"\n{'='*55}\n{CFG}  |  16 morph + {N_PCA} FT-CNN-PCA ({BACKBONE}) + RF + TTA×{TTA_K}\n{'='*55}")
t0 = time.perf_counter()

X_morph_tr, y_tr = load_morph("train")
X_morph_te, y_te = load_morph("test")
le               = load_le()
baseline         = load_rf_baseline()

print("\nLoading + augmenting training images ...")
flat_tr_imgs, _ = load_augmented_flat(target=2000)
flat_te_imgs, _ = load_flat_images("test")
flat_tr_imgs = np.array(flat_tr_imgs, dtype=np.uint8)
flat_te_imgs = np.array(flat_te_imgs, dtype=np.uint8)

# ── CV: TTA on validation set only (train embeds are normal to save time) ──────
cv_scores = []
for rep in range(CV_REP_CNN):
    skf = StratifiedKFold(n_splits=CV_SPLITS, shuffle=True, random_state=SEED+rep)
    for fold, (tr_i, vl_i) in enumerate(skf.split(X_morph_tr, y_tr)):
        t1 = time.perf_counter()
        feat_ext, pool = finetune_and_extract(
            list(flat_tr_imgs[tr_i]), y_tr[tr_i], BACKBONE, FT_EPOCHS)
        Xc_tr_ft = batch_embed_imgs(feat_ext, pool, list(flat_tr_imgs[tr_i]))
        # TTA on validation images
        Xc_vl_ft = tta_embed(feat_ext, pool, list(flat_tr_imgs[vl_i]))
        pca = PCA(n_components=N_PCA, random_state=SEED)
        sc  = StandardScaler()
        Xf_tr = sc.fit_transform(np.hstack([X_morph_tr[tr_i], pca.fit_transform(Xc_tr_ft)]))
        Xf_vl = sc.transform(       np.hstack([X_morph_tr[vl_i], pca.transform(Xc_vl_ft)]))
        clf = rf(); clf.fit(Xf_tr, y_tr[tr_i])
        f1  = f1_score(y_tr[vl_i], clf.predict(Xf_vl), average="macro")
        cv_scores.append(f1)
        print(f"  Rep {rep+1} Fold {fold+1}  F1={f1:.4f}  ({time.perf_counter()-t1:.0f}s)  "
              f"mean: {np.mean(cv_scores):.4f}")
        del feat_ext, pool

# ── Holdout: TTA on all test images ───────────────────────────────────────────
print(f"\nFine-tuning on full training set + TTA×{TTA_K} on test set ...")
feat_ext_h, pool_h = finetune_and_extract(
    list(flat_tr_imgs), y_tr, BACKBONE, FT_EPOCHS)
Xc_tr_h = batch_embed_imgs(feat_ext_h, pool_h, list(flat_tr_imgs))
print(f"  Applying TTA×{TTA_K} to {len(flat_te_imgs)} test images ...")
Xc_te_h = tta_embed(feat_ext_h, pool_h, list(flat_te_imgs))

pca_h = PCA(n_components=N_PCA, random_state=SEED); sc_h = StandardScaler()
Xf_tr_h = sc_h.fit_transform(np.hstack([X_morph_tr, pca_h.fit_transform(Xc_tr_h)]))
Xf_te_h = sc_h.transform(       np.hstack([X_morph_te, pca_h.transform(Xc_te_h)]))
clf_h = rf(); clf_h.fit(Xf_tr_h, y_tr)
y_pred = clf_h.predict(Xf_te_h)

save_results(RES, CFG, np.array(cv_scores), y_te, y_pred, le, baseline)
print(f"Total wall time: {time.perf_counter()-t0:.1f}s")
