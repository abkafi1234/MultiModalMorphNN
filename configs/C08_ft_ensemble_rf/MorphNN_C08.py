"""C08 — Fine-tuned dual-backbone (MobileNetV2 + MobileNetV3) ensemble + morph + RF.
Concatenate fine-tuned features from two backbones: richer texture representation.
16 morph + 32 FT-PCA (V2) + 32 FT-PCA (V3) = 80-dim hybrid.
"""
import sys; sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[1]))
from shared_utils import *
from sklearn.ensemble import RandomForestClassifier

CFG       = "C08_ft_ensemble_rf"
RES       = Path(__file__).parent / "results"
BACKBONES = ["MobileNetV2", "MobileNetV3"]
N_PCA_EA  = 32   # per backbone → 64 CNN dims total

def rf(): return RandomForestClassifier(n_estimators=500, class_weight="balanced",
                                         random_state=SEED, n_jobs=-1)

print(f"\n{'='*55}\n{CFG}  |  16 morph + 2×{N_PCA_EA} FT-CNN-PCA + RF\n{'='*55}")
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

def extract_dual(imgs, tr_i, backbone_list, n_pca_each, ytr):
    """Fine-tune two backbones, extract+PCA each, return stacked PCA features."""
    parts = []
    pcas  = []
    for bb in backbone_list:
        feat_ext, pool = finetune_and_extract(list(imgs[tr_i]), ytr.tolist(), bb, FT_EPOCHS)
        emb = batch_embed_imgs(feat_ext, pool, list(imgs[tr_i]))
        pca = PCA(n_components=n_pca_each, random_state=SEED)
        parts.append(pca.fit_transform(emb))
        pcas.append((feat_ext, pool, pca))
        del feat_ext, pool
    return np.hstack(parts), pcas

cv_scores = []
for rep in range(CV_REP_CNN):
    skf = StratifiedKFold(n_splits=CV_SPLITS, shuffle=True, random_state=SEED+rep)
    for fold, (tr_i, vl_i) in enumerate(skf.split(X_morph_tr, y_tr)):
        t1 = time.perf_counter()
        # Fine-tune + PCA for each backbone on fold training data
        cnn_tr_parts, cnn_vl_parts = [], []
        for bb in BACKBONES:
            feat_ext, pool = finetune_and_extract(
                list(flat_tr_imgs[tr_i]), y_tr[tr_i].tolist(), bb, FT_EPOCHS)
            emb_tr = batch_embed_imgs(feat_ext, pool, list(flat_tr_imgs[tr_i]))
            emb_vl = batch_embed_imgs(feat_ext, pool, list(flat_tr_imgs[vl_i]))
            pca = PCA(n_components=N_PCA_EA, random_state=SEED)
            cnn_tr_parts.append(pca.fit_transform(emb_tr))
            cnn_vl_parts.append(pca.transform(emb_vl))
            del feat_ext, pool
        Xc_tr_cat = np.hstack(cnn_tr_parts)
        Xc_vl_cat = np.hstack(cnn_vl_parts)
        sc  = StandardScaler()
        Xf_tr = sc.fit_transform(np.hstack([X_morph_tr[tr_i], Xc_tr_cat]))
        Xf_vl = sc.transform(       np.hstack([X_morph_tr[vl_i], Xc_vl_cat]))
        clf = rf(); clf.fit(Xf_tr, y_tr[tr_i])
        f1  = f1_score(y_tr[vl_i], clf.predict(Xf_vl), average="macro")
        cv_scores.append(f1)
        print(f"  Rep {rep+1} Fold {fold+1}  F1={f1:.4f}  ({time.perf_counter()-t1:.0f}s)  "
              f"mean: {np.mean(cv_scores):.4f}")

print("\nFine-tuning on full set for holdout ...")
cnn_tr_h_parts, cnn_te_h_parts = [], []
for bb in BACKBONES:
    feat_ext_h, pool_h = finetune_and_extract(
        list(flat_tr_imgs), y_tr.tolist(), bb, FT_EPOCHS)
    emb_tr_h = batch_embed_imgs(feat_ext_h, pool_h, list(flat_tr_imgs))
    emb_te_h = batch_embed_imgs(feat_ext_h, pool_h, list(flat_te_imgs))
    pca_h = PCA(n_components=N_PCA_EA, random_state=SEED)
    cnn_tr_h_parts.append(pca_h.fit_transform(emb_tr_h))
    cnn_te_h_parts.append(pca_h.transform(emb_te_h))
    del feat_ext_h, pool_h

sc_h = StandardScaler()
Xf_tr_h = sc_h.fit_transform(np.hstack([X_morph_tr] + cnn_tr_h_parts))
Xf_te_h = sc_h.transform(       np.hstack([X_morph_te] + cnn_te_h_parts))
clf_h = rf(); clf_h.fit(Xf_tr_h, y_tr)
y_pred = clf_h.predict(Xf_te_h)

save_results(RES, CFG, np.array(cv_scores), y_te, y_pred, le, baseline)
print(f"Total wall time: {time.perf_counter()-t0:.1f}s")
