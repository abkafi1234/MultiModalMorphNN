"""C15 — Fine-tuned MobileNetV3, 10 epochs (2× longer) + PCA-128 + RF.

C11 uses 5 epochs. The model may not have fully converged given 13k training images.
Doubling to 10 epochs + cosine annealing with warm restarts (SGDR) for better
convergence. Hypothesis: the 4 Chickenpox/Monkeypox confusions and Healthy/Measles
confusions stem from insufficient feature adaptation in 5 epochs.
"""
import sys; sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[1]))
from shared_utils import *
from shared_utils import _ImgDs, _ce_weights   # private names not exported by import *
from sklearn.ensemble import RandomForestClassifier

CFG        = "C15_ft_v3_10epochs"
RES        = Path(__file__).parent / "results"
BACKBONE   = "MobileNetV3"
N_PCA      = 128
EPOCHS     = 10

def rf(): return RandomForestClassifier(n_estimators=500, class_weight="balanced",
                                         random_state=SEED, n_jobs=-1)

def finetune_long(imgs_tr, y_tr_in, backbone, epochs=EPOCHS):
    """Fine-tune with cosine annealing warm restarts + label smoothing."""
    y_arr = np.asarray(y_tr_in, dtype=np.int64)
    model = build_cnn_for_finetune(backbone)
    dl    = DataLoader(_ImgDs(imgs_tr, y_arr, TF_TRAIN),
                       batch_size=CNN_BATCH, shuffle=True,
                       num_workers=4, pin_memory=True, persistent_workers=True)
    crit  = nn.CrossEntropyLoss(weight=_ce_weights(y_arr), label_smoothing=0.05)
    opt   = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    # Warm restarts: T_0=4 epochs, T_mult=1 → restarts at epoch 4, 8, ...
    sched = optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=4, T_mult=1)
    for ep in range(epochs):
        model.train()
        for x, lbl in dl:
            x, lbl = x.to(device), lbl.to(device)
            opt.zero_grad()
            crit(model(x), lbl).backward()
            opt.step()
        sched.step()
    feat_ext = get_feature_extractor(model, backbone).to(device).eval()
    pool     = nn.AdaptiveAvgPool2d((1, 1))
    return feat_ext, pool

print(f"\n{'='*55}\n{CFG}  |  16 morph + {N_PCA} FT-CNN-PCA ({BACKBONE}, {EPOCHS} ep) + RF\n{'='*55}")
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

cv_scores = []
for rep in range(CV_REP_CNN):
    skf = StratifiedKFold(n_splits=CV_SPLITS, shuffle=True, random_state=SEED+rep)
    for fold, (tr_i, vl_i) in enumerate(skf.split(X_morph_tr, y_tr)):
        t1 = time.perf_counter()
        feat_ext, pool = finetune_long(list(flat_tr_imgs[tr_i]), y_tr[tr_i], BACKBONE)
        Xc_tr_ft = batch_embed_imgs(feat_ext, pool, list(flat_tr_imgs[tr_i]))
        Xc_vl_ft = batch_embed_imgs(feat_ext, pool, list(flat_tr_imgs[vl_i]))
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

print("\nFine-tuning on full training set for holdout ...")
feat_ext_h, pool_h = finetune_long(list(flat_tr_imgs), y_tr, BACKBONE)
Xc_tr_h = batch_embed_imgs(feat_ext_h, pool_h, list(flat_tr_imgs))
Xc_te_h = batch_embed_imgs(feat_ext_h, pool_h, list(flat_te_imgs))

pca_h = PCA(n_components=N_PCA, random_state=SEED); sc_h = StandardScaler()
Xf_tr_h = sc_h.fit_transform(np.hstack([X_morph_tr, pca_h.fit_transform(Xc_tr_h)]))
Xf_te_h = sc_h.transform(       np.hstack([X_morph_te, pca_h.transform(Xc_te_h)]))
clf_h = rf(); clf_h.fit(Xf_tr_h, y_tr)
y_pred = clf_h.predict(Xf_te_h)

save_results(RES, CFG, np.array(cv_scores), y_te, y_pred, le, baseline)
print(f"Total wall time: {time.perf_counter()-t0:.1f}s")
