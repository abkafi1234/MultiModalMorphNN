"""C19 — Fine-tuned MobileNetV3, 20 epochs, LR=1e-4 + PCA-128 + RF.
C16 (15 ep, 1e-4): 0.9901, 8 errors.
C17 (20 ep, 5e-5): 0.9883, 12 errors — lower LR hurt.
This keeps LR=1e-4 while pushing to 20 epochs.
T_0=5: restarts at ep 5, 10, 15, 20.
"""
import sys; sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[1]))
from shared_utils import *
from shared_utils import _ImgDs, _ce_weights
from sklearn.ensemble import RandomForestClassifier

CFG      = "C19_ft_v3_20epochs_lr1e4"
RES      = Path(__file__).parent / "results"
BACKBONE = "MobileNetV3"
N_PCA    = 128
EPOCHS   = 20

def rf(): return RandomForestClassifier(n_estimators=500, class_weight="balanced",
                                         random_state=SEED, n_jobs=-1)

def finetune_v3(imgs_tr, y_in, epochs=EPOCHS):
    y_arr = np.asarray(y_in, dtype=np.int64)
    model = build_cnn_for_finetune(BACKBONE)
    dl    = DataLoader(_ImgDs(imgs_tr, y_arr, TF_TRAIN),
                       batch_size=CNN_BATCH, shuffle=True,
                       num_workers=4, pin_memory=True, persistent_workers=True)
    crit  = nn.CrossEntropyLoss(weight=_ce_weights(y_arr), label_smoothing=0.05)
    opt   = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    sched = optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=5, T_mult=1)
    for ep in range(epochs):
        model.train()
        for x, lbl in dl:
            x, lbl = x.to(device), lbl.to(device)
            opt.zero_grad(); crit(model(x), lbl).backward(); opt.step()
        sched.step()
    feat_ext = get_feature_extractor(model, BACKBONE).to(device).eval()
    return feat_ext, nn.AdaptiveAvgPool2d((1, 1))

print(f"\n{'='*55}\n{CFG}  |  16 morph + {N_PCA} FT ({BACKBONE}, {EPOCHS} ep, lr=1e-4) + RF\n{'='*55}")
t0 = time.perf_counter()

X_morph_tr, y_tr = load_morph("train")
X_morph_te, y_te = load_morph("test")
le = load_le(); baseline = load_rf_baseline()

print("\nLoading augmented training images ...")
flat_tr_imgs, _ = load_augmented_flat(target=2000)
flat_te_imgs, _ = load_flat_images("test")
flat_tr_imgs = np.array(flat_tr_imgs, dtype=np.uint8)
flat_te_imgs = np.array(flat_te_imgs, dtype=np.uint8)

cv_scores = []
for rep in range(CV_REP_CNN):
    skf = StratifiedKFold(n_splits=CV_SPLITS, shuffle=True, random_state=SEED+rep)
    for fold, (tr_i, vl_i) in enumerate(skf.split(X_morph_tr, y_tr)):
        t1 = time.perf_counter()
        feat_ext, pool = finetune_v3(list(flat_tr_imgs[tr_i]), y_tr[tr_i])
        Xc_tr = batch_embed_imgs(feat_ext, pool, list(flat_tr_imgs[tr_i]))
        Xc_vl = batch_embed_imgs(feat_ext, pool, list(flat_tr_imgs[vl_i]))
        pca = PCA(n_components=N_PCA, random_state=SEED); sc = StandardScaler()
        Xf_tr = sc.fit_transform(np.hstack([X_morph_tr[tr_i], pca.fit_transform(Xc_tr)]))
        Xf_vl = sc.transform(       np.hstack([X_morph_tr[vl_i], pca.transform(Xc_vl)]))
        clf = rf(); clf.fit(Xf_tr, y_tr[tr_i])
        f1  = f1_score(y_tr[vl_i], clf.predict(Xf_vl), average="macro")
        cv_scores.append(f1)
        print(f"  Rep {rep+1} Fold {fold+1}  F1={f1:.4f}  ({time.perf_counter()-t1:.0f}s)  mean:{np.mean(cv_scores):.4f}")
        del feat_ext, pool

print("\nFull fine-tune for holdout ...")
feat_ext_h, pool_h = finetune_v3(list(flat_tr_imgs), y_tr)
Xc_tr_h = batch_embed_imgs(feat_ext_h, pool_h, list(flat_tr_imgs))
Xc_te_h = batch_embed_imgs(feat_ext_h, pool_h, list(flat_te_imgs))
pca_h = PCA(n_components=N_PCA, random_state=SEED); sc_h = StandardScaler()
Xf_tr_h = sc_h.fit_transform(np.hstack([X_morph_tr, pca_h.fit_transform(Xc_tr_h)]))
Xf_te_h = sc_h.transform(       np.hstack([X_morph_te, pca_h.transform(Xc_te_h)]))
clf_h = rf(); clf_h.fit(Xf_tr_h, y_tr)
y_pred = clf_h.predict(Xf_te_h)

save_results(RES, CFG, np.array(cv_scores), y_te, y_pred, le, baseline)
print(f"Wall time: {time.perf_counter()-t0:.1f}s")
