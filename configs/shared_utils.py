"""Shared utilities for all 10 MorphNN configuration experiments."""

import copy, os, pickle, sys, time
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import torchvision.transforms as T
import cv2

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT      = Path(__file__).resolve().parent.parent   # MorphNN New/
CACHE_DIR = ROOT / "morphnn_cache"
DATA_DIR  = ROOT / "Dataset"

# ── Constants ──────────────────────────────────────────────────────────────────
SEED      = 42
N_CLASSES = 6
CLASSES   = ["Chickenpox", "Cowpox", "HFMD", "Healthy", "Measles", "Monkeypox"]
ALPHA     = 0.05
BOOT_N    = 2000
CV_SPLITS   = 5
CV_REPEATS  = 10   # for fast (RF/sklearn) configs
CV_REP_CNN  = 3    # for fine-tuned CNN configs
IMG_SIZE_CNN = 224
CNN_BATCH    = 32
FT_EPOCHS    = 5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Cache loaders ──────────────────────────────────────────────────────────────
def load_morph(split="train"):
    X = np.load(CACHE_DIR / f"morph_{split}_X.npy")
    y = np.load(CACHE_DIR / f"morph_{split}_y.npy")
    return X.astype(np.float32), y.astype(np.int64)

def load_cnn(backbone, split="train"):
    X = np.load(CACHE_DIR / f"cnn_{backbone}_{split}_X.npy")
    y = np.load(CACHE_DIR / f"cnn_{backbone}_{split}_y.npy")
    return X.astype(np.float32), y.astype(np.int64)

def load_le():
    with open(CACHE_DIR / "label_encoder.pkl", "rb") as f:
        return pickle.load(f)

def load_rf_baseline():
    return np.load(CACHE_DIR / "rf_baseline_cv_scores.npy")

# ── Image loading for fine-tuned configs ───────────────────────────────────────
def load_flat_images(split="train", size=IMG_SIZE_CNN):
    """Returns (images_bgr_list, y_array) from Dataset/<split>/ folder."""
    le  = load_le()
    imgs, ys = [], []
    split_dir = DATA_DIR / split
    for cls in sorted(os.listdir(split_dir)):
        cls_dir = split_dir / cls
        if not cls_dir.is_dir():
            continue
        lbl = int(le.transform([cls])[0])
        for fname in sorted(os.listdir(cls_dir)):
            img = cv2.imread(str(cls_dir / fname))
            if img is not None:
                imgs.append(cv2.resize(img, (size, size)))
                ys.append(lbl)
    return imgs, np.array(ys, dtype=np.int64)

def augment_image(img, rng):
    H, W = img.shape[:2]
    out  = img.copy()
    if rng.random() > 0.5:
        out = cv2.flip(out, 1)
    angle = rng.uniform(-20., 20.)
    M = cv2.getRotationMatrix2D((W/2, H/2), angle, 1.)
    out = cv2.warpAffine(out, M, (W, H), borderMode=cv2.BORDER_REFLECT_101)
    a = rng.uniform(0.8, 1.2)
    b = int(rng.integers(-25, 25))
    out = np.clip(a * out.astype(np.float32) + b, 0, 255).astype(np.uint8)
    if rng.random() < 0.3:
        k = int(rng.choice([3, 5]))
        out = cv2.GaussianBlur(out, (k, k), 0)
    frac = rng.uniform(0.8, 1.0)
    cH, cW = int(H*frac), int(W*frac)
    t, l = (H-cH)//2, (W-cW)//2
    out = cv2.resize(out[t:t+cH, l:l+cW], (W, H), interpolation=cv2.INTER_LINEAR)
    return out

def load_augmented_flat(target=2000, seed=SEED, size=IMG_SIZE_CNN):
    """Load training images augmented to target per class. Returns (imgs, y)."""
    le  = load_le()
    rng = np.random.default_rng(seed)
    by_class = {}
    split_dir = DATA_DIR / "train"
    for cls in sorted(os.listdir(split_dir)):
        cls_dir = split_dir / cls
        if not cls_dir.is_dir():
            continue
        imgs = []
        for fname in sorted(os.listdir(cls_dir)):
            img = cv2.imread(str(cls_dir / fname))
            if img is not None:
                imgs.append(cv2.resize(img, (size, size)))
        by_class[cls] = imgs

    flat_imgs, flat_y = [], []
    for cls, imgs in by_class.items():
        lbl = int(le.transform([cls])[0])
        pool = imgs[:]
        n_orig = len(imgs)
        if n_orig < target:
            needed = target - n_orig
            for i in range(needed):
                pool.append(augment_image(imgs[i % n_orig], rng))
        for img in pool:
            flat_imgs.append(img)
            flat_y.append(lbl)
    return flat_imgs, np.array(flat_y, dtype=np.int64)

# ── CNN model builders ─────────────────────────────────────────────────────────
def build_cnn_for_finetune(backbone_name):
    """Full model (features + head) ready for fine-tuning."""
    W = {
        "MobileNetV2": lambda: _mobilenetv2_head(),
        "MobileNetV3": lambda: _mobilenetv3_head(),
        "EfficientNet-B0": lambda: _efficientnet_head(),
        "EfficientNet-B3": lambda: _efficientnet_b3_head(),
        "ResNet18": lambda: _resnet_head("resnet18"),
        "ResNet34": lambda: _resnet_head("resnet34"),
    }
    return W[backbone_name]().to(device)

def _mobilenetv2_head():
    m = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
    m.classifier[1] = nn.Linear(m.classifier[1].in_features, N_CLASSES)
    return m

def _mobilenetv3_head():
    m = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT)
    m.classifier[3] = nn.Linear(m.classifier[3].in_features, N_CLASSES)
    return m

def _efficientnet_head():
    m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    m.classifier[1] = nn.Linear(m.classifier[1].in_features, N_CLASSES)
    return m

def _efficientnet_b3_head():
    m = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.DEFAULT)
    m.classifier[1] = nn.Linear(m.classifier[1].in_features, N_CLASSES)
    return m

def _resnet_head(name):
    ctor = getattr(models, name)
    wt   = getattr(models, f"{name.capitalize()}_Weights").DEFAULT
    m    = ctor(weights=wt)
    m.fc = nn.Linear(m.fc.in_features, N_CLASSES)
    return m

def get_feature_extractor(model, backbone_name):
    """Return the feature-extraction part (no head) of a fine-tuned model."""
    if backbone_name in ("MobileNetV2",):
        return model.features
    if backbone_name in ("MobileNetV3",):
        return nn.Sequential(model.features, model.avgpool)
    if backbone_name in ("EfficientNet-B0", "EfficientNet-B3"):
        return model.features
    if backbone_name in ("ResNet18", "ResNet34"):
        return nn.Sequential(*list(model.children())[:-2])
    raise ValueError(backbone_name)

# ── Fine-tuning ────────────────────────────────────────────────────────────────
class _ImgDs(Dataset):
    def __init__(self, imgs, labels, tf):
        self.imgs, self.labels, self.tf = imgs, labels, tf
    def __len__(self): return len(self.imgs)
    def __getitem__(self, i):
        return self.tf(cv2.cvtColor(self.imgs[i], cv2.COLOR_BGR2RGB)), self.labels[i]

def _ce_weights(y):
    y = np.asarray(y, dtype=np.int64)
    counts = np.array([max(int((y==c).sum()),1) for c in range(N_CLASSES)], np.float32)
    return torch.tensor(len(y) / (N_CLASSES * counts)).to(device)

TF_TRAIN = T.Compose([T.ToPILImage(), T.Resize(IMG_SIZE_CNN),
                       T.RandomHorizontalFlip(),
                       T.ColorJitter(brightness=0.2, contrast=0.2),
                       T.ToTensor(),
                       T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
TF_EVAL  = T.Compose([T.ToPILImage(), T.Resize(IMG_SIZE_CNN),
                       T.ToTensor(),
                       T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])

def finetune_and_extract(imgs_tr, y_tr, backbone_name, epochs=FT_EPOCHS):
    """Fine-tune backbone on imgs_tr, return (model, feature_extractor)."""
    model = build_cnn_for_finetune(backbone_name)
    dl    = DataLoader(_ImgDs(imgs_tr, y_tr, TF_TRAIN),
                       batch_size=CNN_BATCH, shuffle=True,
                       num_workers=4, pin_memory=True, persistent_workers=True)
    crit  = nn.CrossEntropyLoss(weight=_ce_weights(y_tr))
    opt   = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    for _ in range(epochs):
        model.train()
        for x, lbl in dl:
            x, lbl = x.to(device), lbl.to(device)
            opt.zero_grad()
            crit(model(x), lbl).backward()
            opt.step()
        sched.step()
    feat_ext = get_feature_extractor(model, backbone_name).to(device).eval()
    pool     = nn.AdaptiveAvgPool2d((1,1))
    return feat_ext, pool

@torch.no_grad()
def batch_embed_imgs(feat_ext, pool, imgs_bgr, batch_size=64):
    """Embed raw BGR images using a feature extractor + GAP."""
    embs = []
    for s in range(0, len(imgs_bgr), batch_size):
        batch  = [TF_EVAL(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                  for img in imgs_bgr[s:s+batch_size]]
        t      = torch.stack(batch).to(device)
        feat   = feat_ext(t)
        feat   = pool(feat).squeeze(-1).squeeze(-1)
        embs.append(feat.cpu().numpy())
    return np.vstack(embs) if embs else np.empty((0,), np.float32)

# ── Statistics ─────────────────────────────────────────────────────────────────
def bootstrap_ci(scores, n=BOOT_N, seed=SEED):
    rng   = np.random.default_rng(seed)
    means = [rng.choice(scores, len(scores), replace=True).mean() for _ in range(n)]
    return float(scores.mean()), float(np.percentile(means,2.5)), float(np.percentile(means,97.5))

def stat_test(a, b):
    if len(a)==len(b) and np.allclose(a,b): return 0., 1., "Identical"
    try:
        if len(a)==len(b):
            s, p = stats.wilcoxon(a, b, alternative="two-sided"); pfx="W"
        else:
            s, p = stats.mannwhitneyu(a, b, alternative="two-sided"); pfx="MWU"
        lab = f"{pfx} p={'<0.001' if p<0.001 else f'{p:.3f}'}"
        return float(s), float(p), lab + (" *" if p<ALPHA else "")
    except ValueError:
        return 0., 1., "Identical"

# ── Result saving ──────────────────────────────────────────────────────────────
def save_results(results_dir, cfg_name, cv_scores, y_test, y_pred, le, baseline_scores=None):
    os.makedirs(results_dir, exist_ok=True)
    m, lo, hi = bootstrap_ci(cv_scores)
    ho_f1 = f1_score(y_test, y_pred, average="macro")
    rep   = classification_report(y_test, y_pred, target_names=le.classes_, digits=4)
    _, bp, blabel = stat_test(cv_scores, baseline_scores) if baseline_scores is not None else (0,1,"N/A")
    row = {"Config": cfg_name,
           "CV_MacroF1": round(m,4), "CV_Std": round(float(cv_scores.std(ddof=1)),4),
           "CI95": f"[{lo:.4f},{hi:.4f}]",
           "Holdout_MacroF1": round(ho_f1,4),
           "Errors": int((y_pred!=y_test).sum()), "N_test": len(y_test),
           "vs_PhaseRF_baseline": blabel}
    pd.DataFrame([row]).to_csv(f"{results_dir}/summary.csv", index=False)
    with open(f"{results_dir}/classification_report.txt","w") as f:
        f.write(f"Config: {cfg_name}\n\n{rep}")
    np.save(f"{results_dir}/y_pred.npy", y_pred)
    np.save(f"{results_dir}/y_test.npy", y_test)
    print("\n" + "="*55)
    for k,v in row.items(): print(f"  {k}: {v}")
    print(f"\n{rep}")
    print(f"Results → {results_dir}")
    return row

def hybrid_cv_holdout(X_morph_tr, X_cnn_tr, y_tr,
                      X_morph_te, X_cnn_te, y_te,
                      n_pca, clf_factory,
                      n_repeats=CV_REPEATS, n_splits=CV_SPLITS,
                      seed=SEED):
    """PCA + StandardScaler + clf on (morph ‖ CNN-PCA) hybrid features.
    Returns (cv_scores_array, y_pred_holdout)."""
    cv_scores = []
    for rep in range(n_repeats):
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed+rep)
        for tr_i, vl_i in skf.split(X_morph_tr, y_tr):
            Xm_tr, Xm_vl = X_morph_tr[tr_i], X_morph_tr[vl_i]
            Xc_tr, Xc_vl = X_cnn_tr[tr_i],   X_cnn_tr[vl_i]
            ytr,   yvl   = y_tr[tr_i],         y_tr[vl_i]
            pca = PCA(n_components=min(n_pca, Xc_tr.shape[1]-1), random_state=seed)
            sc  = StandardScaler()
            Xf_tr = sc.fit_transform(np.hstack([Xm_tr, pca.fit_transform(Xc_tr)]))
            Xf_vl = sc.transform(       np.hstack([Xm_vl, pca.transform(Xc_vl)]))
            clf = clf_factory()
            clf.fit(Xf_tr, ytr)
            cv_scores.append(f1_score(yvl, clf.predict(Xf_vl), average="macro"))
    # Holdout
    pca_h = PCA(n_components=min(n_pca, X_cnn_tr.shape[1]-1), random_state=seed)
    sc_h  = StandardScaler()
    Xf_tr = sc_h.fit_transform(np.hstack([X_morph_tr, pca_h.fit_transform(X_cnn_tr)]))
    Xf_te = sc_h.transform(       np.hstack([X_morph_te, pca_h.transform(X_cnn_te)]))
    clf_h = clf_factory()
    clf_h.fit(Xf_tr, y_tr)
    y_pred = clf_h.predict(Xf_te)
    return np.array(cv_scores), y_pred
