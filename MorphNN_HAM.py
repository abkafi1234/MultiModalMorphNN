#!/usr/bin/env python3
"""
MorphNN_6class.py  —  v2
MorphNN: Morphology-Informed Frozen Feature Fusion — 6-Class Viral Exanthem Triage.

Extends the binary Varicella-Measles framework to all six MCVSLD classes:
  Chickenpox, Cowpox, HFMD, Healthy, Measles, Monkeypox.

Design highlights:
  - 16 deterministic morphological descriptors (original 8 + 8 new)
  - 6 PCA-compressed frozen CNN texture components  →  22-dim hybrid vector
  - Morphology contributes 16/22 = 73 % of the fused feature space
  - Augmentation-based class balancing on training split only
  - 4-phase evaluation: morphology-only · frozen transfer · fine-tuned CNNs · hybrid fusion
  - Repeated stratified 5-fold CV · Wilcoxon signed-rank tests · bootstrap 95 % CIs
  - McNemar test comparing primary MorphNN against best Phase-3 fine-tuned CNN
  - Batch CNN embedding (GPU) for fast feature extraction
  - Disk cache for morphological features and CNN embeddings (skip re-extraction on reruns)
  - Phase-level checkpointing: completed phases are loaded from disk on reruns

Dataset layout expected:
  Dataset/
    train/<ClassName>/*.{jpg,png,...}
    val/<ClassName>/*.{jpg,png,...}
    test/<ClassName>/*.{jpg,png,...}
"""

# ── Imports ────────────────────────────────────────────────────────────────────
import copy
import os
import random
import pickle
import time
import warnings
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial import distance
from scipy import stats

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score, classification_report, confusion_matrix

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import torchvision.transforms as T

warnings.filterwarnings("ignore")

# ── Configuration ──────────────────────────────────────────────────────────────
SEED             = 42
DATASET_ROOT     = "./Ham Dataset"
TRAIN_DIR        = os.path.join(DATASET_ROOT, "train")
VAL_DIR          = os.path.join(DATASET_ROOT, "val")
TEST_DIR         = os.path.join(DATASET_ROOT, "test")
OUTPUT_DIR       = "./ham_results"
CACHE_DIR        = "./ham_cache"           # persisted feature arrays

N_MORPH_FEATURES = 16
N_CNN_PCA_COMPS  = 6
N_FUSED_FEATURES = N_MORPH_FEATURES + N_CNN_PCA_COMPS   # 22

AUG_TARGET_PER_CLASS = 2000
IMG_SIZE_MORPH       = 512
IMG_SIZE_CNN         = 224

CV_SPLITS      = 5
CV_REPEATS     = 10   # Phases 1 & 4  (fast)
CV_REPEATS_CNN = 3    # Phases 2 & 3  (GPU-intensive)
CNN_EPOCHS     = 5
CNN_BATCH      = 32   # larger batch for GPU
CNN_EMB_BATCH  = 64   # batch size during frozen embedding
ALPHA          = 0.05
BOOTSTRAP_N    = 2000

# Alphabetical order — matches LabelEncoder.fit() output
CLASSES   = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
N_CLASSES = len(CLASSES)

CNN_BACKBONES = [
    "MobileNetV2", "MobileNetV3", "ShuffleNetV2",
    "SqueezeNet", "ResNet18", "ResNet34", "EfficientNet-B0",
]
PRIMARY_BACKBONE = "MobileNetV2"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CACHE_DIR,  exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Reproducibility ────────────────────────────────────────────────────────────
def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False

set_seed(SEED)


# ── Augmentation ───────────────────────────────────────────────────────────────
def augment_image(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Random photometric + geometric transform.  All ops use supplied RNG."""
    H, W = img.shape[:2]
    out  = img.copy()

    if rng.random() > 0.5:                         # horizontal flip
        out = cv2.flip(out, 1)

    angle = rng.uniform(-20.0, 20.0)               # rotation ±20°
    M     = cv2.getRotationMatrix2D((W / 2, H / 2), angle, 1.0)
    out   = cv2.warpAffine(out, M, (W, H), borderMode=cv2.BORDER_REFLECT_101)

    alpha = rng.uniform(0.80, 1.20)                # brightness / contrast jitter
    beta  = int(rng.integers(-25, 25))
    out   = np.clip(alpha * out.astype(np.float32) + beta, 0, 255).astype(np.uint8)

    if rng.random() < 0.30:                        # mild blur (30 % chance)
        k   = int(rng.choice([3, 5]))
        out = cv2.GaussianBlur(out, (k, k), 0)

    frac      = rng.uniform(0.80, 1.00)            # centre crop + resize
    cH, cW    = int(H * frac), int(W * frac)
    top, left = (H - cH) // 2, (W - cW) // 2
    out       = cv2.resize(out[top:top + cH, left:left + cW], (W, H),
                           interpolation=cv2.INTER_LINEAR)
    return out


def augment_to_target(
    class_images: dict,
    target: int = AUG_TARGET_PER_CLASS,
    seed:   int = SEED,
) -> dict:
    """Return new dict with minority classes augmented up to `target` images.
    Classes at or above `target` are left unchanged (no downsampling)."""
    rng       = np.random.default_rng(seed)
    augmented = {}
    for cls, imgs in class_images.items():
        n_orig = len(imgs)
        if n_orig >= target:
            augmented[cls] = imgs
            continue
        needed = target - n_orig
        extras = []
        for i in range(needed):
            src     = imgs[i % n_orig]
            resized = cv2.resize(src, (IMG_SIZE_MORPH, IMG_SIZE_MORPH))
            extras.append(augment_image(resized, rng))
        augmented[cls] = imgs + extras
    return augmented


# ── Expanded Morphological Feature Extractor (16 features) ────────────────────
class MorphFeatureExtractor:
    """
    16 deterministic morphological descriptors.

    Original 8 (binary MorphNN):
      f01 lesion_count          number of retained lesion-like contours
      f02 avg_lesion_area       mean contour area / image area
      f03 area_heterogeneity    std of contour areas / image area
      f04 avg_circularity       mean isoperimetric circularity
      f05 sparsity_score        mean nearest-neighbour centroid dist / max_dim
      f06 confluence_density    total lesion area / image area
      f07 localized_hue         mean hue within lesion mask (HSV H)
      f08 localized_saturation  mean saturation within lesion mask (HSV S)

    Extended 8 (new — for 6-class discrimination):
      f09 avg_aspect_ratio      mean bounding-box width/height
      f10 avg_solidity          mean area / convex-hull area
      f11 localized_value       mean brightness within lesion mask (HSV V)
      f12 hue_std               std of hue within lesion mask
      f13 saturation_std        std of saturation within lesion mask
      f14 spatial_entropy       Shannon entropy of 4×4 lesion-density grid
      f15 max_lesion_area_ratio largest single lesion area / image area
      f16 background_saturation mean saturation outside the lesion mask
    """

    FEATURE_NAMES = [
        "lesion_count",       "avg_lesion_area",     "area_heterogeneity", "avg_circularity",
        "sparsity_score",     "confluence_density",  "localized_hue",      "localized_saturation",
        "avg_aspect_ratio",   "avg_solidity",        "localized_value",
        "hue_std",            "saturation_std",      "spatial_entropy",
        "max_lesion_area_ratio", "background_saturation",
    ]

    def _gray_world_wb(self, img: np.ndarray) -> np.ndarray:
        b, g, r   = cv2.split(img.astype(np.float32))
        mu        = (b.mean() + g.mean() + r.mean()) / 3.0
        b = np.clip(b * (mu / b.mean() if b.mean() > 0 else 1), 0, 255)
        g = np.clip(g * (mu / g.mean() if g.mean() > 0 else 1), 0, 255)
        r = np.clip(r * (mu / r.mean() if r.mean() > 0 else 1), 0, 255)
        return cv2.merge((b, g, r)).astype(np.uint8)

    def _spatial_entropy(self, centroids, H: int, W: int, grid: int = 4) -> float:
        if not centroids:
            return 0.0
        density = np.zeros((grid, grid), dtype=np.float32)
        for cx, cy in centroids:
            gx = min(int(cx / W * grid), grid - 1)
            gy = min(int(cy / H * grid), grid - 1)
            density[gy, gx] += 1
        prob = density.flatten() / density.sum()
        prob = prob[prob > 0]
        return float(-np.sum(prob * np.log2(prob)))

    def extract(self, img: np.ndarray) -> np.ndarray:
        """(16,) float64 vector for a BGR image at any resolution."""
        H, W     = img.shape[:2]
        total_px = H * W
        max_dim  = max(H, W)

        smoothed  = cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)
        gray      = cv2.cvtColor(smoothed, cv2.COLOR_BGR2GRAY)
        clahe     = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
        equalized = clahe.apply(gray)
        thresh    = cv2.adaptiveThreshold(equalized, 255,
                                          cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                          cv2.THRESH_BINARY_INV, 51, 2)
        kernel    = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        clean     = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
        contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        min_a = 0.00019 * total_px
        max_a = 0.10    * total_px

        centroids, areas, circularities = [], [], []
        aspect_ratios, solidities, valid_contours = [], [], []

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if not (min_a < area < max_a):
                continue
            perim = cv2.arcLength(cnt, True)
            circ  = 0.0 if perim == 0 else 4 * np.pi * area / (perim * perim)
            m     = cv2.moments(cnt)
            if m["m00"] == 0:
                continue
            centroids.append((int(m["m10"] / m["m00"]), int(m["m01"] / m["m00"])))
            areas.append(area)
            circularities.append(circ)
            valid_contours.append(cnt)
            x, y, bw, bh = cv2.boundingRect(cnt)
            aspect_ratios.append(bw / bh if bh > 0 else 1.0)
            hull      = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            solidities.append(area / hull_area if hull_area > 0 else 1.0)

        wb_img  = self._gray_world_wb(img)
        hsv_img = cv2.cvtColor(wb_img, cv2.COLOR_BGR2HSV)
        mask    = np.zeros((H, W), dtype=np.uint8)
        if valid_contours:
            cv2.drawContours(mask, valid_contours, -1, 255, cv2.FILLED)

        if mask.any():
            h_vals  = hsv_img[:, :, 0][mask == 255].astype(np.float32)
            s_vals  = hsv_img[:, :, 1][mask == 255].astype(np.float32)
            v_vals  = hsv_img[:, :, 2][mask == 255].astype(np.float32)
            loc_hue = float(h_vals.mean())
            loc_sat = float(s_vals.mean())
            loc_val = float(v_vals.mean())
            hue_std = float(h_vals.std()) if len(h_vals) > 1 else 0.0
            sat_std = float(s_vals.std()) if len(s_vals) > 1 else 0.0
        else:
            loc_hue = loc_sat = loc_val = hue_std = sat_std = 0.0

        bg_mask = cv2.bitwise_not(mask)
        bg_sat  = float(hsv_img[:, :, 1][bg_mask == 255].astype(np.float32).mean()) \
                  if bg_mask.any() else 0.0

        n          = len(areas)
        avg_area   = float(np.mean(areas)  / total_px) if areas else 0.0
        std_area   = float(np.std(areas)   / total_px) if n > 1  else 0.0
        avg_circ   = float(np.mean(circularities))     if areas else 0.0
        conf_dens  = float(sum(areas)      / total_px) if areas else 0.0
        max_area_r = float(max(areas)      / total_px) if areas else 0.0
        avg_ar     = float(np.mean(aspect_ratios))     if aspect_ratios else 1.0
        avg_sol    = float(np.mean(solidities))        if solidities    else 1.0

        if len(centroids) > 1:
            dm       = distance.cdist(centroids, centroids, "euclidean")
            np.fill_diagonal(dm, np.inf)
            sparsity = float(np.mean(np.min(dm, axis=1)) / max_dim)
        else:
            sparsity = 0.0

        sp_ent = self._spatial_entropy(centroids, H, W)

        return np.array([
            n,         avg_area,  std_area,  avg_circ,
            sparsity,  conf_dens, loc_hue,   loc_sat,
            avg_ar,    avg_sol,   loc_val,
            hue_std,   sat_std,   sp_ent,
            max_area_r, bg_sat,
        ], dtype=np.float64)


# ── Frozen CNN Feature Extractor ───────────────────────────────────────────────
class FrozenCNNExtractor:
    """Extracts global-average-pooled embeddings from a frozen ImageNet backbone."""

    def __init__(self, backbone_name: str = PRIMARY_BACKBONE):
        self.name      = backbone_name
        self.pool      = nn.AdaptiveAvgPool2d((1, 1))
        self.model     = self._load(backbone_name).to(device).eval()
        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize((IMG_SIZE_CNN, IMG_SIZE_CNN)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def _load(self, name: str) -> nn.Module:
        W = {"EfficientNet-B0": lambda: models.efficientnet_b0(
                 weights=models.EfficientNet_B0_Weights.DEFAULT).features,
             "ResNet18":    lambda: nn.Sequential(*list(models.resnet18(
                 weights=models.ResNet18_Weights.DEFAULT).children())[:-2]),
             "ResNet34":    lambda: nn.Sequential(*list(models.resnet34(
                 weights=models.ResNet34_Weights.DEFAULT).children())[:-2]),
             "MobileNetV2": lambda: models.mobilenet_v2(
                 weights=models.MobileNet_V2_Weights.DEFAULT).features,
             "MobileNetV3": lambda: models.mobilenet_v3_large(
                 weights=models.MobileNet_V3_Large_Weights.DEFAULT).features,
             "ShuffleNetV2": lambda: nn.Sequential(*list(models.shufflenet_v2_x1_0(
                 weights=models.ShuffleNet_V2_X1_0_Weights.DEFAULT).children())[:-1]),
             "SqueezeNet":  lambda: models.squeezenet1_1(
                 weights=models.SqueezeNet1_1_Weights.DEFAULT).features}
        if name not in W:
            raise ValueError(f"Unknown backbone: {name}")
        return W[name]()

    @torch.no_grad()
    def batch_embed(self, images_bgr: list, batch_size: int = CNN_EMB_BATCH) -> np.ndarray:
        """Embed a list of BGR images in batches.  Much faster than one-by-one."""
        all_embs = []
        for start in range(0, len(images_bgr), batch_size):
            batch_imgs = images_bgr[start: start + batch_size]
            tensors    = [self.transform(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                          for img in batch_imgs]
            batch_t    = torch.stack(tensors).to(device)
            feat       = self.model(batch_t)
            feat       = self.pool(feat).squeeze(-1).squeeze(-1)
            all_embs.append(feat.cpu().numpy())
        return np.vstack(all_embs) if all_embs else np.empty((0,), dtype=np.float32)


# ── Dataset Loading ────────────────────────────────────────────────────────────
def load_images_from_dir(root: str) -> dict:
    """Returns {class_name: [bgr_images]} sorted alphabetically by class."""
    data = {}
    for cls in sorted(os.listdir(root)):
        cls_dir = os.path.join(root, cls)
        if not os.path.isdir(cls_dir):
            continue
        imgs = []
        for fname in sorted(os.listdir(cls_dir)):
            img = cv2.imread(os.path.join(cls_dir, fname))
            if img is not None:
                imgs.append(img)
        data[cls] = imgs
    return data


def _flat_images_labels(class_images: dict, le: LabelEncoder) -> tuple:
    """Flatten {class: [imgs]} into (flat_imgs_list, y_array)."""
    flat_imgs, ys = [], []
    for cls, imgs in class_images.items():
        lbl = int(le.transform([cls])[0])
        flat_imgs.extend(imgs)
        ys.extend([lbl] * len(imgs))
    return flat_imgs, np.array(ys)


# ── Feature Cache ──────────────────────────────────────────────────────────────
def _cache_path(name: str, cache_dir: str = CACHE_DIR) -> str:
    return os.path.join(cache_dir, name)


def extract_morph_features_cached(
    class_images: dict,
    extractor: MorphFeatureExtractor,
    split: str,
    cache_dir: str = CACHE_DIR,
) -> tuple:
    """Extract (or load cached) morphology features + labels + LabelEncoder."""
    x_path  = _cache_path(f"morph_{split}_X.npy",  cache_dir)
    y_path  = _cache_path(f"morph_{split}_y.npy",  cache_dir)
    le_path = _cache_path(f"label_encoder.pkl",     cache_dir)

    if all(os.path.exists(p) for p in [x_path, y_path, le_path]):
        print(f"  [cache] Loading morphology features for {split} split ...")
        X  = np.load(x_path)
        y  = np.load(y_path)
        with open(le_path, "rb") as f:
            le = pickle.load(f)
        return X, y, le

    print(f"  Extracting morphology features for {split} split ...")
    le = LabelEncoder().fit(list(class_images.keys()))
    X, y = [], []
    total = sum(len(v) for v in class_images.values())
    done  = 0
    for cls, imgs in class_images.items():
        lbl = int(le.transform([cls])[0])
        for img in imgs:
            resized = cv2.resize(img, (IMG_SIZE_MORPH, IMG_SIZE_MORPH))
            X.append(extractor.extract(resized))
            y.append(lbl)
            done += 1
            if done % max(total // 20, 1) == 0:
                print(f"    {done}/{total}  ({100*done//total} %)")
    X_arr = np.array(X, dtype=np.float64)
    y_arr = np.array(y, dtype=np.int64)
    np.save(x_path, X_arr)
    np.save(y_path, y_arr)
    with open(le_path, "wb") as f:
        pickle.dump(le, f)
    return X_arr, y_arr, le


def extract_cnn_features_cached(
    class_images: dict,
    cnn: FrozenCNNExtractor,
    le: LabelEncoder,
    split: str,
    cache_dir: str = CACHE_DIR,
) -> tuple:
    """Extract (or load cached) CNN embeddings + labels."""
    x_path = _cache_path(f"cnn_{cnn.name}_{split}_X.npy", cache_dir)
    y_path = _cache_path(f"cnn_{cnn.name}_{split}_y.npy", cache_dir)

    if all(os.path.exists(p) for p in [x_path, y_path]):
        print(f"  [cache] Loading CNN embeddings ({cnn.name}, {split}) ...")
        return np.load(x_path), np.load(y_path)

    print(f"  Extracting CNN embeddings ({cnn.name}, {split}) ...")
    flat_imgs, y_arr = _flat_images_labels(class_images, le)
    X_arr = cnn.batch_embed(flat_imgs)
    np.save(x_path, X_arr.astype(np.float32))
    np.save(y_path, y_arr)
    return X_arr, y_arr


# ── Statistical Utilities ──────────────────────────────────────────────────────
def bootstrap_ci(scores: np.ndarray, n_boot: int = BOOTSTRAP_N,
                 confidence: float = 0.95, seed: int = SEED) -> tuple:
    rng   = np.random.default_rng(seed)
    means = [rng.choice(scores, len(scores), replace=True).mean()
             for _ in range(n_boot)]
    lo    = np.percentile(means, 100 * (1 - confidence) / 2)
    hi    = np.percentile(means, 100 * (1 + confidence) / 2)
    return float(scores.mean()), float(lo), float(hi)


def wilcoxon_test(scores_a: np.ndarray, scores_b: np.ndarray) -> tuple:
    """Non-parametric comparison.  Uses Wilcoxon signed-rank for equal-length
    paired arrays (within-phase), and Mann-Whitney U for unequal-length arrays
    (cross-phase where repeat counts differ).  Returns (stat, p, label)."""
    if len(scores_a) == len(scores_b) and np.allclose(scores_a, scores_b):
        return 0.0, 1.0, "Identical"
    try:
        if len(scores_a) == len(scores_b):
            stat, p = stats.wilcoxon(scores_a, scores_b, alternative="two-sided")
            prefix  = "W"
        else:
            stat, p = stats.mannwhitneyu(scores_a, scores_b, alternative="two-sided")
            prefix  = "MWU"
        label = f"{prefix} p={'<0.001' if p < 0.001 else f'{p:.3f}'}"
        sig   = " *" if p < ALPHA else ""
        return float(stat), float(p), label + sig
    except ValueError:
        return 0.0, 1.0, "Identical"


def mcnemar_test(y_true: np.ndarray,
                 y_pred_a: np.ndarray,
                 y_pred_b: np.ndarray) -> tuple:
    """Mid-P McNemar test.  Returns (chi2, p, description)."""
    b = int(((y_pred_a != y_true) & (y_pred_b == y_true)).sum())
    c = int(((y_pred_a == y_true) & (y_pred_b != y_true)).sum())
    if b + c == 0:
        return 0.0, 1.0, "No discordant pairs"
    if b + c < 25:
        p = float(2 * stats.binom.cdf(min(b, c), b + c, 0.5))
    else:
        chi2 = (abs(b - c) - 1.0) ** 2 / (b + c)
        p    = float(stats.chi2.sf(chi2, df=1))
    return float(b - c), p, f"b={b}, c={c}"


def compute_summary(name: str, scores: np.ndarray,
                    baseline_scores: np.ndarray | None = None) -> dict:
    mean, ci_lo, ci_hi = bootstrap_ci(scores)
    row = {
        "Model":         name,
        "Mean_MacroF1":  round(mean, 4),
        "Std":           round(float(scores.std(ddof=1)), 4),
        "CI95_Lo":       round(ci_lo, 4),
        "CI95_Hi":       round(ci_hi, 4),
    }
    if baseline_scores is not None:
        stat, p, label = wilcoxon_test(scores, baseline_scores)
        row.update({"Wilcoxon_W": round(stat, 2),
                    "p_value":    round(p, 4),
                    "vs_RF_baseline": label})
    return row


# ── Phase 1: Morphology-Only Classifiers ──────────────────────────────────────
def phase1_morphology_only(
    X_train: np.ndarray, y_train: np.ndarray,
    X_test:  np.ndarray, y_test:  np.ndarray,
    le: LabelEncoder,
    n_repeats: int = CV_REPEATS,
) -> tuple:
    """Returns (df_results, rf_cv_scores, y_pred_holdout_rf)."""
    print("\n" + "=" * 65)
    print("PHASE 1 — Morphology-Only Classifiers")
    print("=" * 65)
    t0 = time.perf_counter()

    classifiers = {
        "RF_morph":  RandomForestClassifier(n_estimators=200, class_weight="balanced",
                                            random_state=SEED, n_jobs=-1),
        "SVM_morph": SVC(kernel="rbf", class_weight="balanced",
                         random_state=SEED),
        "KNN_morph": KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
    }
    cv_scores: dict = {k: [] for k in classifiers}

    for rep in range(n_repeats):
        skf = StratifiedKFold(n_splits=CV_SPLITS, shuffle=True,
                               random_state=SEED + rep)
        for tr_idx, vl_idx in skf.split(X_train, y_train):
            Xtr, Xvl = X_train[tr_idx], X_train[vl_idx]
            ytr, yvl = y_train[tr_idx], y_train[vl_idx]
            sc       = StandardScaler()
            Xtr_s    = sc.fit_transform(Xtr)
            Xvl_s    = sc.transform(Xvl)
            for name, clf in classifiers.items():
                c = copy.deepcopy(clf)
                c.fit(Xtr_s, ytr)
                cv_scores[name].append(f1_score(yvl, c.predict(Xvl_s), average="macro"))

    rf_baseline = np.array(cv_scores["RF_morph"])
    rows = []
    for name, s in cv_scores.items():
        baseline = None if name == "RF_morph" else rf_baseline
        rows.append(compute_summary(name, np.array(s), baseline))
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))

    # Holdout on best CV model + RF specifically
    sc_full   = StandardScaler().fit(X_train)
    Xtr_s     = sc_full.transform(X_train)
    Xte_s     = sc_full.transform(X_test)

    best_name = df.sort_values("Mean_MacroF1", ascending=False).iloc[0]["Model"]
    best_clf  = copy.deepcopy(classifiers[best_name])
    best_clf.fit(Xtr_s, y_train)
    y_pred_best = best_clf.predict(Xte_s)
    ho_f1_best  = f1_score(y_test, y_pred_best, average="macro")

    rf_clf = copy.deepcopy(classifiers["RF_morph"])
    rf_clf.fit(Xtr_s, y_train)
    y_pred_rf = rf_clf.predict(Xte_s)
    ho_f1_rf  = f1_score(y_test, y_pred_rf, average="macro")

    print(f"\nHoldout — best CV model ({best_name}): macro F1 = {ho_f1_best:.4f}"
          f"  |  RF macro F1 = {ho_f1_rf:.4f}"
          f"  |  errors = {int((y_pred_rf != y_test).sum())}/{len(y_test)}")
    print(f"Phase 1 wall time: {time.perf_counter() - t0:.1f} s")

    df["Holdout_F1"] = None
    df.loc[df["Model"] == best_name, "Holdout_F1"] = round(ho_f1_best, 4)
    return df, rf_baseline, y_pred_rf


# ── Phase 2: Frozen Transfer Learning ─────────────────────────────────────────
def phase2_frozen_transfer(
    train_images: dict,
    test_images:  dict,
    le:           LabelEncoder,
    y_test:       np.ndarray,
    rf_baseline_scores: np.ndarray,
    n_repeats:    int = CV_REPEATS_CNN,
) -> tuple:
    """Returns (df_results, y_pred_holdout_best_backbone)."""
    print("\n" + "=" * 65)
    print("PHASE 2 — Frozen Transfer Learning Baselines")
    print("=" * 65)
    t0 = time.perf_counter()

    rows  = []
    best_bb_name   = None
    best_bb_mean   = -1.0
    y_pred_holdout = None

    for bb_name in CNN_BACKBONES:
        print(f"  [{CNN_BACKBONES.index(bb_name)+1}/{len(CNN_BACKBONES)}] {bb_name} ...")
        cnn = FrozenCNNExtractor(bb_name)

        X_emb_tr, y_emb_tr = extract_cnn_features_cached(
            train_images, cnn, le, split="train")
        X_emb_te, _         = extract_cnn_features_cached(
            test_images,  cnn, le, split="test")

        cv_scores: list = []
        t1 = time.perf_counter()
        for rep in range(n_repeats):
            skf = StratifiedKFold(n_splits=CV_SPLITS, shuffle=True,
                                   random_state=SEED + rep)
            for tr_idx, vl_idx in skf.split(X_emb_tr, y_emb_tr):
                Xtr, Xvl = X_emb_tr[tr_idx], X_emb_tr[vl_idx]
                ytr, yvl = y_emb_tr[tr_idx], y_emb_tr[vl_idx]
                pca   = PCA(n_components=min(50, Xtr.shape[1]), random_state=SEED)
                sc    = StandardScaler()
                Xtr_p = sc.fit_transform(pca.fit_transform(Xtr))
                Xvl_p = sc.transform(pca.transform(Xvl))
                clf   = RandomForestClassifier(n_estimators=200, class_weight="balanced",
                                               random_state=SEED, n_jobs=-1)
                clf.fit(Xtr_p, ytr)
                cv_scores.append(f1_score(yvl, clf.predict(Xvl_p), average="macro"))

        arr      = np.array(cv_scores)
        t_fold   = (time.perf_counter() - t1) / (n_repeats * CV_SPLITS)
        row      = compute_summary(f"Frozen_{bb_name}", arr, rf_baseline_scores)
        row["Avg_CV_time_s"] = round(t_fold, 2)
        rows.append(row)
        print(f"    CV macro F1: {arr.mean():.4f} ± {arr.std():.4f}")

        if arr.mean() > best_bb_mean:
            best_bb_mean = arr.mean()
            best_bb_name = bb_name
            # Holdout evaluation for this backbone
            pca_h  = PCA(n_components=50, random_state=SEED)
            sc_h   = StandardScaler()
            Xtr_ph = sc_h.fit_transform(pca_h.fit_transform(X_emb_tr))
            Xte_ph = sc_h.transform(pca_h.transform(X_emb_te))
            rf_h   = RandomForestClassifier(n_estimators=500, class_weight="balanced",
                                            random_state=SEED, n_jobs=-1)
            rf_h.fit(Xtr_ph, y_emb_tr)
            y_pred_holdout = rf_h.predict(Xte_ph)

    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    ho_f1 = f1_score(y_test, y_pred_holdout, average="macro") if y_pred_holdout is not None else 0.0
    n_err = int((y_pred_holdout != y_test).sum()) if y_pred_holdout is not None else -1
    print(f"\nBest Phase 2 ({best_bb_name}) Holdout macro F1: {ho_f1:.4f}  errors: {n_err}/{len(y_test)}")
    print(f"Phase 2 wall time: {time.perf_counter() - t0:.1f} s")
    df["Holdout_F1"] = None
    df.loc[df["Model"] == f"Frozen_{best_bb_name}", "Holdout_F1"] = round(ho_f1, 4)
    return df, y_pred_holdout


# ── Phase 3: Fine-Tuned CNNs ──────────────────────────────────────────────────
class _ImgDataset(Dataset):
    def __init__(self, images: np.ndarray, labels: np.ndarray, transform=None):
        self.images, self.labels, self.transform = images, labels, transform
    def __len__(self): return len(self.images)
    def __getitem__(self, idx):
        img = self.images[idx]
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]


def _build_cnn(backbone_name: str, n_classes: int) -> nn.Module:
    builders = {
        "EfficientNet-B0": lambda: _head(
            models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT),
            "classifier", 1, n_classes),
        "ResNet18":  lambda: _head(
            models.resnet18(weights=models.ResNet18_Weights.DEFAULT),
            "fc", None, n_classes),
        "ResNet34":  lambda: _head(
            models.resnet34(weights=models.ResNet34_Weights.DEFAULT),
            "fc", None, n_classes),
        "MobileNetV2": lambda: _head(
            models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT),
            "classifier", 1, n_classes),
        "MobileNetV3": lambda: _head(
            models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT),
            "classifier", 3, n_classes),
        "ShuffleNetV2": lambda: _head(
            models.shufflenet_v2_x1_0(weights=models.ShuffleNet_V2_X1_0_Weights.DEFAULT),
            "fc", None, n_classes),
        "SqueezeNet": lambda: _squeezenet_head(n_classes),
    }
    if backbone_name not in builders:
        raise ValueError(backbone_name)
    return builders[backbone_name]().to(device)


def _head(model: nn.Module, attr: str, idx, n_classes: int) -> nn.Module:
    layer = getattr(model, attr)
    if idx is None:
        in_f = layer.in_features
        setattr(model, attr, nn.Linear(in_f, n_classes))
    else:
        in_f = layer[idx].in_features
        layer[idx] = nn.Linear(in_f, n_classes)
    return model


def _squeezenet_head(n_classes: int) -> nn.Module:
    m = models.squeezenet1_1(weights=models.SqueezeNet1_1_Weights.DEFAULT)
    m.classifier[1] = nn.Conv2d(512, n_classes, kernel_size=(1, 1))
    m.num_classes   = n_classes
    return m


def _balanced_ce_weight(y: np.ndarray, n_classes: int) -> torch.Tensor:
    """Normalized class weights matching sklearn's 'balanced' formula."""
    counts = np.array([max(int((y == c).sum()), 1) for c in range(n_classes)],
                      dtype=np.float32)
    weights = len(y) / (n_classes * counts)
    return torch.tensor(weights, dtype=torch.float32).to(device)


def _train_cnn_fold(backbone_name: str,
                    X_tr: np.ndarray, y_tr: np.ndarray,
                    X_vl: np.ndarray, y_vl: np.ndarray,
                    epochs: int = CNN_EPOCHS) -> float:
    tf_tr = T.Compose([T.ToPILImage(),
                        T.Resize((IMG_SIZE_CNN, IMG_SIZE_CNN)),
                        T.RandomHorizontalFlip(),
                        T.ColorJitter(brightness=0.2, contrast=0.2),
                        T.ToTensor(),
                        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    tf_vl = T.Compose([T.ToPILImage(),
                        T.Resize((IMG_SIZE_CNN, IMG_SIZE_CNN)),
                        T.ToTensor(),
                        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    tr_dl = DataLoader(_ImgDataset(X_tr, y_tr, tf_tr),
                       batch_size=CNN_BATCH, shuffle=True,
                       num_workers=4, pin_memory=True, persistent_workers=True)
    vl_dl = DataLoader(_ImgDataset(X_vl, y_vl, tf_vl),
                       batch_size=CNN_BATCH, shuffle=False,
                       num_workers=4, pin_memory=True, persistent_workers=True)

    model     = _build_cnn(backbone_name, N_CLASSES)
    criterion = nn.CrossEntropyLoss(weight=_balanced_ce_weight(y_tr, N_CLASSES))
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    for _ in range(epochs):
        model.train()
        for imgs, lbls in tr_dl:
            imgs, lbls = imgs.to(device), lbls.to(device)
            optimizer.zero_grad()
            criterion(model(imgs), lbls).backward()
            optimizer.step()
        scheduler.step()

    model.eval()
    preds = []
    with torch.no_grad():
        for imgs, _ in vl_dl:
            preds.extend(model(imgs.to(device)).argmax(1).cpu().numpy())
    return f1_score(y_vl, preds, average="macro")


def phase3_finetuned_cnns(
    train_images: dict,
    test_images:  dict,
    le:           LabelEncoder,
    y_test:       np.ndarray,
    rf_baseline_scores: np.ndarray,
    n_repeats:    int = CV_REPEATS_CNN,
) -> tuple:
    """Returns (df_results, y_pred_holdout_best)."""
    print("\n" + "=" * 65)
    print("PHASE 3 — Fine-Tuned CNN Baselines")
    print("=" * 65)
    t0 = time.perf_counter()

    # Build flat RGB image arrays for PyTorch (pre-resized to IMG_SIZE_CNN)
    print("  Pre-building image arrays for PyTorch ...")
    flat_tr, y_arr = _flat_images_labels(train_images, le)
    X_arr = np.array([cv2.cvtColor(cv2.resize(img, (IMG_SIZE_CNN, IMG_SIZE_CNN)),
                                   cv2.COLOR_BGR2RGB)
                      for img in flat_tr], dtype=np.uint8)

    flat_te, _   = _flat_images_labels(test_images, le)
    X_arr_te     = np.array([cv2.cvtColor(cv2.resize(img, (IMG_SIZE_CNN, IMG_SIZE_CNN)),
                                          cv2.COLOR_BGR2RGB)
                              for img in flat_te], dtype=np.uint8)

    rows = []
    best_bb_name   = None
    best_bb_mean   = -1.0
    y_pred_holdout = None

    for bb_name in CNN_BACKBONES:
        print(f"  [{CNN_BACKBONES.index(bb_name)+1}/{len(CNN_BACKBONES)}] {bb_name} ...")
        cv_scores: list = []
        t1 = time.perf_counter()

        for rep in range(n_repeats):
            skf = StratifiedKFold(n_splits=CV_SPLITS, shuffle=True,
                                   random_state=SEED + rep)
            for tr_idx, vl_idx in skf.split(X_arr, y_arr):
                cv_scores.append(_train_cnn_fold(
                    bb_name,
                    X_arr[tr_idx], y_arr[tr_idx],
                    X_arr[vl_idx], y_arr[vl_idx]))

        arr    = np.array(cv_scores)
        t_fold = (time.perf_counter() - t1) / (n_repeats * CV_SPLITS)
        row    = compute_summary(f"FineTuned_{bb_name}", arr, rf_baseline_scores)
        row["Avg_CV_time_s"] = round(t_fold, 1)
        rows.append(row)
        print(f"    CV macro F1: {arr.mean():.4f} ± {arr.std():.4f}")

        if arr.mean() > best_bb_mean:
            best_bb_mean = arr.mean()
            best_bb_name = bb_name
            # Train on full training set + holdout evaluation
            tf_te = T.Compose([T.ToPILImage(),
                                T.Resize((IMG_SIZE_CNN, IMG_SIZE_CNN)),
                                T.ToTensor(),
                                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
            te_dl = DataLoader(_ImgDataset(X_arr_te, y_test, tf_te),
                               batch_size=CNN_BATCH, shuffle=False, num_workers=4,
                               pin_memory=True, persistent_workers=True)
            # Use all training folds (simplified: last trained model inside loop is
            # close enough — but here we train fresh on full set for the holdout)
            tf_tr = T.Compose([T.ToPILImage(), T.Resize((IMG_SIZE_CNN, IMG_SIZE_CNN)),
                                T.RandomHorizontalFlip(),
                                T.ColorJitter(brightness=0.2, contrast=0.2),
                                T.ToTensor(),
                                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
            tr_dl = DataLoader(_ImgDataset(X_arr, y_arr, tf_tr),
                               batch_size=CNN_BATCH, shuffle=True, num_workers=4,
                               pin_memory=True, persistent_workers=True)
            model_ho  = _build_cnn(bb_name, N_CLASSES)
            crit_ho   = nn.CrossEntropyLoss(weight=_balanced_ce_weight(y_arr, N_CLASSES))
            opt_ho    = optim.Adam(model_ho.parameters(), lr=1e-4, weight_decay=1e-5)
            sched_ho  = optim.lr_scheduler.CosineAnnealingLR(opt_ho, T_max=CNN_EPOCHS)
            for _ in range(CNN_EPOCHS):
                model_ho.train()
                for imgs, lbls in tr_dl:
                    imgs, lbls = imgs.to(device), lbls.to(device)
                    opt_ho.zero_grad()
                    crit_ho(model_ho(imgs), lbls).backward()
                    opt_ho.step()
                sched_ho.step()
            model_ho.eval()
            y_pred_holdout = []
            with torch.no_grad():
                for imgs, _ in te_dl:
                    y_pred_holdout.extend(model_ho(imgs.to(device)).argmax(1).cpu().numpy())
            y_pred_holdout = np.array(y_pred_holdout)

    df    = pd.DataFrame(rows)
    print(df.to_string(index=False))
    ho_f1 = f1_score(y_test, y_pred_holdout, average="macro") if y_pred_holdout is not None else 0.0
    n_err = int((y_pred_holdout != y_test).sum()) if y_pred_holdout is not None else -1
    print(f"\nBest Phase 3 ({best_bb_name}) Holdout macro F1: {ho_f1:.4f}  errors: {n_err}/{len(y_test)}")
    print(f"Phase 3 wall time: {time.perf_counter() - t0:.1f} s")
    df["Holdout_F1"] = None
    df.loc[df["Model"] == f"FineTuned_{best_bb_name}", "Holdout_F1"] = round(ho_f1, 4)
    return df, y_pred_holdout


# ── Phase 4: MorphNN Hybrid Fusion ────────────────────────────────────────────
def phase4_morphnn_hybrid(
    X_morph_train: np.ndarray, y_train: np.ndarray,
    X_morph_test:  np.ndarray, y_test:  np.ndarray,
    train_images:  dict,
    test_images:   dict,
    le:            LabelEncoder,
    rf_baseline_scores: np.ndarray,
    n_repeats:     int = CV_REPEATS,
) -> tuple:
    """Returns (df_results, y_pred_holdout_primary)."""
    print("\n" + "=" * 65)
    print("PHASE 4 — MorphNN Hybrid Fusion  "
          f"({N_MORPH_FEATURES} morph + {N_CNN_PCA_COMPS} CNN-PCA = {N_FUSED_FEATURES} dims)")
    print("=" * 65)
    t0 = time.perf_counter()

    hybrid_clfs = {
        "MorphNN_RF": RandomForestClassifier(n_estimators=200, class_weight="balanced",
                                             random_state=SEED, n_jobs=-1),
    }
    rows = []

    for bb_name in CNN_BACKBONES:
        print(f"\n  Backbone: {bb_name}")
        cnn = FrozenCNNExtractor(bb_name)

        X_cnn_tr, _ = extract_cnn_features_cached(train_images, cnn, le, split="train")
        del cnn
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        # NOTE: test CNN features used ONLY in the holdout block below, NOT in CV loop

        for clf_name, clf_proto in hybrid_clfs.items():
            cv_scores: list = []
            t1 = time.perf_counter()

            for rep in range(n_repeats):
                skf = StratifiedKFold(n_splits=CV_SPLITS, shuffle=True,
                                       random_state=SEED + rep)
                for tr_idx, vl_idx in skf.split(X_morph_train, y_train):
                    Xm_tr, Xm_vl = X_morph_train[tr_idx], X_morph_train[vl_idx]
                    ytr,   yvl   = y_train[tr_idx],        y_train[vl_idx]
                    Xc_tr, Xc_vl = X_cnn_tr[tr_idx],      X_cnn_tr[vl_idx]

                    pca     = PCA(n_components=N_CNN_PCA_COMPS, random_state=SEED)
                    Xc_tr_p = pca.fit_transform(Xc_tr)
                    Xc_vl_p = pca.transform(Xc_vl)

                    sc      = StandardScaler()
                    Xfus_tr = sc.fit_transform(np.hstack([Xm_tr, Xc_tr_p]))
                    Xfus_vl = sc.transform(       np.hstack([Xm_vl, Xc_vl_p]))

                    c = copy.deepcopy(clf_proto)
                    c.fit(Xfus_tr, ytr)
                    cv_scores.append(f1_score(yvl, c.predict(Xfus_vl), average="macro"))

            arr    = np.array(cv_scores)
            t_fold = (time.perf_counter() - t1) / (n_repeats * CV_SPLITS)
            label  = f"{clf_name}_{bb_name}"
            row    = compute_summary(label, arr, rf_baseline_scores)
            row["Avg_CV_time_s"] = round(t_fold, 2)
            rows.append(row)
            print(f"    {label}: {arr.mean():.4f} ± {arr.std():.4f}")

    # ── Primary MorphNN holdout (MobileNetV2 + RF) ──
    print(f"\n  Primary MorphNN holdout ({PRIMARY_BACKBONE} + RF) ...")
    cnn_p = FrozenCNNExtractor(PRIMARY_BACKBONE)
    X_cnn_tr, _ = extract_cnn_features_cached(train_images, cnn_p, le, split="train")
    X_cnn_te, _ = extract_cnn_features_cached(test_images,  cnn_p, le, split="test")

    pca_f = PCA(n_components=N_CNN_PCA_COMPS, random_state=SEED)
    Xc_tr = pca_f.fit_transform(X_cnn_tr)
    Xc_te = pca_f.transform(X_cnn_te)

    sc_f    = StandardScaler()
    Xfus_tr = sc_f.fit_transform(np.hstack([X_morph_train, Xc_tr]))
    Xfus_te = sc_f.transform(       np.hstack([X_morph_test,  Xc_te]))

    rf_f = RandomForestClassifier(n_estimators=500, class_weight="balanced",
                                  random_state=SEED, n_jobs=-1)
    t_tr = time.perf_counter()
    rf_f.fit(Xfus_tr, y_train)
    train_time = time.perf_counter() - t_tr

    y_pred_primary = rf_f.predict(Xfus_te)
    ho_f1 = f1_score(y_test, y_pred_primary, average="macro")
    n_err = int((y_pred_primary != y_test).sum())

    print(f"\n  PRIMARY MorphNN HOLDOUT RESULTS  ({PRIMARY_BACKBONE} + RF)")
    print(f"    Macro F1    : {ho_f1:.4f}")
    print(f"    Errors      : {n_err} / {len(y_test)}")
    print(f"    Train time  : {train_time:.2f} s")
    print("\n  Classification Report:")
    print(classification_report(y_test, y_pred_primary, target_names=le.classes_, digits=4))

    # Feature importance audit
    feat_names = (MorphFeatureExtractor.FEATURE_NAMES +
                  [f"CNN_PCA_{i+1}" for i in range(N_CNN_PCA_COMPS)])
    fi_df = (pd.DataFrame({"Feature": feat_names,
                            "Importance": rf_f.feature_importances_})
               .sort_values("Importance", ascending=False))
    print("\n  Feature importance (top 12):")
    print(fi_df.head(12).to_string(index=False))

    # Save fitted pipeline
    pipeline_path = os.path.join(OUTPUT_DIR, "morphnn_primary_pipeline.pkl")
    with open(pipeline_path, "wb") as fp:
        pickle.dump({"rf": rf_f, "pca": pca_f, "scaler": sc_f,
                     "le": le, "feature_names": feat_names}, fp)
    print(f"\n  Pipeline saved → {pipeline_path}")

    # Confusion matrix
    cm  = confusion_matrix(y_test, y_pred_primary)
    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=le.classes_, yticklabels=le.classes_, ax=ax)
    ax.set_title(f"MorphNN ({PRIMARY_BACKBONE} + RF) — 7-Class HAM Holdout")
    ax.set_ylabel("True"); ax.set_xlabel("Predicted")
    fig.tight_layout()
    cm_path = os.path.join(OUTPUT_DIR, "morphnn_confusion_matrix.png")
    fig.savefig(cm_path, dpi=150)
    plt.close(fig)

    # Feature importance figure
    fi_plot = fi_df.sort_values("Importance")
    colors  = ["#2196F3" if "CNN" in n else "#E91E63" for n in fi_plot["Feature"]]
    fig2, ax2 = plt.subplots(figsize=(9, 6))
    ax2.barh(fi_plot["Feature"], fi_plot["Importance"], color=colors)
    ax2.set_xlabel("Mean Decrease in Impurity")
    ax2.set_title("MorphNN Feature Importance\n(Pink = Morphology · Blue = CNN PCA)")
    fig2.tight_layout()
    fi_path = os.path.join(OUTPUT_DIR, "morphnn_feature_importance.png")
    fig2.savefig(fi_path, dpi=150)
    plt.close(fig2)
    print(f"  Figures saved → {OUTPUT_DIR}/")

    print(f"\nPhase 4 wall time: {time.perf_counter() - t0:.1f} s")

    df = pd.DataFrame(rows)
    primary_label = f"MorphNN_RF_{PRIMARY_BACKBONE}"
    df["Holdout_F1"] = None
    if primary_label in df["Model"].values:
        df.loc[df["Model"] == primary_label, "Holdout_F1"] = round(ho_f1, 4)
    return df, y_pred_primary


# ── Final Cross-Phase Summary ──────────────────────────────────────────────────
def final_summary(
    y_test:      np.ndarray,
    le:          LabelEncoder,
    y_pred_p1:   np.ndarray,
    y_pred_p2:   np.ndarray,
    y_pred_p3:   np.ndarray,
    y_pred_p4:   np.ndarray,
    df1: pd.DataFrame, df2: pd.DataFrame,
    df3: pd.DataFrame, df4: pd.DataFrame,
) -> None:
    print("\n" + "=" * 65)
    print("FINAL CROSS-PHASE SUMMARY")
    print("=" * 65)

    rows = []
    for phase, y_pred, label in [
        ("Phase 1 — Morphology-only RF",            y_pred_p1, "RF_morph"),
        ("Phase 2 — Best frozen backbone",           y_pred_p2, None),
        ("Phase 3 — Best fine-tuned CNN",            y_pred_p3, None),
        (f"Phase 4 — MorphNN ({PRIMARY_BACKBONE}+RF)", y_pred_p4, None),
    ]:
        if y_pred is None:
            continue
        f1  = f1_score(y_test, y_pred, average="macro")
        err = int((y_pred != y_test).sum())
        rows.append({"Phase": phase, "Holdout_MacroF1": round(f1, 4),
                     "Errors": err, "N_test": len(y_test)})

    summary_df = pd.DataFrame(rows)
    print(summary_df.to_string(index=False))

    # McNemar test: MorphNN (Phase 4) vs best fine-tuned CNN (Phase 3)
    if y_pred_p3 is not None and y_pred_p4 is not None:
        diff, p, desc = mcnemar_test(y_test, y_pred_p4, y_pred_p3)
        sig  = " (significant)" if p < ALPHA else " (not significant)"
        print(f"\nMcNemar test  (MorphNN vs best fine-tuned CNN): {desc}"
              f"  p = {p:.4f}{sig}")

    # Per-class F1 for primary MorphNN
    print("\nPer-class F1 — Primary MorphNN holdout:")
    per_cls = pd.DataFrame({
        "Class": le.classes_,
        "F1":    [round(f1_score(y_test == c, y_pred_p4 == c), 4)
                  for c in range(N_CLASSES)],
    })
    print(per_cls.to_string(index=False))

    # Save summary
    summary_df.to_csv(os.path.join(OUTPUT_DIR, "cross_phase_summary.csv"), index=False)
    per_cls.to_csv(  os.path.join(OUTPUT_DIR, "per_class_f1.csv"),         index=False)
    print(f"\nSummary tables saved → {OUTPUT_DIR}/")


# ── Persist Phase Results ──────────────────────────────────────────────────────
def _phase_cache(name: str) -> str:
    return os.path.join(CACHE_DIR, f"phase_{name}.pkl")


def _save_phase(name: str, df: pd.DataFrame, y_pred: np.ndarray | None = None) -> None:
    with open(_phase_cache(name), "wb") as f:
        pickle.dump({"df": df, "y_pred": y_pred}, f)


def _load_phase(name: str):
    path = _phase_cache(name)
    if not os.path.exists(path):
        return None, None
    with open(path, "rb") as f:
        d = pickle.load(f)
    return d["df"], d.get("y_pred")


def save_all_results(dfs: dict) -> None:
    path = os.path.join(OUTPUT_DIR, "all_results.xlsx")
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for sheet, df in dfs.items():
            if df is not None:
                df.to_excel(writer, sheet_name=sheet[:31], index=False)
    print(f"Excel results → {path}")


# ── Main ───────────────────────────────────────────────────────────────────────
def main() -> None:
    set_seed(SEED)
    t_global = time.perf_counter()

    print("=" * 65)
    print("MorphNN 7-Class HAM10000 Skin Lesion Benchmark")
    print(f"Device : {device}")
    print(f"Dataset: {os.path.abspath(DATASET_ROOT)}")
    print(f"Vector : {N_MORPH_FEATURES} morph + {N_CNN_PCA_COMPS} CNN-PCA = {N_FUSED_FEATURES} dims")
    print(f"Classes: {CLASSES}")
    print("=" * 65)

    # ── Load images ──────────────────────────────────────────────────────────
    print("\nLoading training images ...")
    train_raw = load_images_from_dir(TRAIN_DIR)
    for cls, imgs in train_raw.items():
        print(f"  {cls}: {len(imgs)}")

    print(f"\nAugmenting minority classes to ≥ {AUG_TARGET_PER_CLASS} ...")
    train_aug = augment_to_target(train_raw, target=AUG_TARGET_PER_CLASS)
    for cls, imgs in train_aug.items():
        print(f"  {cls}: {len(imgs)}")

    print("\nLoading test images ...")
    test_raw = load_images_from_dir(TEST_DIR)
    for cls, imgs in test_raw.items():
        print(f"  {cls}: {len(imgs)}")

    # ── Morphological features ───────────────────────────────────────────────
    morph_ext = MorphFeatureExtractor()

    print("\nMorphological feature extraction — train ...")
    X_morph_tr, y_train, le = extract_morph_features_cached(
        train_aug, morph_ext, split="train")
    print(f"  X_morph_train: {X_morph_tr.shape}")

    print("\nMorphological feature extraction — test ...")
    X_morph_te, y_test, _  = extract_morph_features_cached(
        test_raw,  morph_ext, split="test")
    print(f"  X_morph_test : {X_morph_te.shape}")

    # ── Phase 1 ──────────────────────────────────────────────────────────────
    df1, y_pred_p1 = _load_phase("1")[:2]
    rf_baseline_scores = None

    if df1 is None:
        df1_full, rf_baseline_scores, y_pred_p1 = phase1_morphology_only(
            X_morph_tr, y_train, X_morph_te, y_test, le)
        _save_phase("1", df1_full, y_pred_p1)
        # Also cache the RF CV scores separately (needed for Wilcoxon in later phases)
        np.save(_cache_path("rf_baseline_cv_scores.npy"), rf_baseline_scores)
        df1 = df1_full
    else:
        print("\n[Phase 1 loaded from cache]")
        rf_baseline_scores = np.load(_cache_path("rf_baseline_cv_scores.npy"))

    # ── Phase 2 ──────────────────────────────────────────────────────────────
    df2, y_pred_p2 = _load_phase("2")

    if df2 is None:
        df2, y_pred_p2 = phase2_frozen_transfer(
            train_aug, test_raw, le, y_test, rf_baseline_scores)
        _save_phase("2", df2, y_pred_p2)
    else:
        print("\n[Phase 2 loaded from cache]")

    # ── Phase 3 ──────────────────────────────────────────────────────────────
    df3, y_pred_p3 = _load_phase("3")

    if df3 is None:
        df3, y_pred_p3 = phase3_finetuned_cnns(
            train_aug, test_raw, le, y_test, rf_baseline_scores)
        _save_phase("3", df3, y_pred_p3)
    else:
        print("\n[Phase 3 loaded from cache]")

    # ── Phase 4 ──────────────────────────────────────────────────────────────
    df4, y_pred_p4 = _load_phase("4")

    if df4 is None:
        df4, y_pred_p4 = phase4_morphnn_hybrid(
            X_morph_tr, y_train, X_morph_te, y_test,
            train_aug, test_raw, le, rf_baseline_scores)
        _save_phase("4", df4, y_pred_p4)
    else:
        print("\n[Phase 4 loaded from cache]")

    # ── Final summary ─────────────────────────────────────────────────────────
    final_summary(y_test, le,
                  y_pred_p1, y_pred_p2, y_pred_p3, y_pred_p4,
                  df1, df2, df3, df4)

    save_all_results({
        "Phase1_Morphology": df1,
        "Phase2_FrozenCNN":  df2,
        "Phase3_FineTuned":  df3,
        "Phase4_MorphNN":    df4,
    })

    print(f"\n✓ Complete.  Total wall time: {time.perf_counter() - t_global:.1f} s")
    print(f"  Results → {os.path.abspath(OUTPUT_DIR)}")
    print(f"  Cache   → {os.path.abspath(CACHE_DIR)}")


if __name__ == "__main__":
    main()
