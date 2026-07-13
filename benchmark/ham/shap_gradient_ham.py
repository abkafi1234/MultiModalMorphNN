"""Gradient-based morphological feature attribution for MorphNN on HAM10000.

Method: Input × Gradient (Integrated Gradients approximation).
  1. Train one MorphNN model on full training data (20 epochs).
  2. For each test image, compute gradient of P(predicted class) w.r.t. morph input.
  3. Attribution = |gradient × morph_input| per feature.
  4. Average over test set → mean absolute attribution per feature.

Produces:
  shap_morph_attribution_HAM.csv      — per-feature mean |grad×input|
  shap_morph_attribution_HAM.png      — bar chart: aligned vs misaligned features
  shap_morph_attribution_heatmap_HAM.png — per-class attribution heatmap
  morph_feature_variance_HAM.png      — variance of each feature (confirms misalignment)
"""
import sys, os
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[2] / "configs"))
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[2]))
from pathlib import Path
import numpy as np, cv2, time, pickle
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, accuracy_score
import torch, torch.nn as nn, torch.optim as optim
import torchvision.models as models
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd

ROOT      = Path(__file__).resolve().parents[2]
CACHE_DIR = ROOT / "ham_cache"
DATA_DIR  = ROOT / "Ham Dataset"
OUT       = Path(__file__).resolve().parent

SEED      = 42
N_CLASSES = 7
CLASSES   = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
EPOCHS    = 20
CNN_BATCH = 32
AUG_TARGET = 2000

FEATURE_NAMES = [
    "lesion_count",        # f1  idx0  MISALIGNED
    "avg_lesion_area",     # f2  idx1
    "area_heterogeneity",  # f3  idx2
    "avg_circularity",     # f4  idx3
    "sparsity_score",      # f5  idx4  MISALIGNED
    "confluence_density",  # f6  idx5  MISALIGNED
    "localized_hue",       # f7  idx6
    "localized_saturation",# f8  idx7
    "avg_aspect_ratio",    # f9  idx8
    "avg_solidity",        # f10 idx9
    "localized_value",     # f11 idx10
    "hue_std",             # f12 idx11
    "saturation_std",      # f13 idx12
    "spatial_entropy",     # f14 idx13 MISALIGNED
    "max_area_ratio",      # f15 idx14 MISALIGNED
    "background_sat",      # f16 idx15
]

MISALIGNED_IDX = {0, 4, 5, 13, 14}   # f1,f5,f6,f14,f15

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n{'='*60}\nSHAP Gradient Attribution — HAM10000 | Device: {device}\n{'='*60}")

TF_TRAIN = T.Compose([T.ToPILImage(), T.Resize(224), T.RandomHorizontalFlip(),
                       T.RandomRotation(20), T.ColorJitter(0.2,0.2,0.1,0.05),
                       T.ToTensor(), T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
TF_EVAL  = T.Compose([T.ToPILImage(), T.Resize(224), T.CenterCrop(224),
                       T.ToTensor(), T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])

class JointMorphCNN(nn.Module):
    def __init__(self):
        super().__init__()
        base = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT)
        self.features = base.features
        self.avgpool  = base.avgpool
        self.morph_fc = nn.Sequential(nn.Linear(16, 64), nn.Hardswish(), nn.Dropout(0.3))
        self.head = nn.Sequential(
            nn.Linear(960+64, 512), nn.Hardswish(), nn.Dropout(0.2),
            nn.Linear(512, N_CLASSES)
        )
    def forward(self, img, morph):
        x = self.avgpool(self.features(img)).flatten(1)
        return self.head(torch.cat([x, self.morph_fc(morph)], dim=1))

class _JointDs(Dataset):
    def __init__(self, imgs, morphs, labels, tf):
        self.imgs   = imgs
        self.morphs = torch.tensor(morphs, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.tf     = tf
    def __len__(self): return len(self.imgs)
    def __getitem__(self, i):
        return self.tf(cv2.cvtColor(self.imgs[i], cv2.COLOR_BGR2RGB)), self.morphs[i], self.labels[i]

def _ce_weights(y):
    y = np.asarray(y, dtype=np.int64)
    counts = np.array([max(int((y==c).sum()),1) for c in range(N_CLASSES)], np.float32)
    return torch.tensor(len(y)/(N_CLASSES*counts)).to(device)

def finetune_joint(imgs_tr, morphs_tr, y_in):
    y_arr = np.asarray(y_in, dtype=np.int64)
    model = JointMorphCNN().to(device)
    ds = _JointDs(imgs_tr, morphs_tr, y_arr, TF_TRAIN)
    dl = DataLoader(ds, batch_size=CNN_BATCH, shuffle=True,
                    num_workers=4, pin_memory=True, persistent_workers=True)
    crit = nn.CrossEntropyLoss(weight=_ce_weights(y_arr), label_smoothing=0.05)
    opt  = optim.AdamW([
        {'params': list(model.features.parameters())+list(model.avgpool.parameters()), 'lr':1e-4},
        {'params': list(model.morph_fc.parameters())+list(model.head.parameters()),    'lr':1e-3},
    ], weight_decay=1e-4)
    sched = optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=10, T_mult=1)
    for ep in range(EPOCHS):
        model.train()
        for img, morph, lbl in dl:
            img, morph, lbl = img.to(device), morph.to(device), lbl.to(device)
            opt.zero_grad(); crit(model(img, morph), lbl).backward(); opt.step()
        sched.step()
        print(f"  Epoch {ep+1}/{EPOCHS} done")
    return model.eval()

def load_morph(split):
    X = np.load(CACHE_DIR / f"morph_{split}_X.npy").astype(np.float32)
    y = np.load(CACHE_DIR / f"morph_{split}_y.npy").astype(np.int64)
    return X, y

def load_flat_images(split):
    le = pickle.load(open(CACHE_DIR/"label_encoder.pkl","rb"))
    imgs, ys = [], []
    for cls in sorted(os.listdir(DATA_DIR/split)):
        cls_dir = DATA_DIR/split/cls
        if not cls_dir.is_dir(): continue
        lbl = int(le.transform([cls])[0])
        for f_ in sorted(os.listdir(cls_dir)):
            img = cv2.imread(str(cls_dir/f_))
            if img is not None:
                imgs.append(cv2.resize(img,(224,224))); ys.append(lbl)
    return imgs, np.array(ys, dtype=np.int64)

def load_augmented_flat():
    rng = np.random.default_rng(SEED)
    le = pickle.load(open(CACHE_DIR/"label_encoder.pkl","rb"))
    by_cls = {}
    for cls in sorted(os.listdir(DATA_DIR/"train")):
        cls_dir = DATA_DIR/"train"/cls
        if not cls_dir.is_dir(): continue
        imgs = []
        for f_ in sorted(os.listdir(cls_dir)):
            img = cv2.imread(str(cls_dir/f_))
            if img is not None: imgs.append(cv2.resize(img,(224,224)))
        by_cls[cls] = imgs
    flat_imgs, flat_y = [], []
    for cls, imgs in by_cls.items():
        lbl = int(le.transform([cls])[0])
        pool = imgs[:]
        if len(imgs) < AUG_TARGET:
            for i in range(AUG_TARGET-len(imgs)):
                src = imgs[i % len(imgs)].copy()
                if rng.random()>0.5: src=cv2.flip(src,1)
                ang = rng.uniform(-20,20)
                M = cv2.getRotationMatrix2D((112,112),ang,1.)
                src = cv2.warpAffine(src,M,(224,224),borderMode=cv2.BORDER_REFLECT_101)
                pool.append(src)
        for img in pool: flat_imgs.append(img); flat_y.append(lbl)
    return flat_imgs, np.array(flat_y, dtype=np.int64)

# ── Step 0: Feature variance analysis (no training needed) ────────────────────
print("\n[0] Feature variance analysis on HAM10000 test set")
X_te_raw, y_te = load_morph("test")
X_tr_raw, y_tr = load_morph("train")

variances = X_te_raw.std(axis=0)
print("Per-feature std (test):")
for i, (name, std) in enumerate(zip(FEATURE_NAMES, variances)):
    tag = " ← MISALIGNED" if i in MISALIGNED_IDX else ""
    print(f"  f{i+1:2d} {name:25s}: std={std:.4f}{tag}")

# ── Step 1: Train full model ──────────────────────────────────────────────────
sc = StandardScaler()
X_tr_n = sc.fit_transform(X_tr_raw).astype(np.float32)
X_te_n = sc.transform(X_te_raw).astype(np.float32)

print(f"\n[1] Training MorphNN for gradient attribution ({EPOCHS} epochs) ...")
t0 = time.perf_counter()
flat_tr_imgs, _ = load_augmented_flat()
flat_te_imgs, _ = load_flat_images("test")
flat_tr_imgs = np.array(flat_tr_imgs, dtype=np.uint8)
flat_te_imgs = np.array(flat_te_imgs, dtype=np.uint8)

model = finetune_joint(list(flat_tr_imgs), X_tr_n, y_tr)
print(f"Training done in {time.perf_counter()-t0:.0f}s")

# Verify model quality
with torch.no_grad():
    preds = []
    for s in range(0, len(flat_te_imgs), 64):
        e = min(s+64, len(flat_te_imgs))
        t_imgs  = torch.stack([TF_EVAL(cv2.cvtColor(flat_te_imgs[i], cv2.COLOR_BGR2RGB)) for i in range(s,e)]).to(device)
        t_morph = torch.tensor(X_te_n[s:e], dtype=torch.float32).to(device)
        preds.append(model(t_imgs, t_morph).argmax(1).cpu().numpy())
y_pred = np.concatenate(preds)
print(f"Holdout Macro F1: {f1_score(y_te, y_pred, average='macro'):.4f}  Acc: {accuracy_score(y_te, y_pred):.4f}")

# ── Step 2: Gradient × Input attribution ─────────────────────────────────────
print("\n[2] Computing Input×Gradient morph attribution on test set ...")

model.train()  # enable gradients through dropout for more realistic attribution
all_grad_x_input = []   # shape: (n_test, 16)
all_class_preds  = []
BATCH = 32

for s in range(0, len(flat_te_imgs), BATCH):
    e = min(s+BATCH, len(flat_te_imgs))
    t_imgs  = torch.stack([TF_EVAL(cv2.cvtColor(flat_te_imgs[i], cv2.COLOR_BGR2RGB)) for i in range(s,e)]).to(device)
    # Move to device FIRST, then set requires_grad so the tensor is a leaf node
    t_morph = torch.tensor(X_te_n[s:e], dtype=torch.float32).to(device)
    t_morph.requires_grad_(True)

    logits = model(t_imgs, t_morph)             # (batch, 7)
    pred_classes = logits.argmax(dim=1)         # predicted class per sample
    all_class_preds.append(pred_classes.detach().cpu().numpy())

    # Score = sum of predicted-class logits
    score = logits[range(len(pred_classes)), pred_classes].sum()
    score.backward()

    grad = t_morph.grad.detach().cpu().numpy()  # (batch, 16)
    inp  = X_te_n[s:e]                          # (batch, 16)
    all_grad_x_input.append(np.abs(grad * inp))

model.eval()

gxi = np.vstack(all_grad_x_input)   # (n_test, 16)
class_preds = np.concatenate(all_class_preds)
print(f"Computed attribution for {len(gxi)} test samples")

# ── Step 3: Aggregate and save ────────────────────────────────────────────────
mean_attr   = gxi.mean(axis=0)       # (16,)
median_attr = np.median(gxi, axis=0)

df_attr = pd.DataFrame({
    "feature_idx": range(16),
    "feature_name": FEATURE_NAMES,
    "aligned": ["No" if i in MISALIGNED_IDX else "Yes" for i in range(16)],
    "mean_abs_grad_x_input": mean_attr,
    "median_abs_grad_x_input": median_attr,
    "test_std": variances,
})
df_attr.sort_values("mean_abs_grad_x_input", ascending=False, inplace=True)
df_attr.to_csv(OUT / "shap_morph_attribution_HAM.csv", index=False)
print("\nFeature attribution ranking:")
print(df_attr[["feature_name","aligned","mean_abs_grad_x_input"]].to_string(index=False))

# ── Step 4: Attribution bar chart ────────────────────────────────────────────
COLORS = {"Yes": "#2563eb", "No": "#dc2626"}
fig, ax = plt.subplots(figsize=(13, 5.5))
order = np.argsort(mean_attr)[::-1]
names_sorted  = [FEATURE_NAMES[i] for i in order]
attrs_sorted  = mean_attr[order]
align_sorted  = ["No" if i in MISALIGNED_IDX else "Yes" for i in order]
bar_colors    = [COLORS[a] for a in align_sorted]

bars = ax.bar(range(16), attrs_sorted, color=bar_colors, edgecolor="white", linewidth=0.5)
ax.set_xticks(range(16))
ax.set_xticklabels(names_sorted, rotation=45, ha="right", fontsize=8.5)
ax.set_ylabel("Mean |Gradient × Input|", fontsize=11)
ax.set_title("MorphNN — Morphological Feature Attribution on HAM10000\n(gradient-based; higher = more influential to prediction)", fontsize=10, pad=8)
ax.grid(axis="y", alpha=0.3)

legend_patches = [
    mpatches.Patch(color="#2563eb", label="Domain-aligned (11 features)"),
    mpatches.Patch(color="#dc2626", label="Misaligned: multi-lesion features (5 features)"),
]
ax.legend(handles=legend_patches, fontsize=9, loc="upper right")

# Annotate mean value
for bar, val in zip(bars, attrs_sorted):
    ax.text(bar.get_x()+bar.get_width()/2, val+max(attrs_sorted)*0.01,
            f"{val:.3f}", ha="center", va="bottom", fontsize=7, rotation=90)

plt.tight_layout()
p = OUT / "shap_morph_attribution_HAM.png"
fig.savefig(p, dpi=200, bbox_inches="tight"); plt.close(fig)
print(f"\n  saved → {p}")

# ── Step 5: Per-class attribution heatmap ────────────────────────────────────
heat = np.zeros((N_CLASSES, 16))
for c in range(N_CLASSES):
    mask = class_preds == c
    if mask.sum() > 0:
        heat[c] = gxi[mask].mean(axis=0)

fig, ax = plt.subplots(figsize=(14, 5))
# Normalise per feature for visual clarity
heat_norm = heat / (heat.max(axis=0, keepdims=True) + 1e-9)
im = ax.imshow(heat_norm, cmap="YlOrRd", aspect="auto", vmin=0, vmax=1)
ax.set_xticks(range(16))
ax.set_xticklabels(FEATURE_NAMES, rotation=45, ha="right", fontsize=8)
ax.set_yticks(range(N_CLASSES))
ax.set_yticklabels(CLASSES, fontsize=10)
ax.set_title("Per-Class Morphological Feature Attribution (HAM10000)\nNormalised mean |Gradient × Input| — red = high attribution", fontsize=10, pad=8)
plt.colorbar(im, ax=ax, fraction=0.02, pad=0.02, label="Normalised attribution")

# Highlight misaligned columns with border
for idx in MISALIGNED_IDX:
    ax.axvline(idx - 0.5, color="#dc2626", linewidth=2.5, alpha=0.7)
    ax.axvline(idx + 0.5, color="#dc2626", linewidth=2.5, alpha=0.7)

plt.tight_layout()
p = OUT / "shap_morph_attribution_heatmap_HAM.png"
fig.savefig(p, dpi=200, bbox_inches="tight"); plt.close(fig)
print(f"  saved → {p}")

# ── Step 6: Feature variance plot (confirms segmentation problem) ─────────────
fig, ax = plt.subplots(figsize=(13, 4.5))
bar_colors_var = [COLORS["No"] if i in MISALIGNED_IDX else COLORS["Yes"] for i in range(16)]
ax.bar(range(16), variances / variances.max(), color=bar_colors_var, edgecolor="white")
ax.set_xticks(range(16))
ax.set_xticklabels(FEATURE_NAMES, rotation=45, ha="right", fontsize=8.5)
ax.set_ylabel("Normalised Std Dev (HAM test)", fontsize=11)
ax.set_title("Feature Discriminability on HAM10000 — Low variance = low information\n(red = multi-lesion features designed for distributed rashes, misaligned with single-lesion dermoscopy)", fontsize=9, pad=8)
ax.legend(handles=legend_patches, fontsize=9)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
p = OUT / "morph_feature_variance_HAM.png"
fig.savefig(p, dpi=200, bbox_inches="tight"); plt.close(fig)
print(f"  saved → {p}")

# ── Summary printout ──────────────────────────────────────────────────────────
print("\n" + "="*60)
print("ATTRIBUTION SUMMARY FOR PAPER:")
print("="*60)
misaligned_mean = mean_attr[list(MISALIGNED_IDX)].mean()
aligned_mean    = mean_attr[[i for i in range(16) if i not in MISALIGNED_IDX]].mean()
print(f"Mean attribution — ALIGNED features:    {aligned_mean:.4f}")
print(f"Mean attribution — MISALIGNED features: {misaligned_mean:.4f}")
print(f"Ratio (aligned / misaligned):           {aligned_mean/misaligned_mean:.2f}x")
print(f"\nIndividual misaligned feature attributions:")
for i in sorted(MISALIGNED_IDX):
    print(f"  {FEATURE_NAMES[i]:25s}: {mean_attr[i]:.4f}  (rank {list(np.argsort(mean_attr)[::-1]).index(i)+1}/16)")
print(f"\nAll figures saved to: {OUT}")
