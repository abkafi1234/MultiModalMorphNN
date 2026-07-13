"""C21-HAM — End-to-end Joint CNN + Morph model for HAM10000 (7 classes).
Same architecture as C21 for MCVSLD, adapted for 7-class skin lesion classification.
MobileNetV3 features (960-d) fused with morph_fc(16→64) at head. No PCA, no RF.
"""
import sys, os
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[2] / "configs"))
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[2]))
from pathlib import Path
import numpy as np, cv2, time, pickle
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import f1_score, classification_report
import torch, torch.nn as nn, torch.optim as optim
import torchvision.models as models
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader

# ── HAM-specific constants ─────────────────────────────────────────────────────
HAM_ROOT  = Path(__file__).resolve().parents[2]
CACHE_DIR = HAM_ROOT / "ham_cache"
DATA_DIR  = HAM_ROOT / "Ham Dataset"
RES       = Path(__file__).parent / "results"
os.makedirs(RES, exist_ok=True)

SEED      = 42
N_CLASSES = 7
CLASSES   = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
EPOCHS    = 20
CV_SPLITS = 5
CV_REPS   = 3
CNN_BATCH = 32
AUG_TARGET = 2000
CFG        = "C21_joint_morphcnn_HAM"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TF_TRAIN = T.Compose([T.ToPILImage(), T.Resize(224), T.RandomHorizontalFlip(),
                       T.RandomRotation(20), T.ColorJitter(0.2,0.2,0.1,0.05),
                       T.ToTensor(), T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
TF_EVAL  = T.Compose([T.ToPILImage(), T.Resize(224), T.CenterCrop(224),
                       T.ToTensor(), T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])

# ── Joint model ────────────────────────────────────────────────────────────────
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

def finetune_joint(imgs_tr, morphs_tr, y_in, epochs=EPOCHS):
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
    for ep in range(epochs):
        model.train()
        for img, morph, lbl in dl:
            img, morph, lbl = img.to(device), morph.to(device), lbl.to(device)
            opt.zero_grad(); crit(model(img, morph), lbl).backward(); opt.step()
        sched.step()
    return model.eval()

@torch.no_grad()
def predict_joint(model, imgs, morphs, batch=64):
    preds = []
    for s in range(0, len(imgs), batch):
        e = min(s+batch, len(imgs))
        t_imgs  = torch.stack([TF_EVAL(cv2.cvtColor(imgs[i], cv2.COLOR_BGR2RGB)) for i in range(s,e)]).to(device)
        t_morph = torch.tensor(morphs[s:e], dtype=torch.float32).to(device)
        preds.append(model(t_imgs, t_morph).argmax(dim=1).cpu().numpy())
    return np.concatenate(preds) if preds else np.array([], dtype=np.int64)

# ── Load cached HAM data ───────────────────────────────────────────────────────
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
        for f in sorted(os.listdir(cls_dir)):
            img = cv2.imread(str(cls_dir/f))
            if img is not None:
                imgs.append(cv2.resize(img,(224,224))); ys.append(lbl)
    return imgs, np.array(ys, dtype=np.int64)

def load_augmented_flat():
    import random; rng = np.random.default_rng(SEED)
    le = pickle.load(open(CACHE_DIR/"label_encoder.pkl","rb"))
    by_cls = {}
    for cls in sorted(os.listdir(DATA_DIR/"train")):
        cls_dir = DATA_DIR/"train"/cls
        if not cls_dir.is_dir(): continue
        imgs = []
        for f in sorted(os.listdir(cls_dir)):
            img = cv2.imread(str(cls_dir/f))
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

# ── Bootstrap CI + stat test ───────────────────────────────────────────────────
from scipy import stats as scipy_stats

def bootstrap_ci(scores, n=2000):
    rng = np.random.default_rng(SEED)
    m = [rng.choice(scores,len(scores),replace=True).mean() for _ in range(n)]
    return float(scores.mean()), float(np.percentile(m,2.5)), float(np.percentile(m,97.5))

def stat_test(a, b):
    try:
        if len(a)==len(b):
            s,p = scipy_stats.wilcoxon(a,b,alternative="two-sided"); pfx="W"
        else:
            s,p = scipy_stats.mannwhitneyu(a,b,alternative="two-sided"); pfx="MWU"
        return f"{pfx} p={'<0.001' if p<0.001 else f'{p:.3f}'}{'*' if p<0.05 else ''}"
    except: return "N/A"

def save_results(cv_scores, y_te, y_pred):
    import pandas as pd
    le = pickle.load(open(CACHE_DIR/"label_encoder.pkl","rb"))
    baseline = np.load(CACHE_DIR/"rf_baseline_cv_scores.npy")
    m, lo, hi = bootstrap_ci(np.array(cv_scores))
    ho = f1_score(y_te, y_pred, average="macro")
    rep = classification_report(y_te, y_pred, target_names=le.classes_, digits=4)
    row = {"Config": CFG, "CV_MacroF1": round(m,4), "CV_Std": round(float(np.array(cv_scores).std(ddof=1)),4),
           "CI95": f"[{lo:.4f},{hi:.4f}]", "Holdout_MacroF1": round(ho,4),
           "Errors": int((y_pred!=y_te).sum()), "N_test": len(y_te),
           "vs_RF_baseline": stat_test(np.array(cv_scores), baseline)}
    pd.DataFrame([row]).to_csv(RES/"summary.csv", index=False)
    with open(RES/"classification_report.txt","w") as f: f.write(f"Config: {CFG}\n\n{rep}")
    np.save(RES/"y_pred.npy", y_pred); np.save(RES/"y_test.npy", y_te)
    print("\n"+"="*55)
    for k,v in row.items(): print(f"  {k}: {v}")
    print(f"\n{rep}")

# ── Main ───────────────────────────────────────────────────────────────────────
print(f"\n{'='*55}\n{CFG}  |  HAM10000 7-class Joint MorphCNN\n{'='*55}")
t0 = time.perf_counter()

X_morph_tr, y_tr = load_morph("train")
X_morph_te, y_te = load_morph("test")
print(f"Morph train: {X_morph_tr.shape}, test: {X_morph_te.shape}")

sc = StandardScaler()
X_morph_tr_n = sc.fit_transform(X_morph_tr).astype(np.float32)
X_morph_te_n = sc.transform(X_morph_te).astype(np.float32)

print("Loading augmented training images ...")
flat_tr_imgs, _ = load_augmented_flat()
flat_te_imgs, _ = load_flat_images("test")
flat_tr_imgs = np.array(flat_tr_imgs, dtype=np.uint8)
flat_te_imgs = np.array(flat_te_imgs, dtype=np.uint8)
print(f"Train images: {len(flat_tr_imgs)}, Test images: {len(flat_te_imgs)}")

cv_scores = []
for rep in range(CV_REPS):
    skf = StratifiedKFold(n_splits=CV_SPLITS, shuffle=True, random_state=SEED+rep)
    for fold, (tr_i, vl_i) in enumerate(skf.split(X_morph_tr, y_tr)):
        t1 = time.perf_counter()
        model = finetune_joint(list(flat_tr_imgs[tr_i]), X_morph_tr_n[tr_i], y_tr[tr_i])
        y_pred_vl = predict_joint(model, list(flat_tr_imgs[vl_i]), X_morph_tr_n[vl_i])
        f1 = f1_score(y_tr[vl_i], y_pred_vl, average="macro")
        cv_scores.append(f1)
        print(f"  Rep {rep+1} Fold {fold+1}  F1={f1:.4f}  ({time.perf_counter()-t1:.0f}s)  mean:{np.mean(cv_scores):.4f}")
        del model; torch.cuda.empty_cache() if torch.cuda.is_available() else None

print("\nFull fine-tune for holdout ...")
model_h = finetune_joint(list(flat_tr_imgs), X_morph_tr_n, y_tr)
y_pred  = predict_joint(model_h, list(flat_te_imgs), X_morph_te_n)

save_results(cv_scores, y_te, y_pred)
print(f"Wall time: {time.perf_counter()-t0:.1f}s")
