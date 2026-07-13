"""C21 — End-to-end Joint CNN + Morph model.
Architecture: MobileNetV3 features (960-d) fused with morph_fc(16→64) at the head.
Trained jointly end-to-end — morph features guide the CNN's decision boundary.
No RF, no PCA — direct softmax classification.
Separate LRs: backbone=1e-4, head+morph_fc=1e-3.
AdamW, CosineAnnealingWarmRestarts(T_0=10), label_smoothing=0.05, 20 epochs.
"""
import sys; sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[1]))
from shared_utils import *
from shared_utils import _ce_weights

CFG    = "C21_joint_morphcnn"
RES    = Path(__file__).parent / "results"
EPOCHS = 20

# ── Joint model ────────────────────────────────────────────────────────────────
class JointMorphCNN(nn.Module):
    """MobileNetV3-Large backbone + morphology branch fused before classification."""
    def __init__(self):
        super().__init__()
        base = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT)
        self.features = base.features          # → (B, 960, 7, 7) for 224×224 input
        self.avgpool  = base.avgpool           # AdaptiveAvgPool2d(1) → (B, 960, 1, 1)
        self.morph_fc = nn.Sequential(
            nn.Linear(16, 64), nn.Hardswish(), nn.Dropout(0.3)
        )
        self.head = nn.Sequential(
            nn.Linear(960 + 64, 512), nn.Hardswish(), nn.Dropout(0.2),
            nn.Linear(512, N_CLASSES)
        )

    def forward(self, img, morph):
        x = self.avgpool(self.features(img)).flatten(1)   # (B, 960)
        m = self.morph_fc(morph)                           # (B, 64)
        return self.head(torch.cat([x, m], dim=1))         # (B, N_CLASSES)

class _JointDs(Dataset):
    def __init__(self, imgs_bgr, morphs, labels, tf):
        self.imgs   = imgs_bgr
        self.morphs = torch.tensor(morphs, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.tf     = tf
    def __len__(self): return len(self.imgs)
    def __getitem__(self, i):
        img = self.tf(cv2.cvtColor(self.imgs[i], cv2.COLOR_BGR2RGB))
        return img, self.morphs[i], self.labels[i]

def finetune_joint(imgs_tr, morphs_tr, y_in, epochs=EPOCHS):
    y_arr = np.asarray(y_in, dtype=np.int64)
    model = JointMorphCNN().to(device)
    ds = _JointDs(imgs_tr, morphs_tr, y_arr, TF_TRAIN)
    dl = DataLoader(ds, batch_size=CNN_BATCH, shuffle=True,
                    num_workers=4, pin_memory=True, persistent_workers=True)
    crit = nn.CrossEntropyLoss(weight=_ce_weights(y_arr), label_smoothing=0.05)
    opt  = optim.AdamW([
        {'params': list(model.features.parameters()) + list(model.avgpool.parameters()),
         'lr': 1e-4},
        {'params': list(model.morph_fc.parameters()) + list(model.head.parameters()),
         'lr': 1e-3},
    ], weight_decay=1e-4)
    sched = optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=10, T_mult=1)
    for ep in range(epochs):
        model.train()
        for img, morph, lbl in dl:
            img, morph, lbl = img.to(device), morph.to(device), lbl.to(device)
            opt.zero_grad()
            crit(model(img, morph), lbl).backward()
            opt.step()
        sched.step()
    return model.eval()

@torch.no_grad()
def predict_joint(model, imgs_bgr, morphs, batch=64):
    preds = []
    for s in range(0, len(imgs_bgr), batch):
        end = min(s + batch, len(imgs_bgr))
        t_imgs  = torch.stack([TF_EVAL(cv2.cvtColor(imgs_bgr[i], cv2.COLOR_BGR2RGB))
                                for i in range(s, end)]).to(device)
        t_morph = torch.tensor(morphs[s:end], dtype=torch.float32).to(device)
        preds.append(model(t_imgs, t_morph).argmax(dim=1).cpu().numpy())
    return np.concatenate(preds) if preds else np.array([], dtype=np.int64)

print(f"\n{'='*55}\n{CFG}  |  Joint end-to-end CNN+Morph ({EPOCHS} ep)\n{'='*55}")
t0 = time.perf_counter()

X_morph_tr, y_tr = load_morph("train")
X_morph_te, y_te = load_morph("test")
le = load_le(); baseline = load_rf_baseline()

# Normalise morph features before feeding to FC layer
sc_morph = StandardScaler()
X_morph_tr_n = sc_morph.fit_transform(X_morph_tr).astype(np.float32)
X_morph_te_n = sc_morph.transform(X_morph_te).astype(np.float32)

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
        model = finetune_joint(
            list(flat_tr_imgs[tr_i]), X_morph_tr_n[tr_i], y_tr[tr_i]
        )
        y_pred_vl = predict_joint(model, list(flat_tr_imgs[vl_i]), X_morph_tr_n[vl_i])
        f1 = f1_score(y_tr[vl_i], y_pred_vl, average="macro")
        cv_scores.append(f1)
        print(f"  Rep {rep+1} Fold {fold+1}  F1={f1:.4f}  ({time.perf_counter()-t1:.0f}s)  mean:{np.mean(cv_scores):.4f}")
        del model

print("\nFull fine-tune for holdout ...")
model_h = finetune_joint(list(flat_tr_imgs), X_morph_tr_n, y_tr)
y_pred  = predict_joint(model_h, list(flat_te_imgs), X_morph_te_n)

save_results(RES, CFG, np.array(cv_scores), y_te, y_pred, le, baseline)
print(f"Wall time: {time.perf_counter()-t0:.1f}s")
