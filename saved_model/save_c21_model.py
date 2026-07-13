"""Train C21 on full training set and save weights for the Streamlit app."""
import sys, pickle
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[1] / "configs"))
from shared_utils import *
from shared_utils import _ce_weights
import torch.nn as nn

SAVE_DIR = Path(__file__).resolve().parent
EPOCHS   = 20

class JointMorphCNN(nn.Module):
    def __init__(self):
        super().__init__()
        base = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT)
        self.features = base.features
        self.avgpool  = base.avgpool
        self.morph_fc = nn.Sequential(
            nn.Linear(16, 64), nn.Hardswish(), nn.Dropout(0.3)
        )
        self.head = nn.Sequential(
            nn.Linear(960 + 64, 512), nn.Hardswish(), nn.Dropout(0.2),
            nn.Linear(512, N_CLASSES)
        )

    def forward(self, img, morph):
        x = self.avgpool(self.features(img)).flatten(1)
        m = self.morph_fc(morph)
        return self.head(torch.cat([x, m], dim=1))

class _JointDs(Dataset):
    def __init__(self, imgs_bgr, morphs, labels, tf):
        self.imgs   = imgs_bgr
        self.morphs = torch.tensor(morphs, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.tf     = tf
    def __len__(self): return len(self.imgs)
    def __getitem__(self, i):
        return self.tf(cv2.cvtColor(self.imgs[i], cv2.COLOR_BGR2RGB)), self.morphs[i], self.labels[i]

print("Loading data ...")
X_morph_tr, y_tr = load_morph("train")
le = load_le()

sc_morph = StandardScaler()
X_morph_tr_n = sc_morph.fit_transform(X_morph_tr).astype(np.float32)

flat_tr_imgs, _ = load_augmented_flat(target=2000)
flat_tr_imgs = np.array(flat_tr_imgs, dtype=np.uint8)

print(f"Training JointMorphCNN on full set ({len(flat_tr_imgs)} images, {EPOCHS} epochs) ...")
t0 = time.perf_counter()

model = JointMorphCNN().to(device)
ds = _JointDs(list(flat_tr_imgs), X_morph_tr_n, y_tr, TF_TRAIN)
dl = DataLoader(ds, batch_size=CNN_BATCH, shuffle=True,
                num_workers=4, pin_memory=True, persistent_workers=True)
crit = nn.CrossEntropyLoss(weight=_ce_weights(y_tr), label_smoothing=0.05)
opt  = optim.AdamW([
    {'params': list(model.features.parameters()) + list(model.avgpool.parameters()), 'lr': 1e-4},
    {'params': list(model.morph_fc.parameters()) + list(model.head.parameters()), 'lr': 1e-3},
], weight_decay=1e-4)
sched = optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=10, T_mult=1)

for ep in range(EPOCHS):
    model.train()
    total_loss = 0.
    for img, morph, lbl in dl:
        img, morph, lbl = img.to(device), morph.to(device), lbl.to(device)
        opt.zero_grad()
        loss = crit(model(img, morph), lbl)
        loss.backward(); opt.step()
        total_loss += loss.item()
    sched.step()
    print(f"  Epoch {ep+1:2d}/{EPOCHS}  loss={total_loss/len(dl):.4f}")

print(f"\nTraining done in {time.perf_counter()-t0:.1f}s")

# Save model weights
torch.save(model.state_dict(), SAVE_DIR / "c21_joint_morphcnn.pth")
print(f"  Model → {SAVE_DIR}/c21_joint_morphcnn.pth")

# Save morph scaler
with open(SAVE_DIR / "morph_scaler.pkl", "wb") as f:
    pickle.dump(sc_morph, f)
print(f"  Scaler → {SAVE_DIR}/morph_scaler.pkl")

# Copy label encoder
import shutil
shutil.copy(ROOT / "morphnn_cache" / "label_encoder.pkl", SAVE_DIR / "label_encoder.pkl")
print(f"  LabelEncoder → {SAVE_DIR}/label_encoder.pkl")
print("\nAll saved. Ready for the Streamlit app.")
