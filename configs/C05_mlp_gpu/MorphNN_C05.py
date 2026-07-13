"""C05 — 3-layer MLP (GPU) on 16 morph + 128 CNN-PCA = 144-dim hybrid.
Neural classifier replaces RF. Dropout + BatchNorm. 50 epochs.
Can learn non-linear feature interactions that RF misses.
"""
import sys; sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[1]))
from shared_utils import *
from torch.utils.data import TensorDataset
import torch.nn.functional as F

CFG = "C05_mlp_gpu"
RES = Path(__file__).parent / "results"
N_PCA    = 128
MLP_IN   = 16 + N_PCA   # 144
MLP_EPOCHS = 60
MLP_BATCH  = 256

class MLP(nn.Module):
    def __init__(self, in_dim=MLP_IN, n_cls=N_CLASSES, dropout=0.35):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(512, 256),   nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, 128),   nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(dropout/2),
            nn.Linear(128, n_cls),
        )
    def forward(self, x): return self.net(x)

def train_mlp(Xtr, ytr, Xvl, yvl, epochs=MLP_EPOCHS):
    Xtr_t = torch.tensor(Xtr, dtype=torch.float32)
    ytr_t = torch.tensor(ytr, dtype=torch.long)
    Xvl_t = torch.tensor(Xvl, dtype=torch.float32)
    yvl_t = torch.tensor(yvl, dtype=torch.long)
    counts = np.array([max(int((ytr==c).sum()),1) for c in range(N_CLASSES)], np.float32)
    w = torch.tensor(len(ytr)/(N_CLASSES*counts)).to(device)
    dl = DataLoader(TensorDataset(Xtr_t, ytr_t), batch_size=MLP_BATCH, shuffle=True)
    model = MLP().to(device)
    opt   = optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-4)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit  = nn.CrossEntropyLoss(weight=w)
    best_f1, best_state = -1., None
    for ep in range(epochs):
        model.train()
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(); crit(model(xb), yb).backward(); opt.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            preds = model(Xvl_t.to(device)).argmax(1).cpu().numpy()
        f1 = f1_score(yvl, preds, average="macro")
        if f1 > best_f1:
            best_f1 = f1; best_state = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    return model

print(f"\n{'='*55}\n{CFG}  |  16 morph + {N_PCA} CNN-PCA + MLP (GPU)\n{'='*55}")
t0 = time.perf_counter()

X_morph_tr, y_tr = load_morph("train")
X_morph_te, y_te = load_morph("test")
X_cnn_tr,   _    = load_cnn("MobileNetV2", "train")
X_cnn_te,   _    = load_cnn("MobileNetV2", "test")
le               = load_le()
baseline         = load_rf_baseline()

cv_scores = []
for rep in range(CV_REPEATS):
    skf = StratifiedKFold(n_splits=CV_SPLITS, shuffle=True, random_state=SEED+rep)
    for tr_i, vl_i in skf.split(X_morph_tr, y_tr):
        Xm_tr, Xm_vl = X_morph_tr[tr_i], X_morph_tr[vl_i]
        Xc_tr, Xc_vl = X_cnn_tr[tr_i],   X_cnn_tr[vl_i]
        ytr,   yvl   = y_tr[tr_i],         y_tr[vl_i]
        pca = PCA(n_components=N_PCA, random_state=SEED)
        sc  = StandardScaler()
        Xf_tr = sc.fit_transform(np.hstack([Xm_tr, pca.fit_transform(Xc_tr)]))
        Xf_vl = sc.transform(       np.hstack([Xm_vl, pca.transform(Xc_vl)]))
        mlp   = train_mlp(Xf_tr, ytr, Xf_vl, yvl, epochs=MLP_EPOCHS)
        mlp.eval()
        with torch.no_grad():
            preds = mlp(torch.tensor(Xf_vl, dtype=torch.float32).to(device)).argmax(1).cpu().numpy()
        cv_scores.append(f1_score(yvl, preds, average="macro"))
    print(f"  Rep {rep+1}/{CV_REPEATS}  current mean: {np.mean(cv_scores):.4f}")

# Holdout — use a stratified 10% split for early-stopping validation
from sklearn.model_selection import train_test_split
pca_h = PCA(n_components=N_PCA, random_state=SEED); sc_h = StandardScaler()
Xf_tr = sc_h.fit_transform(np.hstack([X_morph_tr, pca_h.fit_transform(X_cnn_tr)]))
Xf_te = sc_h.transform(       np.hstack([X_morph_te, pca_h.transform(X_cnn_te)]))
Xf_tr2, Xf_val, ytr2, yval = train_test_split(
    Xf_tr, y_tr, test_size=0.1, stratify=y_tr, random_state=SEED)
mlp_h = train_mlp(Xf_tr2, ytr2, Xf_val, yval, epochs=MLP_EPOCHS)
mlp_h.eval()
with torch.no_grad():
    y_pred = mlp_h(torch.tensor(Xf_te, dtype=torch.float32).to(device)).argmax(1).cpu().numpy()

save_results(RES, CFG, np.array(cv_scores), y_te, y_pred, le, baseline)
print(f"Wall time: {time.perf_counter()-t0:.1f}s")
