"""
MorphNN Streamlit App — Viral Exanthem Classification with Explainability
Joint Morphology + CNN model (C21). Explains predictions via GradCAM, LIME, SHAP.
Run: streamlit run app.py
"""
import sys, pickle
from pathlib import Path
import numpy as np
import cv2
import torch
import torch.nn as nn
import torchvision.transforms as T
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

ROOT      = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "saved_model"
EXP_DIR   = ROOT / "explainability"
CACHE_DIR = ROOT / "morphnn_cache"
sys.path.insert(0, str(ROOT / "configs"))
sys.path.insert(0, str(EXP_DIR))

from explain import (compute_gradcam, gradcam_overlay,
                     compute_shap, compute_lime, lime_overlay,
                     TF_EVAL, CLASSES, MORPH_FEATURE_NAMES)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MorphNN — Viral Exanthem AI",
    page_icon="🔬",
    layout="wide",
)

# ── Model definition (must match saved_model/save_c21_model.py) ────────────────
N_CLASSES = len(CLASSES)

class JointMorphCNN(nn.Module):
    def __init__(self):
        super().__init__()
        import torchvision.models as models
        base = models.mobilenet_v3_large(weights=None)
        self.features = base.features
        self.avgpool  = base.avgpool
        self.morph_fc = nn.Sequential(nn.Linear(16, 64), nn.Hardswish(), nn.Dropout(0.3))
        self.head = nn.Sequential(
            nn.Linear(960 + 64, 512), nn.Hardswish(), nn.Dropout(0.2),
            nn.Linear(512, N_CLASSES)
        )

    def forward(self, img, morph):
        x = self.avgpool(self.features(img)).flatten(1)
        m = self.morph_fc(morph)
        return self.head(torch.cat([x, m], dim=1))


@st.cache_resource(show_spinner="Loading MorphNN model...")
def load_model():
    model = JointMorphCNN().to(device)
    model.load_state_dict(torch.load(MODEL_DIR / "c21_joint_morphcnn.pth",
                                     map_location=device))
    model.eval()
    return model

@st.cache_resource(show_spinner="Loading support files...")
def load_support():
    with open(MODEL_DIR / "morph_scaler.pkl", "rb") as f:
        sc = pickle.load(f)
    with open(MODEL_DIR / "label_encoder.pkl", "rb") as f:
        le = pickle.load(f)
    X_morph_tr = np.load(CACHE_DIR / "morph_train_X.npy").astype(np.float32)
    bg_morphs  = sc.transform(X_morph_tr)
    return sc, le, bg_morphs

@st.cache_resource(show_spinner="Loading morph extractor...")
def load_morph_extractor():
    sys.path.insert(0, str(ROOT))
    from MorphNN_6class import MorphFeatureExtractor
    return MorphFeatureExtractor()


def predict(model, img_bgr, morph_norm):
    img_t   = TF_EVAL(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)).unsqueeze(0).to(device)
    morph_t = torch.tensor(morph_norm, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(img_t, morph_t)
        probs  = logits.softmax(dim=1).cpu().numpy()[0]
    return probs


def prob_bar_chart(probs, predicted_class):
    fig, ax = plt.subplots(figsize=(6, 3.2))
    colors  = ["#dc2626" if i == predicted_class else "#93c5fd" for i in range(N_CLASSES)]
    bars    = ax.barh(CLASSES, probs, color=colors, edgecolor="white", height=0.6)
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("Probability", fontsize=10)
    ax.set_title("Class Probabilities", fontsize=11)
    for bar, p in zip(bars, probs):
        ax.text(p + 0.01, bar.get_y() + bar.get_height()/2,
                f"{p:.3f}", va="center", fontsize=9)
    ax.axvline(0.5, color="#64748b", linestyle="--", linewidth=0.8, alpha=0.6)
    plt.tight_layout()
    return fig


def morph_radar_chart(morph_raw):
    fig, ax = plt.subplots(figsize=(5, 4.5))
    normed  = (morph_raw - morph_raw.min()) / (morph_raw.max() - morph_raw.min() + 1e-8)
    y       = normed
    x       = np.arange(len(MORPH_FEATURE_NAMES))
    ax.barh(x, y, color="#3b82f6", edgecolor="white", height=0.65)
    ax.set_yticks(x)
    ax.set_yticklabels(MORPH_FEATURE_NAMES, fontsize=8)
    ax.set_xlim(0, 1.1)
    ax.set_xlabel("Normalised value", fontsize=9)
    ax.set_title("16 Morphological Features", fontsize=10)
    plt.tight_layout()
    return fig


def shap_bar_chart(shap_vals, feature_names, class_name):
    fig, ax = plt.subplots(figsize=(6, 4.5))
    idx     = np.argsort(np.abs(shap_vals))
    sv      = shap_vals[idx]
    fn      = [feature_names[i] for i in idx]
    colors  = ["#dc2626" if v > 0 else "#3b82f6" for v in sv]
    ax.barh(fn, sv, color=colors, edgecolor="white", height=0.65)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("SHAP value (impact on prediction)", fontsize=9)
    ax.set_title(f"SHAP — Morph Feature Impact\nPredicted: {class_name}", fontsize=10)
    ax.tick_params(axis="y", labelsize=8)
    plt.tight_layout()
    return fig


# ── UI ─────────────────────────────────────────────────────────────────────────
st.title("🔬 MorphNN — Viral Exanthem Classifier")
st.markdown(
    "**Joint Morphology + CNN model** (C21) · Macro F1 = **0.9977** · "
    "Explains predictions with GradCAM, LIME, and SHAP."
)

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    lime_samples  = st.slider("LIME samples", 200, 1000, 400, 100,
                               help="More = better LIME but slower")
    shap_bg       = st.slider("SHAP background size", 10, 100, 30, 10,
                               help="More = better SHAP but slower")
    gradcam_alpha = st.slider("GradCAM overlay α", 0.2, 0.8, 0.45, 0.05)
    run_lime  = st.checkbox("Run LIME", value=True)
    run_shap  = st.checkbox("Run SHAP (slow ~30s)", value=True)

    st.markdown("---")
    st.markdown("**Classes**")
    for c in CLASSES:
        st.markdown(f"· {c}")

uploaded = st.file_uploader("Upload a skin lesion image", type=["jpg","jpeg","png","bmp"])

if uploaded is not None:
    # Load resources
    model       = load_model()
    sc, le, bg_morphs = load_support()
    morph_ext   = load_morph_extractor()

    # Decode image
    file_bytes = np.frombuffer(uploaded.read(), np.uint8)
    img_bgr    = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    img_bgr    = cv2.resize(img_bgr, (224, 224))
    img_rgb    = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # Extract morphological features
    with st.spinner("Extracting morphological features..."):
        morph_raw  = morph_ext.extract(img_bgr).astype(np.float32)
        morph_norm = sc.transform(morph_raw.reshape(1, -1))[0].astype(np.float32)

    # Predict
    probs         = predict(model, img_bgr, morph_norm)
    pred_idx      = int(probs.argmax())
    pred_class    = CLASSES[pred_idx]
    confidence    = float(probs[pred_idx])

    # ── Result header ──────────────────────────────────────────────────────────
    col_img, col_pred = st.columns([1, 1])
    with col_img:
        st.subheader("Input Image")
        st.image(img_rgb, use_container_width=True)
        if confidence > 0.9:
            st.success(f"**Predicted: {pred_class}** ({confidence*100:.1f}%)")
        elif confidence > 0.7:
            st.warning(f"**Predicted: {pred_class}** ({confidence*100:.1f}%)")
        else:
            st.error(f"**Predicted: {pred_class}** ({confidence*100:.1f}%) — low confidence")

    with col_pred:
        st.subheader("Class Probabilities")
        st.pyplot(prob_bar_chart(probs, pred_idx), use_container_width=True)

    # ── Explainability tabs ────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("🧠 Explainability")
    tab_cam, tab_lime, tab_shap, tab_morph = st.tabs(
        ["🌡️ GradCAM", "🟩 LIME", "📊 SHAP", "🔎 Morph Features"]
    )

    # GradCAM
    with tab_cam:
        st.markdown(
            "**GradCAM** highlights image regions that most influenced the prediction. "
            "Red = high importance, blue = low importance."
        )
        with st.spinner("Computing GradCAM..."):
            cam      = compute_gradcam(model, img_bgr, morph_norm, class_idx=pred_idx)
            overlay  = gradcam_overlay(img_bgr, cam, alpha=gradcam_alpha)
            overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

        c1, c2 = st.columns(2)
        with c1:
            st.image(img_rgb, caption="Original", use_container_width=True)
        with c2:
            st.image(overlay_rgb, caption=f"GradCAM — {pred_class}", use_container_width=True)

        # Heatmap colorbar
        fig_cb, ax_cb = plt.subplots(figsize=(5, 0.4))
        cb = plt.colorbar(
            plt.cm.ScalarMappable(norm=mcolors.Normalize(0,1), cmap="jet"),
            cax=ax_cb, orientation="horizontal"
        )
        cb.set_label("Activation intensity", fontsize=9)
        st.pyplot(fig_cb, use_container_width=True)
        plt.close(fig_cb)

    # LIME
    with tab_lime:
        st.markdown(
            "**LIME** perturbs the image to find which superpixel regions are "
            "**most important** (green boundary = supports prediction)."
        )
        if run_lime:
            with st.spinner(f"Running LIME ({lime_samples} samples)..."):
                explanation, img_rgb_lime, _ = compute_lime(
                    model, img_bgr, morph_norm,
                    class_idx=pred_idx, num_samples=lime_samples
                )
                lime_img = lime_overlay(explanation, img_rgb_lime, pred_idx,
                                        positive_only=True, num_features=8)
            c1, c2 = st.columns(2)
            with c1:
                st.image(img_rgb, caption="Original", use_container_width=True)
            with c2:
                st.image(lime_img, caption=f"LIME — {pred_class}", use_container_width=True)
        else:
            st.info("LIME disabled in sidebar settings.")

    # SHAP
    with tab_shap:
        st.markdown(
            "**SHAP** quantifies how much each of the 16 morphological features "
            "**pushed** the prediction toward or away from the predicted class. "
            "🔴 Red = increases probability · 🔵 Blue = decreases probability."
        )
        if run_shap:
            with st.spinner(f"Running SHAP (background={shap_bg})..."):
                shap_vals = compute_shap(
                    model, morph_norm, img_bgr,
                    background_morphs=bg_morphs,
                    class_idx=pred_idx, n_bg=shap_bg
                )
            st.pyplot(shap_bar_chart(shap_vals, MORPH_FEATURE_NAMES, pred_class),
                      use_container_width=True)
            # Top 3 features
            top3 = np.argsort(np.abs(shap_vals))[::-1][:3]
            st.markdown("**Top 3 most influential morphological features:**")
            for rank, i in enumerate(top3, 1):
                direction = "increases" if shap_vals[i] > 0 else "decreases"
                st.markdown(f"{rank}. **{MORPH_FEATURE_NAMES[i]}** (SHAP={shap_vals[i]:+.4f}) — {direction} probability of {pred_class}")
        else:
            st.info("SHAP disabled in sidebar settings.")

    # Morph features table
    with tab_morph:
        st.markdown(
            "**Raw morphological features** extracted from the uploaded image "
            "using the 16-descriptor pipeline (lesion detection, shape, color, texture)."
        )
        c1, c2 = st.columns([1, 1])
        with c1:
            import pandas as pd
            df_morph = pd.DataFrame({
                "Feature": MORPH_FEATURE_NAMES,
                "Value":   [f"{v:.4f}" for v in morph_raw],
            })
            st.dataframe(df_morph, use_container_width=True, height=440)
        with c2:
            st.pyplot(morph_radar_chart(morph_raw), use_container_width=True)

else:
    # Welcome screen
    st.info(
        "👆 Upload a skin lesion image to classify it and see explainability maps.\n\n"
        "**Supported conditions:** Chickenpox · Cowpox · HFMD · Healthy · Measles · Monkeypox"
    )
    st.markdown("""
    ### How it works
    | Step | Method | Purpose |
    |------|--------|---------|
    | 1️⃣ | Morphological feature extraction | 16 lesion descriptors from OpenCV |
    | 2️⃣ | Joint CNN + Morph inference | MobileNetV3 + morph branch, end-to-end |
    | 3️⃣ | GradCAM | Which image regions drove the decision |
    | 4️⃣ | LIME | Which superpixels support the prediction |
    | 5️⃣ | SHAP | Which morphological features mattered most |

    **Model performance:** Macro F1 = **0.9977** (1,134 test images) — beats standalone CNN (0.9946)
    """)
