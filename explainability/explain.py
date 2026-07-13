"""
Explainability module for C21 JointMorphCNN.
Provides GradCAM, SHAP (morph features), and LIME (image regions).
"""
import numpy as np
import cv2
import torch
import torch.nn as nn
import torchvision.transforms as T
from pathlib import Path

# ── Constants ──────────────────────────────────────────────────────────────────
CLASSES = ["Chickenpox", "Cowpox", "HFMD", "Healthy", "Measles", "Monkeypox"]
MORPH_FEATURE_NAMES = [
    "lesion_count",        "avg_lesion_area",    "area_heterogeneity",  "avg_circularity",
    "sparsity_score",      "confluence_density",  "localized_hue",       "localized_saturation",
    "avg_aspect_ratio",    "avg_solidity",        "localized_value",     "hue_std",
    "saturation_std",      "spatial_entropy",     "max_lesion_area_ratio","background_saturation",
]

TF_EVAL = T.Compose([
    T.ToPILImage(), T.Resize(224), T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── GradCAM ────────────────────────────────────────────────────────────────────
class _GradCAMHook:
    """Vanilla GradCAM for the last conv block of the joint model."""
    def __init__(self, layer):
        self.activations = None
        self.gradients   = None
        self._fh = layer.register_forward_hook(self._save_act)
        self._bh = layer.register_full_backward_hook(self._save_grad)

    def _save_act(self, _, __, output): self.activations = output.detach()
    def _save_grad(self, _, __, grad_out): self.gradients = grad_out[0].detach()
    def remove(self): self._fh.remove(); self._bh.remove()


def compute_gradcam(model, img_bgr: np.ndarray, morph_norm: np.ndarray,
                    class_idx: int = None) -> np.ndarray:
    """
    Returns a (H, W) float heatmap in [0,1] for the GradCAM of img_bgr.
    class_idx=None uses the argmax prediction.
    """
    model.eval()
    target_layer = model.features[-1]
    hook = _GradCAMHook(target_layer)

    img_t   = TF_EVAL(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)).unsqueeze(0).to(device)
    morph_t = torch.tensor(morph_norm, dtype=torch.float32).unsqueeze(0).to(device)

    img_t.requires_grad_(False)
    logits = model(img_t, morph_t)

    if class_idx is None:
        class_idx = int(logits.argmax(dim=1).item())

    model.zero_grad()
    logits[0, class_idx].backward()

    acts  = hook.activations[0]             # (C, h, w)
    grads = hook.gradients[0]               # (C, h, w)
    hook.remove()

    weights = grads.mean(dim=(1, 2))        # (C,)
    cam     = (weights[:, None, None] * acts).sum(0)  # (h, w)
    cam     = torch.relu(cam).cpu().numpy()

    H, W    = img_bgr.shape[:2]
    cam     = cv2.resize(cam, (W, H))
    if cam.max() > 0:
        cam = (cam - cam.min()) / (cam.max() - cam.min())
    return cam.astype(np.float32)


def gradcam_overlay(img_bgr: np.ndarray, cam: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Blend GradCAM heatmap over the original image. Returns BGR uint8."""
    heatmap = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET)
    return cv2.addWeighted(img_bgr, 1 - alpha, heatmap, alpha, 0)


# ── SHAP (morphological features) ──────────────────────────────────────────────
def compute_shap(model, morph_norm: np.ndarray, img_bgr: np.ndarray,
                 background_morphs: np.ndarray, class_idx: int = None,
                 n_bg: int = 50) -> np.ndarray:
    """
    Returns SHAP values (16,) for the morph features, explaining the class_idx score.
    Uses KernelExplainer with a scalar output (prob of predicted class) for robustness.
    """
    import shap

    img_t = TF_EVAL(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)).unsqueeze(0).to(device)

    # Determine predicted class first
    if class_idx is None:
        morph_t = torch.tensor(morph_norm, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            class_idx = int(model(img_t, morph_t).argmax(dim=1).item())

    # Scalar output: probability of the predicted class only (avoids multi-output indexing issues)
    def morph_predict_scalar(morphs_np):
        results = []
        for row in morphs_np:
            m_t = torch.tensor(row, dtype=torch.float32).unsqueeze(0).to(device)
            with torch.no_grad():
                prob = model(img_t, m_t).softmax(dim=1).cpu().numpy()[0, class_idx]
            results.append(float(prob))
        return np.array(results)

    rng = np.random.default_rng(42)
    bg  = background_morphs[rng.choice(len(background_morphs), size=min(n_bg, len(background_morphs)), replace=False)]

    explainer   = shap.KernelExplainer(morph_predict_scalar, bg)
    shap_values = explainer.shap_values(morph_norm.reshape(1, -1), nsamples=200, silent=True)

    # shap_values is (1, 16) for scalar output — return the (16,) vector
    sv = np.array(shap_values)
    return sv.reshape(-1)[-16:]  # always return exactly 16 values


# ── LIME (image regions) ────────────────────────────────────────────────────────
def compute_lime(model, img_bgr: np.ndarray, morph_norm: np.ndarray,
                 class_idx: int = None, num_samples: int = 500):
    """
    Returns (explanation, img_rgb, class_idx).
    explanation is a lime_image Explanation object.
    """
    from lime import lime_image

    morph_t = torch.tensor(morph_norm, dtype=torch.float32).unsqueeze(0).to(device)
    model.eval()

    if class_idx is None:
        img_t = TF_EVAL(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)).unsqueeze(0).to(device)
        with torch.no_grad():
            class_idx = int(model(img_t, morph_t).argmax(dim=1).item())

    def predict_fn(images_rgb_float):
        # LIME passes RGB float in [0,1] — TF_EVAL expects RGB uint8
        preds = []
        for img_arr in images_rgb_float:
            img_u8 = (img_arr * 255).astype(np.uint8)
            t = TF_EVAL(img_u8).unsqueeze(0).to(device)
            with torch.no_grad():
                prob = model(t, morph_t).softmax(dim=1).cpu().numpy()[0]
            preds.append(prob)
        return np.array(preds)

    img_rgb    = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    explainer  = lime_image.LimeImageExplainer(random_state=42)
    explanation = explainer.explain_instance(
        img_rgb.astype(np.float32) / 255.0, predict_fn,
        top_labels=N_CLASSES, hide_color=0, num_samples=num_samples,
        random_seed=42
    )
    return explanation, img_rgb, class_idx


def lime_overlay(explanation, img_rgb: np.ndarray, class_idx: int,
                 positive_only: bool = True, num_features: int = 10) -> np.ndarray:
    """Returns RGB uint8 image with LIME boundary overlay."""
    from skimage.segmentation import mark_boundaries
    temp, mask = explanation.get_image_and_mask(
        class_idx, positive_only=positive_only,
        num_features=num_features, hide_rest=False
    )
    img_lime = mark_boundaries((temp * 255).astype(np.uint8), mask)
    return (img_lime * 255).astype(np.uint8)


N_CLASSES = len(CLASSES)
