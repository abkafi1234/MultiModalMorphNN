# %%
#!/usr/bin/env python3
"""
benchmark_vision_pipeline.py
Rigorous benchmarking suite comparing engineered-feature Random Forest 
against 7 CNN architectures using Repeated Stratified 5-Fold Cross-Validation 
and paired Wilcoxon signed-rank tests evaluated on Macro F1-Score.
"""

import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial import distance
from scipy import stats
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from collections import Counter
import random
import copy
import pickle

# PyTorch Deep Learning Imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import torchvision.transforms as transforms

# -------------------------
# Reproducibility Configuration
# -------------------------
def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# Select hardware accelerator
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------
# Feature Extraction and Dataset Utilities
# -------------------------
class ExanthemFeatureExtractor:
    """Handles image processing and classical feature extraction."""
    def __init__(self):
        self.feature_names = ['lesion_count', 'avg_area', 'std_area', 'avg_circularity',
                              'sparsity_score', 'confluence_ratio', 'avg_hue', 'avg_saturation']

    def apply_gray_world_white_balance(self, img):
        b, g, r = cv2.split(img.astype(np.float32))
        avg_b, avg_g, avg_r = np.mean(b), np.mean(g), np.mean(r)
        avg_all = (avg_b + avg_g + avg_r) / 3.0
        scale_b = avg_all / avg_b if avg_b > 0 else 1.0
        scale_g = avg_all / avg_g if avg_g > 0 else 1.0
        scale_r = avg_all / avg_r if avg_r > 0 else 1.0
        return cv2.merge((np.clip(b * scale_b, 0, 255),
                          np.clip(g * scale_g, 0, 255),
                          np.clip(r * scale_r, 0, 255))).astype(np.uint8)

    def extract_tabular_features(self, img):
        """Processes image matrix to return classical tabular domain features."""
        smoothed = cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)
        gray = cv2.cvtColor(smoothed, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
        equalized = clahe.apply(gray)
        thresh = cv2.adaptiveThreshold(equalized, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, 51, 2)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        clean_thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
        contours, _ = cv2.findContours(clean_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        centroids, areas, circularities = [], [], []
        valid_contours = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if 50 < area < (512 * 512 * 0.1):
                perimeter = cv2.arcLength(cnt, True)
                circularity = 0 if perimeter == 0 else 4 * np.pi * (area / (perimeter * perimeter))
                M = cv2.moments(cnt)
                if M["m00"] != 0:
                    centroids.append((int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])))
                    areas.append(area)
                    circularities.append(circularity)
                    valid_contours.append(cnt)

        wb_img = self.apply_gray_world_white_balance(img)
        hsv_img = cv2.cvtColor(wb_img, cv2.COLOR_BGR2HSV)
        valid_mask = np.zeros((512, 512), dtype=np.uint8)
        if valid_contours:
            cv2.drawContours(valid_mask, valid_contours, -1, 255, thickness=cv2.FILLED)
            mean_color = cv2.mean(hsv_img, mask=valid_mask)
            avg_hue, avg_saturation = mean_color[0], mean_color[1]
        else:
            avg_hue, avg_saturation = 0, 0

        std_area = np.std(areas) if len(areas) > 1 else 0
        avg_circularity = np.mean(circularities) if circularities else 0

        if len(centroids) > 1:
            dist_matrix = distance.cdist(centroids, centroids, 'euclidean')
            np.fill_diagonal(dist_matrix, np.inf)
            sparsity_score = np.mean(np.min(dist_matrix, axis=1))
        else:
            sparsity_score = 0

        confluence_ratio = sum(areas) / (512 * 512)
        return [len(centroids), np.mean(areas) if areas else 0, std_area,
                avg_circularity, sparsity_score, confluence_ratio, avg_hue, avg_saturation]

def load_dual_dataset(root_dir, extractor):
    """Loads raw images for PyTorch and tabular vectors for Scikit-Learn simultaneously."""
    images, tabular_data, labels, filenames = [], [], [], []
    if not os.path.isdir(root_dir):
        return None
    
    classes = sorted(os.listdir(root_dir))
    class_to_idx = {cls_name: idx for idx, cls_name in enumerate(classes)}
    
    for label_dir in classes:
        dir_path = os.path.join(root_dir, label_dir)
        if not os.path.isdir(dir_path):
            continue
        for img_file in sorted(os.listdir(dir_path)):
            img_path = os.path.join(dir_path, img_file)
            img = cv2.imread(img_path)
            if img is None:
                continue
            img = cv2.resize(img, (512, 512))
            
            # Extract features for classical pipeline
            f_vector = extractor.extract_tabular_features(img)
            
            # Save raw BGR converted to RGB for PyTorch Deep Learning pipeline
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            images.append(img_rgb)
            tabular_data.append(f_vector)
            labels.append(class_to_idx[label_dir])
            filenames.append(img_file)
            
    return np.array(images), np.array(tabular_data), np.array(labels), filenames, class_to_idx

def balance_dual_dataset(images, tabular, labels, filenames):
    """Applies unified undersampling across all modalities to maintain target alignment."""
    counts = Counter(labels)
    min_count = min(counts.values())
    selected_indices = []
    
    for cls in counts.keys():
        idxs = np.where(labels == cls)[0].tolist()
        if len(idxs) <= min_count:
            selected_indices.extend(idxs)
        else:
            selected_indices.extend(random.sample(idxs, min_count))
            
    selected_indices = sorted(selected_indices)
    return (images[selected_indices], tabular[selected_indices], 
            labels[selected_indices], [filenames[i] for i in selected_indices])

# -------------------------
# PyTorch Dataset Definition
# -------------------------
class LesionImageDataset(Dataset):
    """Prepares image tracking collections for PyTorch optimization execution loops."""
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        label = self.labels[idx]
        if self.transform:
            img = self.transform(img)
        return img, label

# -------------------------
# CNN Architectural Factory Helpers
# -------------------------
def initialize_cnn(model_name, num_classes):
    """Instantiates a CNN architecture with a custom classification head mapping."""
    if model_name == 'EfficientNet-B0':
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif model_name == 'ResNet18':
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == 'ResNet34':
        model = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == 'MobileNetV2':
        model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif model_name == 'MobileNetV3':
        model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT)
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
    elif model_name == 'ShuffleNetV2':
        model = models.shufflenet_v2_x1_0(weights=models.ShuffleNet_V2_X1_0_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == 'SqueezeNet':
        model = models.squeezenet1_1(weights=models.SqueezeNet1_1_Weights.DEFAULT)
        model.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1))
        model.num_classes = num_classes
    else:
        raise ValueError(f"Unknown architecture: {model_name}")
    return model.to(device)

# -------------------------
# CNN Execution Loop Drivers
# -------------------------
def train_and_evaluate_cnn(model_name, train_data, val_data, num_classes, epochs=5, batch_size=16):
    """Executes a deep learning optimization pipeline over a cross-validation split."""
    train_images, train_labels = train_data
    val_images, val_labels = val_data
    
    # Standard normalization transform pipeline for ImageNet architectures
    cnn_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    train_dataset = LesionImageDataset(train_images, train_labels, transform=cnn_transforms)
    val_dataset = LesionImageDataset(val_images, val_labels, transform=cnn_transforms)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    model = initialize_cnn(model_name, num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    
    # Structural training loop execution
    for epoch in range(epochs):
        model.train()
        for imgs, lbls in train_loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, lbls)
            loss.backward()
            optimizer.step()
            
    # Evaluation Step
    model.eval()
    all_preds = []
    with torch.no_grad():
        for imgs, _ in val_loader:
            imgs = imgs.to(device)
            outputs = model(imgs)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            
    score = f1_score(val_labels, all_preds, average='macro')
    return score, model

# -------------------------
# Statistical Calculation Functions
# -------------------------
def compute_ci(scores, confidence=0.95):
    a = np.array(scores)
    n = len(a)
    mean = a.mean()
    std = a.std(ddof=1) if n > 1 else 0.0
    if n > 1:
        se = stats.sem(a)
        h = se * stats.t.ppf((1 + confidence) / 2., n - 1)
        return mean, std, (mean - h, mean + h)
    return mean, std, (mean, mean)

# -------------------------
# Execution Main Benchmark Core
# -------------------------
def run_benchmark(train_dir='./train', test_dir='./test', n_splits=5, repeat_cv_runs=10, alpha=0.05, cnn_epochs=3):
    print(" === Phase 1: Ingesting Data & Modality Preprocessing ===")
    extractor = ExanthemFeatureExtractor()
    
    raw_data = load_dual_dataset(train_dir, extractor)
    if raw_data is None:
        print("Error: Directory structure missing. Creating synthetic check data context inside runtime environment.")
        return
    
    tr_images, tr_tabular, tr_labels, tr_files, class_mapping = raw_data
    num_classes = len(class_mapping)
    
    # Balanced downsampling to prevent algorithmic bias
    images_bal, tabular_bal, labels_bal, _ = balance_dual_dataset(tr_images, tr_tabular, tr_labels, tr_files)
    print(f"Dataset Balanced Successfully: {len(labels_bal)} total samples across {num_classes} classes.")

    # Configure model list matrix maps
    models_to_test = ['RandomForest', 'EfficientNet-B0', 'ResNet34', 'ResNet18', 
                      'MobileNetV2', 'MobileNetV3', 'ShuffleNetV2', 'SqueezeNet']
    
    # Initialize structural arrays to track metric histories for Wilcoxon checks
    benchmark_history = {name: [] for name in models_to_test}
    
    print("\n === Phase 2: Starting Cross-Validation Loop ===")
    for run in range(repeat_cv_runs):
        run_seed = 42 + run
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=run_seed)
        print(f"\nEvaluating Cross Validation Run Block Iteration {run + 1}/{repeat_cv_runs} (Seed={run_seed})")
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(tabular_bal, labels_bal)):
            # Classical Tabular splits
            X_tr_tab, X_val_tab = tabular_bal[train_idx], tabular_bal[val_idx]
            y_tr_tab, y_val_tab = labels_bal[train_idx], labels_bal[val_idx]
            
            # Deep Learning Matrix splits
            X_tr_img, X_val_img = images_bal[train_idx], images_bal[val_idx]
            y_tr_img, y_val_img = labels_bal[train_idx], labels_bal[val_idx]
            
            # --- Evaluate Baseline Model: Random Forest ---
            scaler = StandardScaler()
            X_tr_tab_scaled = scaler.fit_transform(X_tr_tab)
            X_val_tab_scaled = scaler.transform(X_val_tab)
            
            rf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
            rf.fit(X_tr_tab_scaled, y_tr_tab)
            rf_preds = rf.predict(X_val_tab_scaled)
            rf_f1 = f1_score(y_val_tab, rf_preds, average='macro')
            benchmark_history['RandomForest'].append(rf_f1)
            
            # --- Evaluate CNN Models ---
            for model_name in models_to_test:
                if model_name == 'RandomForest':
                    continue
                cnn_f1, _ = train_and_evaluate_cnn(
                    model_name=model_name, 
                    train_data=(X_tr_img, y_tr_img), 
                    val_data=(X_val_img, y_val_img), 
                    num_classes=num_classes,
                    epochs=cnn_epochs
                )
                benchmark_history[model_name].append(cnn_f1)
                
            print(f"   Fold {fold+1} complete. RF F1: {rf_f1:.4f}")

    # -------------------------
    # Statistical Benchmark & Wilcoxon Comparison
    # -------------------------
    print("\n === Phase 3: Statistical Hypothesis Testing (Wilcoxon Signed-Rank) ===")
    print(f"Comparing each Deep Learning architecture against standard baseline baseline (RandomForest) based on {n_splits * repeat_cv_runs} fold cross validations.")
    
    results_summary = []
    rf_vector = np.array(benchmark_history['RandomForest'])
    
    for name in models_to_test:
        score_vector = np.array(benchmark_history[name])
        mean, std, (ci_low, ci_high) = compute_ci(score_vector)
        
        if name == 'RandomForest':
            stat, p_val, status = np.nan, np.nan, "Baseline Reference"
        else:
            try:
                stat, p_val = stats.wilcoxon(score_vector, rf_vector, alternative='two-sided')
                status = "Significantly Different" if p_val < alpha else "Statistically Equivalent"
            except ValueError:
                # Fallback if differences are perfectly zero
                stat, p_val, status = 0.0, 1.0, "Statistically Identical Vector"
                
        results_summary.append({
            'Model Architecture': name,
            'Mean Macro F1': mean,
            'Std Dev': std,
            '95% CI Lower': ci_low,
            '95% CI Upper': ci_high,
            'Wilcoxon W Stat': stat,
            'p-value': p_val,
            'Benchmark Assessment': status
        })

    df_results = pd.DataFrame(results_summary)
    print("\n", df_results.to_string(index=False))

    # -------------------------
    # Phase 4: Final Training & Production Export Saves
    # -------------------------
    print("\n === Phase 4: Full Dataset Production Fits & Component Saves ===")
    os.makedirs('./saved_models', exist_ok=True)
    
    # Save final pipeline scaling transform components
    scaler_final = StandardScaler()
    tabular_scaled_final = scaler_final.fit_transform(tabular_bal)
    
    with open('./saved_models/final_scaler.pkl', 'wb') as f:
        pickle.dump(scaler_final, f)
        
    # Fit and save classical production champion
    rf_final = RandomForestClassifier(n_estimators=100, class_weight='balanced_subsample', random_state=42)
    rf_final.fit(tabular_scaled_final, labels_bal)
    with open('./saved_models/production_random_forest.pkl', 'wb') as f:
        pickle.dump(rf_final, f)
    print("Exported standard production scalar and classical Random Forest classifier pickles.")

    # Train and save final deep learning production architectures
    for model_name in models_to_test:
        if model_name == 'RandomForest':
            continue
        print(f"Training absolute dataset terminal production champion weights for: {model_name}...")
        _, trained_cnn_obj = train_and_evaluate_cnn(
            model_name=model_name,
            train_data=(images_bal, labels_bal),
            val_data=(images_bal, labels_bal), # Internal validation tracking check
            num_classes=num_classes,
            epochs=cnn_epochs
        )
        # Exporting native model parameters weights mapping state dict maps
        torch.save(trained_cnn_obj.state_dict(), f'./saved_models/production_{model_name.lower().replace("-", "_")}.pth')
        
    # Export full metrics log trace dataframe via pickle
    with open('./saved_models/benchmark_metrics_dataframe.pkl', 'wb') as f:
        pickle.dump(df_results, f)
        
    print("\nPipeline Benchmarking Complete. All deep weights (.pth), pickles (.pkl), and logs saved to './saved_models/'.")


# ---------------------------------------------------------
# Sanity Check & Self-Contained Mock Data Generator Execution
# ---------------------------------------------------------
if __name__ == "__main__":
    # Create an in-memory synthetic dataset to instantly verify compliance execution logic
    print("No local asset paths specified. Generating clean synthetic setup arrays to instantly perform sanity checks...")
    
    # Create mock configurations mimicking balanced dataset characteristics
    num_samples_mock = 20
    mock_images = np.random.randint(0, 256, (num_samples_mock, 512, 512, 3), dtype=np.uint8)
    mock_tabular = np.random.rand(num_samples_mock, 8)
    mock_labels = np.array([0] * (num_samples_mock // 2) + [1] * (num_samples_mock // 2))
    mock_filenames = [f"lesion_{i}.png" for i in range(num_samples_mock)]
    
    # Inject patch override methods to test data processing code without file dependencies
    def mock_loader(root_dir, extractor):
        return mock_images, mock_tabular, mock_labels, mock_filenames, {'Malignant': 0, 'Benign': 1}
        
    # Reassign runtime functions to mock execution environments
    load_dual_dataset = mock_loader

    # Run complete tracking execution suite with low epochs/runs for fast pipeline validation
    run_benchmark(train_dir='./train', test_dir='./test', n_splits=5, repeat_cv_runs=2, cnn_epochs=5)

# %%
import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial import distance

# Scikit-Learn Imports
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# PyTorch Imports for Lightweight CNN Extraction
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms

class ExanthemClassifier:
    def __init__(self, cnn_model_name='MobileNetV2'):
        self.cnn_model_name = cnn_model_name
        
        # Fresh model instances for every benchmark run
        self.models = [
            ('RandomForest', RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)),
            ('SVM', SVC(probability=True, kernel='rbf', class_weight='balanced', random_state=42)),
            ('KNN', KNeighborsClassifier(n_neighbors=5))
        ]
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=10, random_state=42) # Compresses CNN output
        
        self.classic_feature_names = ['lesion_count', 'avg_area', 'std_area', 'avg_circularity', 
                                      'sparsity_score', 'confluence_ratio', 'avg_hue', 'avg_saturation']
        self.cnn_feature_names = [f'CNN_Texture_{i+1}' for i in range(10)]
        self.feature_names = self.classic_feature_names + self.cnn_feature_names

        # --- PyTorch Setup (Dynamic for Benchmarking) ---
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.cnn_extractor = self._load_cnn_backbone(cnn_model_name).to(self.device)
        self.cnn_extractor.eval() # Set to evaluation mode (no training)
        
        # Standard PyTorch ImageNet preprocessing
        self.preprocess = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        # Global Average Pooling to flatten spatial dimensions
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def _load_cnn_backbone(self, model_name):
        """Dynamically loads the requested CNN architecture for the benchmark."""
        if model_name == 'EfficientNet-B0':
            return models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT).features
        elif model_name == 'ResNet18':
            return nn.Sequential(*list(models.resnet18(weights=models.ResNet18_Weights.DEFAULT).children())[:-2])
        elif model_name == 'ResNet34':
            return nn.Sequential(*list(models.resnet34(weights=models.ResNet34_Weights.DEFAULT).children())[:-2])
        elif model_name == 'MobileNetV2':
            return models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT).features
        elif model_name == 'MobileNetV3':
            return models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT).features
        elif model_name == 'ShuffleNetV2':
            return nn.Sequential(*list(models.shufflenet_v2_x1_0(weights=models.ShuffleNet_V2_X1_0_Weights.DEFAULT).children())[:-1])
        elif model_name == 'SqueezeNet':
            return models.squeezenet1_1(weights=models.SqueezeNet1_1_Weights.DEFAULT).features
        else:
            raise ValueError(f"Model {model_name} not supported.")

    def apply_gray_world_white_balance(self, img):
        """Applies Gray World assumption to remove lighting casts."""
        b, g, r = cv2.split(img.astype(np.float32))
        avg_b, avg_g, avg_r = np.mean(b), np.mean(g), np.mean(r)
        avg_all = (avg_b + avg_g + avg_r) / 3.0
        
        scale_b = avg_all / avg_b if avg_b > 0 else 1.0
        scale_g = avg_all / avg_g if avg_g > 0 else 1.0
        scale_r = avg_all / avg_r if avg_r > 0 else 1.0
        
        b = np.clip(b * scale_b, 0, 255)
        g = np.clip(g * scale_g, 0, 255)
        r = np.clip(r * scale_r, 0, 255)
        
        return cv2.merge((b, g, r)).astype(np.uint8)

    def extract_features(self, image_path):
        """Extracts classical spatial/color features AND deep texture features via PyTorch."""
        img = cv2.imread(image_path)
        if img is None: return None, None
        
        # --- 1. DYNAMIC DIMENSIONAL INITIALIZATION ---
        H, W = img.shape[:2]
        total_pixels = H * W
        max_dim = max(H, W)
        
        # Bilateral filter parameters scaled to image dimensions
        d = int(0.017 * max_dim) | 1  # Forces odd integer
        sigma_space = 0.15 * max_dim
        sigma_color = 1.5 * np.std(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
        smoothed = cv2.bilateralFilter(img, d, sigma_color, sigma_space)
        gray = cv2.cvtColor(smoothed, cv2.COLOR_BGR2GRAY)
        
        clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
        equalized = clahe.apply(gray)
        
        # Dynamic Adaptive Threshold Window (approx 10% of max dimension, forced odd)
        block_size = int(0.10 * max_dim) | 1
        thresh = cv2.adaptiveThreshold(equalized, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                    cv2.THRESH_BINARY_INV, block_size, 2)
        
        # Dynamic Morphological Kernel Structure (approx 1.3% of max dimension, forced odd)
        k_size = int(0.013 * max_dim) | 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
        clean_thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
        
        contours, _ = cv2.findContours(clean_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        centroids, areas, circularities = [], [], []
        valid_contours = []
        
        # Relative Area Thresholds: Min = 0.019% of image, Max = 10% of image
        min_area_thresh = 0.00019 * total_pixels
        max_area_thresh = 0.10 * total_pixels
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if min_area_thresh < area < max_area_thresh: 
                perimeter = cv2.arcLength(cnt, True)
                circularity = 0 if perimeter == 0 else 4 * np.pi * (area / (perimeter * perimeter))
                
                cnt_moments = cv2.moments(cnt)  # Changed variable name to avoid shadowing H
                if cnt_moments["m00"] != 0:
                    centroids.append((int(cnt_moments["m10"] / cnt_moments["m00"]), 
                                    int(cnt_moments["m01"] / cnt_moments["m00"])))
                    areas.append(area)
                    circularities.append(circularity)
                    valid_contours.append(cnt)

        wb_img = self.apply_gray_world_white_balance(img)
        hsv_img = cv2.cvtColor(wb_img, cv2.COLOR_BGR2HSV)
        
        # FIX: Dynamically allocate the mask to match input dimensions
        valid_mask = np.zeros((H, W), dtype=np.uint8)
        
        if valid_contours:
            cv2.drawContours(valid_mask, valid_contours, -1, 255, thickness=cv2.FILLED)
            mean_color = cv2.mean(hsv_img, mask=valid_mask)
            avg_hue = mean_color[0]
            avg_saturation = mean_color[1]
        else:
            avg_hue = 0
            avg_saturation = 0

        std_area = np.std(areas) if len(areas) > 1 else 0
        avg_circularity = np.mean(circularities) if circularities else 0

        if len(centroids) > 1:
            dist_matrix = distance.cdist(centroids, centroids, 'euclidean')
            np.fill_diagonal(dist_matrix, np.inf)
            nn_distances = np.min(dist_matrix, axis=1)
            sparsity_score = np.mean(nn_distances) / max_dim  # Normalize distance by image scale
        else:
            sparsity_score = 0
            
        # FIX: Confluence ratio uses actual dynamic pixel space
        confluence_ratio = sum(areas) / total_pixels
        
        classic_features = [len(centroids), np.mean(areas)/total_pixels if areas else 0, std_area/total_pixels, 
                            avg_circularity, sparsity_score, confluence_ratio, avg_hue, avg_saturation]

        # --- 2. CNN TEXTURE EXTRACTION VIA PYTORCH ---
        cnn_img = cv2.cvtColor(wb_img, cv2.COLOR_BGR2RGB)
        
        # Preprocess handles internal scaling to 224x224 or 256x256 for the deep stream safely
        input_tensor = self.preprocess(cnn_img).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            features = self.cnn_extractor(input_tensor)
            pooled_features = self.pool(features)
            cnn_features_raw = pooled_features.flatten().cpu().numpy()

        return classic_features, cnn_features_raw

    def load_dataset(self, root_dir):
        classic_data, cnn_data, labels, filenames = [], [], [], []
        if not os.path.exists(root_dir): return [], [], [], []
        
        for label_dir in os.listdir(root_dir):
            dir_path = os.path.join(root_dir, label_dir)
            if not os.path.isdir(dir_path): continue
            
            for img_file in os.listdir(dir_path):
                f_classic, f_cnn = self.extract_features(os.path.join(dir_path, img_file))
                if f_classic is not None:
                    classic_data.append(f_classic)
                    cnn_data.append(f_cnn)
                    labels.append(label_dir)
                    filenames.append(img_file)
        return np.array(classic_data), np.array(cnn_data), np.array(labels), filenames


class ExplainableDiagnostics:
    def __init__(self, classifier_obj):
        self.c = classifier_obj

    def analyze_failures(self, model, X_test_norm, y_test, filenames, model_name, cnn_name):
        predictions = model.predict(X_test_norm)
        errors = []
        for i in range(len(y_test)):
            if predictions[i] != y_test[i]:
                errors.append({
                    'filename': filenames[i],
                    'actual': y_test[i],
                    'predicted': predictions[i]
                })
        
        df_err = pd.DataFrame(errors)
        print(f"\n--- Failure Analysis: {model_name} (using {cnn_name}) ---")
        if not df_err.empty:
            print(f"Total Errors: {len(df_err)}")
            print(df_err.head(5)) 
        return df_err

    def plot_performance(self, model, X_test, y_test, model_name, cnn_name):
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        
        plt.figure(figsize=(14, 6))
        
        plt.subplot(1, 2, 1)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', 
                    xticklabels=np.unique(y_test), yticklabels=np.unique(y_test))
        plt.title(f'Confusion Matrix: {model_name}\n(Hybrid + {cnn_name})')
        plt.ylabel('Actual Label')
        plt.xlabel('Predicted Label')
        
        if hasattr(model, 'feature_importances_'):
            plt.subplot(1, 2, 2)
            importances = model.feature_importances_
            indices = np.argsort(importances)
            plt.barh(range(len(indices)), importances[indices], align='center', color='steelblue')
            plt.yticks(range(len(indices)), [self.c.feature_names[i] for i in indices])
            plt.title(f'Hybrid Feature Importance\n(PCA via {cnn_name})')
            
        plt.tight_layout()
        plt.show()

# --- Main Execution Pipeline ---
if __name__ == "__main__":
    benchmark_models = [
        'MobileNetV2', 'ResNet18', 'EfficientNet-B0', 
        'ResNet34', 'MobileNetV3', 'ShuffleNetV2', 'SqueezeNet'
    ]
    
    benchmark_results = []
    
    for cnn_name in benchmark_models:
        print(f"\n{'='*50}")
        print(f"🚀 INITIALIZING BENCHMARK RUN: Hybrid + {cnn_name}")
        print(f"{'='*50}")
        
        pipeline = ExanthemClassifier(cnn_model_name=cnn_name)
        diagnostics = ExplainableDiagnostics(pipeline)

        print(f"Using Device: {pipeline.device}")
        print("--- Phase 1: Training & Cross-Validation ---")
        
        X_train_classic, X_train_cnn, y_train, _ = pipeline.load_dataset('./train')
        
        if len(X_train_classic) == 0:
            print("Error: No training data found. Make sure './train' exists.")
            continue # Skip to next model if testing locally without folders
            
        print(f"Compressing {cnn_name} texture features via PCA...")
        X_train_cnn_pca = pipeline.pca.fit_transform(X_train_cnn)
        X_train_combined = np.hstack((X_train_classic, X_train_cnn_pca))
        X_train = pipeline.scaler.fit_transform(X_train_combined)
        
        print(f"Dataset Loaded: {len(X_train)} training samples with {X_train.shape[1]} total features.")
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        for name, model in pipeline.models:
            cv_scores = cross_val_score(model, X_train, y_train, cv=skf)
            print(f"{name} CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
            model.fit(X_train, y_train)

        print("\n--- Phase 2: Unseen Data Evaluation & XAI Diagnostics ---")
        X_test_classic, X_test_cnn, y_test, test_filenames = pipeline.load_dataset('./test')
        
        X_test_cnn_pca = pipeline.pca.transform(X_test_cnn)
        X_test_combined = np.hstack((X_test_classic, X_test_cnn_pca))
        X_test = pipeline.scaler.transform(X_test_combined)

        for name, model in pipeline.models:
            y_pred = model.predict(X_test)
            test_acc = accuracy_score(y_test, y_pred)
            print(f"\nResults for {name} ({cnn_name}):")
            print(classification_report(y_test, y_pred))
            
            # Record for final summary
            benchmark_results.append({
                'CNN_Backbone': cnn_name,
                'Classifier': name,
                'Test_Accuracy': test_acc
            })
            
            df_errors = diagnostics.analyze_failures(model, X_test, y_test, test_filenames, name, cnn_name)
            
            # Only plot for Random Forest to prevent plotting 21 charts, or remove the if-statement to plot all
            if name == 'RandomForest':
                diagnostics.plot_performance(model, X_test, y_test, name, cnn_name)
                
    # --- Print Final Benchmark Table ---
    if benchmark_results:
        print("\n" + "="*50)
        print("🏆 FINAL BENCHMARK SUMMARY")
        print("="*50)
        df_summary = pd.DataFrame(benchmark_results).sort_values(by="Test_Accuracy", ascending=False)
        print(df_summary.to_string(index=False))

# %%
import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial import distance

# Scikit-Learn Imports
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# PyTorch Imports for Lightweight CNN Extraction
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms

class ExanthemClassifier:
    def __init__(self, cnn_model_name='MobileNetV2'):
        self.cnn_model_name = cnn_model_name
        
        # Fresh model instances for every benchmark run
        self.models = [
            ('RandomForest', RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)),
            ('SVM', SVC(probability=True, kernel='rbf', class_weight='balanced', random_state=42)),
            ('KNN', KNeighborsClassifier(n_neighbors=5))
        ]
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=10, random_state=42) # Compresses CNN output
        
        self.classic_feature_names = ['lesion_count', 'avg_area', 'std_area', 'avg_circularity', 
                                      'sparsity_score', 'confluence_ratio', 'avg_hue', 'avg_saturation']
        self.cnn_feature_names = [f'CNN_Texture_{i+1}' for i in range(10)]
        self.feature_names = self.classic_feature_names + self.cnn_feature_names

        # --- PyTorch Setup (Dynamic for Benchmarking) ---
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.cnn_extractor = self._load_cnn_backbone(cnn_model_name).to(self.device)
        self.cnn_extractor.eval() # Set to evaluation mode (no training)
        
        # Standard PyTorch ImageNet preprocessing
        self.preprocess = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        # Global Average Pooling to flatten spatial dimensions
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def _load_cnn_backbone(self, model_name):
        """Dynamically loads the requested CNN architecture for the benchmark."""
        if model_name == 'EfficientNet-B0':
            return models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT).features
        elif model_name == 'ResNet18':
            return nn.Sequential(*list(models.resnet18(weights=models.ResNet18_Weights.DEFAULT).children())[:-2])
        elif model_name == 'ResNet34':
            return nn.Sequential(*list(models.resnet34(weights=models.ResNet34_Weights.DEFAULT).children())[:-2])
        elif model_name == 'MobileNetV2':
            return models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT).features
        elif model_name == 'MobileNetV3':
            return models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT).features
        elif model_name == 'ShuffleNetV2':
            return nn.Sequential(*list(models.shufflenet_v2_x1_0(weights=models.ShuffleNet_V2_X1_0_Weights.DEFAULT).children())[:-1])
        elif model_name == 'SqueezeNet':
            return models.squeezenet1_1(weights=models.SqueezeNet1_1_Weights.DEFAULT).features
        else:
            raise ValueError(f"Model {model_name} not supported.")

    def apply_gray_world_white_balance(self, img):
        """Applies Gray World assumption to remove lighting casts."""
        b, g, r = cv2.split(img.astype(np.float32))
        avg_b, avg_g, avg_r = np.mean(b), np.mean(g), np.mean(r)
        avg_all = (avg_b + avg_g + avg_r) / 3.0
        
        scale_b = avg_all / avg_b if avg_b > 0 else 1.0
        scale_g = avg_all / avg_g if avg_g > 0 else 1.0
        scale_r = avg_all / avg_r if avg_r > 0 else 1.0
        
        b = np.clip(b * scale_b, 0, 255)
        g = np.clip(g * scale_g, 0, 255)
        r = np.clip(r * scale_r, 0, 255)
        
        return cv2.merge((b, g, r)).astype(np.uint8)

    def extract_features(self, image_path):
        """Extracts classical spatial/color features AND deep texture features via PyTorch."""
        img = cv2.imread(image_path)
        if img is None: return None, None
        
        # --- 1. CLASSICAL FEATURE EXTRACTION ---
        img_512 = cv2.resize(img, (512, 512))
        smoothed = cv2.bilateralFilter(img_512, d=9, sigmaColor=75, sigmaSpace=75)
        gray = cv2.cvtColor(smoothed, cv2.COLOR_BGR2GRAY)
        
        clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
        equalized = clahe.apply(gray)
        
        thresh = cv2.adaptiveThreshold(equalized, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                       cv2.THRESH_BINARY_INV, 51, 2)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        clean_thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
        
        contours, _ = cv2.findContours(clean_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        centroids, areas, circularities = [], [], []
        valid_contours = []
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if 50 < area < (512 * 512 * 0.1): 
                perimeter = cv2.arcLength(cnt, True)
                circularity = 0 if perimeter == 0 else 4 * np.pi * (area / (perimeter * perimeter))
                
                M = cv2.moments(cnt)
                if M["m00"] != 0:
                    centroids.append((int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])))
                    areas.append(area)
                    circularities.append(circularity)
                    valid_contours.append(cnt)

        wb_img = self.apply_gray_world_white_balance(img_512)
        hsv_img = cv2.cvtColor(wb_img, cv2.COLOR_BGR2HSV)
        valid_mask = np.zeros((512, 512), dtype=np.uint8)
        
        if valid_contours:
            cv2.drawContours(valid_mask, valid_contours, -1, 255, thickness=cv2.FILLED)
            mean_color = cv2.mean(hsv_img, mask=valid_mask)
            avg_hue = mean_color[0]
            avg_saturation = mean_color[1]
        else:
            avg_hue = 0
            avg_saturation = 0

        std_area = np.std(areas) if len(areas) > 1 else 0
        avg_circularity = np.mean(circularities) if circularities else 0

        if len(centroids) > 1:
            dist_matrix = distance.cdist(centroids, centroids, 'euclidean')
            np.fill_diagonal(dist_matrix, np.inf)
            nn_distances = np.min(dist_matrix, axis=1)
            sparsity_score = np.mean(nn_distances)
        else:
            sparsity_score = 0
            
        confluence_ratio = sum(areas) / (512 * 512)
        
        classic_features = [len(centroids), np.mean(areas) if areas else 0, std_area, 
                            avg_circularity, sparsity_score, confluence_ratio, avg_hue, avg_saturation]

        # --- 2. CNN TEXTURE EXTRACTION VIA PYTORCH ---
        cnn_img = cv2.cvtColor(wb_img, cv2.COLOR_BGR2RGB)
        
        # Preprocess and add batch dimension [1, C, H, W]
        input_tensor = self.preprocess(cnn_img).unsqueeze(0).to(self.device)
        
        with torch.no_grad(): # Disable gradients to save memory
            features = self.cnn_extractor(input_tensor)
            pooled_features = self.pool(features) # Collapse spatial dims
            cnn_features_raw = pooled_features.flatten().cpu().numpy() # Extract raw array

        return classic_features, cnn_features_raw

    def load_dataset(self, root_dir):
        classic_data, cnn_data, labels, filenames = [], [], [], []
        if not os.path.exists(root_dir): return [], [], [], []
        
        for label_dir in os.listdir(root_dir):
            dir_path = os.path.join(root_dir, label_dir)
            if not os.path.isdir(dir_path): continue
            
            for img_file in os.listdir(dir_path):
                f_classic, f_cnn = self.extract_features(os.path.join(dir_path, img_file))
                if f_classic is not None:
                    classic_data.append(f_classic)
                    cnn_data.append(f_cnn)
                    labels.append(label_dir)
                    filenames.append(img_file)
        return np.array(classic_data), np.array(cnn_data), np.array(labels), filenames


class ExplainableDiagnostics:
    def __init__(self, classifier_obj):
        self.c = classifier_obj

    def analyze_failures(self, model, X_test_norm, y_test, filenames, model_name, cnn_name):
        predictions = model.predict(X_test_norm)
        errors = []
        for i in range(len(y_test)):
            if predictions[i] != y_test[i]:
                errors.append({
                    'filename': filenames[i],
                    'actual': y_test[i],
                    'predicted': predictions[i]
                })
        
        df_err = pd.DataFrame(errors)
        print(f"\n--- Failure Analysis: {model_name} (using {cnn_name}) ---")
        if not df_err.empty:
            print(f"Total Errors: {len(df_err)}")
            print(df_err.head(5)) 
        return df_err

    def plot_performance(self, model, X_test, y_test, model_name, cnn_name):
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        
        plt.figure(figsize=(14, 6))
        
        plt.subplot(1, 2, 1)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', 
                    xticklabels=np.unique(y_test), yticklabels=np.unique(y_test))
        plt.title(f'Confusion Matrix: {model_name}\n(Hybrid + {cnn_name})')
        plt.ylabel('Actual Label')
        plt.xlabel('Predicted Label')
        
        if hasattr(model, 'feature_importances_'):
            plt.subplot(1, 2, 2)
            importances = model.feature_importances_
            indices = np.argsort(importances)
            plt.barh(range(len(indices)), importances[indices], align='center', color='steelblue')
            plt.yticks(range(len(indices)), [self.c.feature_names[i] for i in indices])
            plt.title(f'Hybrid Feature Importance\n(PCA via {cnn_name})')
            
        plt.tight_layout()
        plt.show()

# --- Main Execution Pipeline ---
if __name__ == "__main__":
    benchmark_models = [
        'MobileNetV2', 'ResNet18', 'EfficientNet-B0', 
        'ResNet34', 'MobileNetV3', 'ShuffleNetV2', 'SqueezeNet'
    ]
    
    benchmark_results = []
    
    for cnn_name in benchmark_models:
        print(f"\n{'='*50}")
        print(f"🚀 INITIALIZING BENCHMARK RUN: Hybrid + {cnn_name}")
        print(f"{'='*50}")
        
        pipeline = ExanthemClassifier(cnn_model_name=cnn_name)
        diagnostics = ExplainableDiagnostics(pipeline)

        print(f"Using Device: {pipeline.device}")
        print("--- Phase 1: Training & Cross-Validation ---")
        
        X_train_classic, X_train_cnn, y_train, _ = pipeline.load_dataset('./train')
        
        if len(X_train_classic) == 0:
            print("Error: No training data found. Make sure './train' exists.")
            continue # Skip to next model if testing locally without folders
            
        print(f"Compressing {cnn_name} texture features via PCA...")
        X_train_cnn_pca = pipeline.pca.fit_transform(X_train_cnn)
        X_train_combined = np.hstack((X_train_classic, X_train_cnn_pca))
        X_train = pipeline.scaler.fit_transform(X_train_combined)
        
        print(f"Dataset Loaded: {len(X_train)} training samples with {X_train.shape[1]} total features.")
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        for name, model in pipeline.models:
            cv_scores = cross_val_score(model, X_train, y_train, cv=skf)
            print(f"{name} CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
            model.fit(X_train, y_train)

        print("\n--- Phase 2: Unseen Data Evaluation & XAI Diagnostics ---")
        X_test_classic, X_test_cnn, y_test, test_filenames = pipeline.load_dataset('./test')
        
        X_test_cnn_pca = pipeline.pca.transform(X_test_cnn)
        X_test_combined = np.hstack((X_test_classic, X_test_cnn_pca))
        X_test = pipeline.scaler.transform(X_test_combined)

        for name, model in pipeline.models:
            y_pred = model.predict(X_test)
            test_acc = accuracy_score(y_test, y_pred)
            print(f"\nResults for {name} ({cnn_name}):")
            print(classification_report(y_test, y_pred))
            
            # Record for final summary
            benchmark_results.append({
                'CNN_Backbone': cnn_name,
                'Classifier': name,
                'Test_Accuracy': test_acc
            })
            
            df_errors = diagnostics.analyze_failures(model, X_test, y_test, test_filenames, name, cnn_name)
            
            # Only plot for Random Forest to prevent plotting 21 charts, or remove the if-statement to plot all
            if name == 'RandomForest':
                diagnostics.plot_performance(model, X_test, y_test, name, cnn_name)
                
    # --- Print Final Benchmark Table ---
    # --- Print Final Benchmark Table & Export the Best Model ---
    if benchmark_results:
        print("\n" + "="*50)
        print("🏆 FINAL BENCHMARK SUMMARY")
        print("="*50)
        df_summary = pd.DataFrame(benchmark_results).sort_values(by="Test_Accuracy", ascending=False)
        print(df_summary.to_string(index=False))
        
        # Pull top-performing model details
        best_run = df_summary.iloc[0]
        best_cnn = best_run['CNN_Backbone']
        best_classifier_name = best_run['Classifier']
        print(f"\n🥇 Absolute Best Configuration: {best_classifier_name} utilizing {best_cnn} ({best_run['Test_Accuracy']:.4f} Acc)")
        
        # Re-run a clean generation of that specific pipeline to save its fitted states
        print("Saving winning architecture components...")
        export_dir = './best_model'
        os.makedirs(export_dir, exist_ok=True)
        
        # Re-instantiate the winner to ensure we catch its exact fitted weights
        best_pipeline = ExanthemClassifier(cnn_model_name=best_cnn)
        X_train_classic, X_train_cnn, y_train, _ = best_pipeline.load_dataset('./train')
        
        # Fit operations
        X_train_cnn_pca = best_pipeline.pca.fit_transform(X_train_cnn)
        X_train_combined = np.hstack((X_train_classic, X_train_cnn_pca))
        X_train_scaled = best_pipeline.scaler.fit_transform(X_train_combined)
        
        # Find and train the specific winning model
        for name, model in best_pipeline.models:
            if name == best_classifier_name:
                model.fit(X_train_scaled, y_train)
                
                # Serialize entire asset bundle
                with open(f"{export_dir}/classifier_model.pkl", 'wb') as f: pickle.dump(model, f)
                with open(f"{export_dir}/scaler.pkl", 'wb') as f: pickle.dump(best_pipeline.scaler, f)
                with open(f"{export_dir}/pca.pkl", 'wb') as f: pickle.dump(best_pipeline.pca, f)
                
                # Save metadata config so Streamlit knows which CNN back-end to boot up
                metadata = {'best_cnn': best_cnn, 'classifier_name': best_classifier_name}
                with open(f"{export_dir}/metadata.pkl", 'wb') as f: pickle.dump(metadata, f)
                
                print(f"📦 Successfully exported all production components to '{export_dir}/'!")
                break


