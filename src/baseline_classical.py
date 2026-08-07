import os
import cv2
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
from tqdm import tqdm
from PIL import Image

from src.utils import set_seed, compute_metrics

LABEL_MAP_4CLASS = {
    "normal": 0,
    "bacterial_pneumonia": 1,
    "viral_pneumonia": 2,
    "tuberculosis": 3
}

LABEL_MAP_3CLASS = {
    "normal": 0,
    "bacterial_pneumonia": 1,
    "viral_pneumonia": 2
}

def extract_hog_features(filepath, image_size=128):
    """Extract HOG (Histogram of Oriented Gradients) and color intensity features."""
    try:
        img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        if img is None:
            img = np.array(Image.open(filepath).convert("L"))
        
        img_resized = cv2.resize(img, (image_size, image_size))
        
        # HOG Feature extraction
        hog_feat = hog(
            img_resized,
            orientations=9,
            pixels_per_cell=(16, 16),
            cells_per_block=(2, 2),
            block_norm='L2-Hys',
            visualize=False
        )
        
        # Intensity Histogram feature (32 bins)
        hist_feat, _ = np.histogram(img_resized, bins=32, range=(0, 256), density=True)
        
        # Concatenate features
        feature_vec = np.hstack([hog_feat, hist_feat])
        return feature_vec
    except Exception as e:
        print(f"[WARN] Error extracting HOG features for {filepath}: {e}")
        return None

def run_classical_baseline(config_path="configs/default.yaml"):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    
    seed = cfg.get("seed", 42)
    set_seed(seed)
    num_classes = cfg["data"].get("num_classes", 4)
    splits_dir = Path(cfg["data"]["splits_dir"])
    label_map = LABEL_MAP_4CLASS if num_classes == 4 else LABEL_MAP_3CLASS
    
    print("[INFO] Loading datasets for Classical ML Baseline (HOG + SVM / LogReg)...")
    train_df = pd.read_csv(splits_dir / "train.csv")
    test_df = pd.read_csv(splits_dir / "test.csv")
    ext_df = pd.read_csv(splits_dir / "external_test.csv") if (splits_dir / "external_test.csv").exists() else None
    
    print(f"[INFO] Extracting HOG features for Train set ({len(train_df)} images)...")
    X_train, y_train = [], []
    for _, row in tqdm(train_df.iterrows(), total=len(train_df), desc="Train Features"):
        feat = extract_hog_features(row["filepath"])
        if feat is not None:
            X_train.append(feat)
            y_train.append(label_map[row["label"]])
            
    print(f"[INFO] Extracting HOG features for Internal Test set ({len(test_df)} images)...")
    X_test, y_test = [], []
    for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Test Features"):
        feat = extract_hog_features(row["filepath"])
        if feat is not None:
            X_test.append(feat)
            y_test.append(label_map[row["label"]])

    X_ext, y_ext = [], []
    if ext_df is not None:
        print(f"[INFO] Extracting HOG features for External Test set ({len(ext_df)} images)...")
        for _, row in tqdm(ext_df.iterrows(), total=len(ext_df), desc="External Features"):
            feat = extract_hog_features(row["filepath"])
            if feat is not None:
                X_ext.append(feat)
                y_ext.append(label_map[row["label"]])

    X_train, y_train = np.array(X_train), np.array(y_train)
    X_test, y_test = np.array(X_test), np.array(y_test)
    X_ext, y_ext = np.array(X_ext), np.array(y_ext)

    # Standard Scaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_ext_scaled = scaler.transform(X_ext) if len(X_ext) > 0 else None

    # Train SVM Model
    print("[INFO] Training SVM classifier (kernel='rbf', probability=True)...")
    svm_model = SVC(kernel="rbf", C=1.0, probability=True, random_state=seed)
    svm_model.fit(X_train_scaled, y_train)

    # Evaluate on Internal Test set
    preds_test = svm_model.predict(X_test_scaled)
    probs_test = svm_model.predict_proba(X_test_scaled)
    test_metrics = compute_metrics(y_test, preds_test, probs_test, num_classes=num_classes)

    print("\n--- Classical ML Baseline (SVM + HOG) - Internal Test Results ---")
    for k, v in test_metrics.items():
        print(f"  {k}: {v:.4f}")

    ext_metrics = None
    if X_ext_scaled is not None and len(X_ext_scaled) > 0:
        preds_ext = svm_model.predict(X_ext_scaled)
        probs_ext = svm_model.predict_proba(X_ext_scaled)
        ext_metrics = compute_metrics(y_ext, preds_ext, probs_ext, num_classes=num_classes)
        print("\n--- Classical ML Baseline (SVM + HOG) - External Test Results (Montgomery) ---")
        for k, v in ext_metrics.items():
            print(f"  {k}: {v:.4f}")

    # Export report for thesis documentation
    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)
    report_path = results_dir / "classical_baseline.md"
    
    md_content = f"# Classical ML Baseline Benchmark (HOG + SVM)\n\n"
    md_content += f"- **Feature Extractor:** HOG (orientations=9, pixels_per_cell=16x16) + Intensity Histogram (32 bins)\n"
    md_content += f"- **Classifier:** Support Vector Machine (RBF Kernel)\n"
    md_content += f"- **Random Seed:** {seed}\n\n"
    md_content += "## Performance Metrics\n\n"
    md_content += "| Evaluation Set | Accuracy | Precision (Macro) | Recall (Macro) | F1 (Macro) | AUC (Macro) |\n"
    md_content += "|---|---|---|---|---|---|\n"
    md_content += f"| **Internal Test Set** | {test_metrics['accuracy']:.4f} | {test_metrics['precision_macro']:.4f} | {test_metrics['recall_macro']:.4f} | {test_metrics['f1_macro']:.4f} | {test_metrics['auc_macro']:.4f} |\n"
    if ext_metrics:
        md_content += f"| **External Test Set (Montgomery)** | {ext_metrics['accuracy']:.4f} | {ext_metrics['precision_macro']:.4f} | {ext_metrics['recall_macro']:.4f} | {ext_metrics['f1_macro']:.4f} | {ext_metrics['auc_macro']:.4f} |\n"

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(md_content)

    print(f"\n[SUCCESS] Exported Classical ML Baseline results to {report_path}")

if __name__ == "__main__":
    run_classical_baseline()
