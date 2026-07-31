import os
import sys
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import cv2

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline

from src.utils import compute_metrics


def extract_hog_features(img, resize=(128, 128)):
    from skimage.feature import hog
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, resize)
    features = hog(img, orientations=9, pixels_per_cell=(8, 8),
                   cells_per_block=(2, 2), block_norm="L2-Hys",
                   feature_vector=True)
    return features


def extract_lbp_features(img, resize=(128, 128), radius=1, n_points=8):
    from skimage.feature import local_binary_pattern
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, resize)
    lbp = local_binary_pattern(img, n_points, radius, method="uniform")
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3),
                           range=(0, n_points + 2))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-7)
    return hist


def extract_features_from_df(df, feature_fn, desc="Extracting features"):
    X, y = [], []
    errors = 0
    for _, row in tqdm(df.iterrows(), total=len(df), desc=desc):
        img = cv2.imread(row["filepath"])
        if img is None:
            errors += 1
            continue
        try:
            feat = feature_fn(img)
            X.append(feat)
            y.append(row["label"])
        except Exception as e:
            errors += 1
            continue
    if errors:
        print(f"[WARN] {errors} images failed to load/process")
    return np.array(X), np.array(y)


def run_baseline(config_path="configs/default.yaml", feature_type="hog", classifier_type="svm"):
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    splits_dir = Path(cfg["data"]["splits_dir"])
    num_classes = cfg["data"].get("num_classes", 4)

    train_df = pd.read_csv(splits_dir / "train.csv")
    test_df = pd.read_csv(splits_dir / "test.csv")

    print(f"[INFO] Extracting {feature_type.upper()} features...")
    feature_fn = extract_hog_features if feature_type == "hog" else extract_lbp_features

    X_train, y_train = extract_features_from_df(train_df, feature_fn, desc="Train features")
    X_test, y_test = extract_features_from_df(test_df, feature_fn, desc="Test features")

    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_test_enc = le.transform(y_test)

    print(f"[INFO] Training set: {X_train.shape}, Test set: {X_test.shape}")
    print(f"[INFO] Classes: {le.classes_}")

    if classifier_type == "svm":
        clf = Pipeline([
            ("scaler", StandardScaler()),
            ("svm", SVC(kernel="rbf", C=1.0, gamma="scale", probability=True, random_state=42))
        ])
    else:
        clf = Pipeline([
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(max_iter=1000, multi_class="multinomial", random_state=42))
        ])

    print(f"[INFO] Training {classifier_type.upper()}...")
    clf.fit(X_train, y_train_enc)

    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)

    metrics = compute_metrics(y_test_enc, y_pred, y_prob, num_classes=len(le.classes_))
    print(f"\n--- Classical Baseline ({feature_type.upper()} + {classifier_type.upper()}) Results ---")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    return metrics


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--feature", choices=["hog", "lbp"], default="hog")
    parser.add_argument("--classifier", choices=["svm", "logreg"], default="svm")
    args = parser.parse_args()
    run_baseline(args.config, args.feature, args.classifier)
