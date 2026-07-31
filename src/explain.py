import os
import yaml
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
import cv2

from src.models import build_model
from src.datasets import LABEL_MAP_4CLASS, LABEL_MAP_3CLASS


def generate_gradcam(model, input_tensor, target_layer, target_class=None):
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

    cam = GradCAM(model=model, target_layers=[target_layer])
    if target_class is not None:
        targets = [ClassifierOutputTarget(target_class)]
    else:
        targets = None
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
    return grayscale_cam[0, :]


def overlay_heatmap(img_bgr, heatmap, alpha=0.4):
    heatmap = np.uint8(255 * heatmap)
    heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img_bgr, 1 - alpha, heatmap_colored, alpha, 0)
    return overlay


def run_explain(config_path="configs/default.yaml", checkpoint_path=None,
                sample_dir=None, output_dir="results/gradcam_samples"):
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    num_classes = cfg["data"].get("num_classes", 4)
    backbone = cfg["model"]["backbone"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    label_map = LABEL_MAP_4CLASS if num_classes == 4 else LABEL_MAP_3CLASS
    class_names = [k for k, v in sorted(label_map.items(), key=lambda x: x[1])]

    model = build_model(backbone_name=backbone, num_classes=num_classes,
                        pretrained=False, drop_rate=0).to(device)
    model.eval()

    if checkpoint_path is None:
        checkpoint_path = Path("results/checkpoints") / f"best_{backbone}_seed{cfg.get('seed', 42)}_cls{num_classes}.pt"

    if not Path(checkpoint_path).exists():
        print(f"[ERROR] Checkpoint not found at {checkpoint_path}")
        return

    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"[INFO] Loaded checkpoint from {checkpoint_path}")

    # Determine target layer for Grad-CAM
    if "densenet" in backbone:
        target_layer = model.model.features.denseblock4
    elif "resnet" in backbone:
        target_layer = model.model.layer4
    elif "efficientnet" in backbone:
        target_layer = model.model.blocks[-1]
    elif "vit" in backbone or "swin" in backbone:
        target_layer = model.model.blocks[-1]
    else:
        target_layer = model.model.layer4
        print(f"[WARN] Unknown backbone {backbone}, using last conv layer heuristic")

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    from torchvision import transforms as T
    normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    # Use sample images from test set if no directory specified
    if sample_dir is None:
        splits_dir = Path(cfg["data"]["splits_dir"])
        test_csv = splits_dir / "test.csv"
        if not test_csv.exists():
            print("[ERROR] No sample_dir and no test.csv found")
            return
        import pandas as pd
        df = pd.read_csv(test_csv)
        # Pick a few examples per class
        sample_paths = df.groupby("label").apply(lambda x: x.sample(min(3, len(x)), random_state=42))
        sample_paths = sample_paths["filepath"].tolist()
    else:
        sample_paths = list(Path(sample_dir).rglob("*.png")) + list(Path(sample_dir).rglob("*.jpg")) + list(Path(sample_dir).rglob("*.jpeg"))

    for img_path in sample_paths:
        if not os.path.exists(img_path):
            continue
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (224, 224))
        img_tensor = torch.tensor(img_resized).permute(2, 0, 1).float() / 255.0
        img_tensor = normalize(img_tensor).unsqueeze(0).to(device)

        # Predict
        with torch.no_grad():
            output = model(img_tensor)
            probs = torch.softmax(output, dim=1)
            pred_class = torch.argmax(probs, dim=1).item()
            confidence = probs[0, pred_class].item()

        # Generate Grad-CAM for predicted class
        heatmap = generate_gradcam(model, img_tensor, target_layer, target_class=pred_class)
        overlay = overlay_heatmap(img_bgr, heatmap)

        # Save overlay
        stem = Path(img_path).stem
        overlay_path = out_dir / f"{stem}_gradcam.png"
        cv2.imwrite(str(overlay_path), overlay)

        pred_label = class_names[pred_class] if pred_class < len(class_names) else str(pred_class)
        print(f"  {stem}: predicted={pred_label} ({confidence:.3f}) -> {overlay_path.name}")

    print(f"[SUCCESS] Grad-CAM overlays saved to {out_dir}/")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--sample-dir", default=None)
    parser.add_argument("--output-dir", default="results/gradcam_samples")
    args = parser.parse_args()
    run_explain(args.config, args.checkpoint, args.sample_dir, args.output_dir)
