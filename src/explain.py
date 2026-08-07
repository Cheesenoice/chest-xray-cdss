import os
import cv2
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
import yaml
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from src.models import build_model
from src.datasets import LABEL_MAP_4CLASS, LABEL_MAP_3CLASS, get_transforms

INV_LABEL_MAP_4CLASS = {v: k for k, v in LABEL_MAP_4CLASS.items()}
INV_LABEL_MAP_3CLASS = {v: k for k, v in LABEL_MAP_3CLASS.items()}

def get_target_layer(model, backbone_name):
    """Retrieve the target convolutional layer for Grad-CAM visualization."""
    if "densenet" in backbone_name:
        return [model.model.features.denseblock4]
    elif "resnet" in backbone_name:
        return [model.model.layer4[-1]]
    elif "efficientnet" in backbone_name:
        return [model.model.conv_head]
    else:
        # Fallback to last conv module
        conv_layers = [module for module in model.modules() if isinstance(module, torch.nn.Conv2d)]
        return [conv_layers[-1]]

def generate_gradcam_overlay(model, image_path, target_layer, label_map, device, image_size=224):
    """Generate Grad-CAM heatmap overlay for a single image."""
    model.eval()
    
    # Read original image
    orig_img = cv2.imread(image_path)
    if orig_img is None:
        orig_img = np.array(Image.open(image_path).convert("RGB"))
    else:
        orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
    
    orig_resized = cv2.resize(orig_img, (image_size, image_size))
    rgb_float = orig_resized.astype(np.float32) / 255.0

    # Transform for model inference
    transform = get_transforms(image_size=image_size, is_train=False)
    input_tensor = transform(image=orig_img)["image"].unsqueeze(0).to(device)

    # Model inference
    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        pred_idx = int(np.argmax(probs))
        confidence = probs[pred_idx]

    inv_map = {v: k for k, v in label_map.items()}
    pred_label = inv_map.get(pred_idx, "Unknown")

    # Grad-CAM extraction
    cam = GradCAM(model=model, target_layers=target_layer)
    targets = [ClassifierOutputTarget(pred_idx)]
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]

    # Visual overlay
    visualization = show_cam_on_image(rgb_float, grayscale_cam, use_rgb=True)

    return orig_resized, visualization, pred_label, confidence, probs

def generate_gradcam_gallery(config_path="configs/default.yaml", checkpoint_path=None, num_samples_per_class=2):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = cfg["data"].get("num_classes", 4)
    backbone = cfg["model"]["backbone"]
    seed = cfg.get("seed", 42)
    label_map = LABEL_MAP_4CLASS if num_classes == 4 else LABEL_MAP_3CLASS
    inv_label_map = {v: k for k, v in label_map.items()}

    # Checkpoint path
    if checkpoint_path is None:
        checkpoint_path = f"results/checkpoints/best_{backbone}_seed{seed}.pt"

    print(f"[INFO] Building model {backbone}...")
    model = build_model(backbone_name=backbone, num_classes=num_classes, pretrained=False).to(device)

    if os.path.exists(checkpoint_path):
        print(f"[INFO] Loading checkpoint from {checkpoint_path}...")
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        print(f"[WARN] Checkpoint {checkpoint_path} not found. Running with initial weights for dry-run testing.")

    target_layer = get_target_layer(model, backbone)

    # Load test split
    splits_dir = Path(cfg["data"]["splits_dir"])
    test_df = pd.read_csv(splits_dir / "test.csv")

    output_dir = Path("results/figures/gradcam_samples")
    output_dir.mkdir(parents=True, exist_ok=True)

    sampled_records = []
    for lbl_name in label_map.keys():
        sub_df = test_df[test_df["label"] == lbl_name]
        if len(sub_df) > 0:
            samples = sub_df.sample(min(num_samples_per_class, len(sub_df)), random_state=seed)
            sampled_records.append(samples)

    if not sampled_records:
        print("[ERROR] No samples found in test set.")
        return

    sample_df = pd.concat(sampled_records, ignore_index=True)

    fig, axes = plt.subplots(len(sample_df), 2, figsize=(10, 4 * len(sample_df)))
    if len(sample_df) == 1:
        axes = np.expand_dims(axes, 0)

    print(f"[INFO] Generating Grad-CAM heatmaps for {len(sample_df)} test samples...")
    for idx, (_, row) in enumerate(sample_df.iterrows()):
        orig_img, cam_overlay, pred_label, conf, _ = generate_gradcam_overlay(
            model, row["filepath"], target_layer, label_map, device, image_size=cfg["data"].get("image_size", 224)
        )

        axes[idx, 0].imshow(orig_img)
        axes[idx, 0].set_title(f"Original X-Ray\nTrue Label: {row['label']}", fontsize=11, fontweight="bold")
        axes[idx, 0].axis("off")

        axes[idx, 1].imshow(cam_overlay)
        axes[idx, 1].set_title(f"Grad-CAM Heatmap\nPred: {pred_label} ({conf*100:.1f}%)", fontsize=11, fontweight="bold", color="darkred")
        axes[idx, 1].axis("off")

    plt.tight_layout()
    gallery_path = output_dir / f"gradcam_gallery_{backbone}.png"
    plt.savefig(gallery_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"[SUCCESS] Exported Grad-CAM visualization gallery to {gallery_path}")

if __name__ == "__main__":
    generate_gradcam_gallery()
