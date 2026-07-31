import os
import sys
import yaml
import torch
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    roc_auc_score, roc_curve, confusion_matrix,
    balanced_accuracy_score
)

from src.utils import set_seed, compute_metrics
from src.datasets import build_dataloaders
from src.models import build_model


def bootstrap_metric(y_true, y_pred, y_prob, metric_fn, n_bootstrap=1000, ci=95):
    rng = np.random.RandomState(42)
    indices = np.arange(len(y_true))
    scores = []
    for _ in range(n_bootstrap):
        idx = rng.choice(indices, size=len(indices), replace=True)
        scores.append(metric_fn(y_true[idx], y_pred[idx], y_prob[idx]))
    alpha = (100 - ci) / 2
    lower = np.percentile(scores, alpha)
    upper = np.percentile(scores, 100 - alpha)
    return lower, upper


def plot_confusion_matrix(y_true, y_pred, class_names, save_path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[INFO] Confusion matrix saved to {save_path}")


def plot_roc_curve(y_true, y_prob, class_names, save_path):
    n_classes = len(class_names)
    plt.figure(figsize=(8, 6))
    for i in range(n_classes):
        y_bin = (np.array(y_true) == i).astype(int)
        fpr, tpr, _ = roc_curve(y_bin, y_prob[:, i])
        auc = roc_auc_score(y_bin, y_prob[:, i])
        plt.plot(fpr, tpr, label=f"{class_names[i]} (AUC={auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.5)
    plt.xlim([0, 1])
    plt.ylim([0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves (One-vs-Rest)")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[INFO] ROC curve saved to {save_path}")


@torch.no_grad()
def evaluate_model(model, loader, device, max_batches=None):
    model.eval()
    all_preds, all_targets, all_probs = [], [], []
    for i, (images, targets, _) in enumerate(loader):
        if max_batches and i >= max_batches:
            break
        images = images.to(device)
        outputs = model(images)
        probs = torch.softmax(outputs, dim=1).cpu().numpy()
        preds = np.argmax(probs, axis=1)
        all_preds.extend(preds)
        all_targets.extend(targets.numpy())
        all_probs.extend(probs)
    return np.array(all_targets), np.array(all_preds), np.array(all_probs)


def run_evaluation(config_path="configs/default.yaml", checkpoint_path=None, dry_run=False):
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    seed = cfg.get("seed", 42)
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Evaluating on device: {device}")

    num_classes = cfg["data"].get("num_classes", 4)
    splits_dir = cfg["data"]["splits_dir"]

    train_loader, val_loader, test_loader, ext_loader, label_map = build_dataloaders(
        csv_dir=splits_dir,
        image_size=cfg["data"].get("image_size", 224),
        batch_size=cfg["train"]["batch_size"],
        num_workers=cfg["train"].get("num_workers", 2),
        num_classes=num_classes,
        imbalance_strategy="none",
        aug_cfg=None
    )

    model = build_model(
        backbone_name=cfg["model"]["backbone"],
        num_classes=num_classes,
        pretrained=False,
        drop_rate=0
    ).to(device)

    if checkpoint_path is None:
        checkpoint_path = Path("results/checkpoints") / f"best_{cfg['model']['backbone']}_seed{seed}_cls{num_classes}.pt"

    if not Path(checkpoint_path).exists():
        print(f"[ERROR] Checkpoint not found at {checkpoint_path}")
        return

    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"[INFO] Loaded checkpoint from epoch {ckpt.get('epoch', '?')}")

    class_names = [k for k, v in sorted(label_map.items(), key=lambda x: x[1])]
    print(f"[INFO] Class names: {class_names}")

    max_b = 5 if dry_run else None
    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)

    for split_name, loader in [("Test", test_loader), ("External", ext_loader)]:
        if loader is None:
            continue
        y_true, y_pred, y_prob = evaluate_model(model, loader, device, max_batches=max_b)
        metrics = compute_metrics(y_true, y_pred, y_prob, num_classes=num_classes)
        acc = metrics["accuracy"]

        print(f"\n--- {split_name} Set Results ---")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")

        ci_low, ci_high = bootstrap_metric(
            y_true, y_pred, y_prob,
            lambda yt, yp, _: accuracy_score(yt, yp),
            n_bootstrap=100
        )
        print(f"  accuracy_95ci: ({ci_low:.4f}, {ci_high:.4f})")

        # Save figures
        plot_confusion_matrix(y_true, y_pred, class_names,
                              results_dir / f"cm_{split_name.lower()}.png")
        if y_prob is not None:
            plot_roc_curve(y_true, y_prob, class_names,
                           results_dir / f"roc_{split_name.lower()}.png")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    run_evaluation(args.config, args.checkpoint, dry_run=args.dry_run)
