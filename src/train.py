import os
import sys
import argparse
import yaml
import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import numpy as np

from src.utils import set_seed, compute_metrics
from src.datasets import build_dataloaders
from src.models import build_model

def train_epoch(model, loader, criterion, optimizer, scaler, device, use_amp=True, max_batches=None):
    model.train()
    running_loss = 0.0
    all_preds, all_targets, all_probs = [], [], []

    for i, (images, targets, _) in enumerate(tqdm(loader, desc="Train Iter", leave=False)):
        if max_batches and i >= max_batches:
            break
        images, targets = images.to(device), targets.to(device)
        optimizer.zero_grad()

        if use_amp and device.type == "cuda":
            with autocast(device.type):
                outputs = model(images)
                loss = criterion(outputs, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

        running_loss += loss.item() * images.size(0)
        probs = torch.softmax(outputs, dim=1).detach().cpu().numpy()
        preds = np.argmax(probs, axis=1)

        all_preds.extend(preds)
        all_targets.extend(targets.cpu().numpy())
        all_probs.extend(probs)

    epoch_loss = running_loss / len(loader.dataset)
    metrics = compute_metrics(np.array(all_targets), np.array(all_preds), np.array(all_probs), num_classes=outputs.shape[1])
    metrics["loss"] = epoch_loss
    return metrics

@torch.no_grad()
def evaluate_model(model, loader, criterion, device, max_batches=None):
    model.eval()
    running_loss = 0.0
    all_preds, all_targets, all_probs = [], [], []

    for i, (images, targets, _) in enumerate(tqdm(loader, desc="Eval Iter", leave=False)):
        if max_batches and i >= max_batches:
            break
        images, targets = images.to(device), targets.to(device)
        outputs = model(images)
        loss = criterion(outputs, targets)

        running_loss += loss.item() * images.size(0)
        probs = torch.softmax(outputs, dim=1).cpu().numpy()
        preds = np.argmax(probs, axis=1)

        all_preds.extend(preds)
        all_targets.extend(targets.cpu().numpy())
        all_probs.extend(probs)

    epoch_loss = running_loss / len(loader.dataset)
    metrics = compute_metrics(np.array(all_targets), np.array(all_preds), np.array(all_probs), num_classes=outputs.shape[1])
    metrics["loss"] = epoch_loss
    return metrics

def run_training(config_path="configs/default.yaml", dry_run=False, backbone_override=None, seed_override=None):
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if backbone_override is not None:
        cfg["model"]["backbone"] = backbone_override
    seed = seed_override if seed_override is not None else cfg.get("seed", 42)
    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Training on device: {device}")

    num_classes = cfg["data"].get("num_classes", 4)
    splits_dir = cfg["data"]["splits_dir"]

    train_loader, val_loader, test_loader, ext_loader, label_map = build_dataloaders(
        csv_dir=splits_dir,
        image_size=cfg["data"].get("image_size", 224),
        batch_size=cfg["train"]["batch_size"] if not dry_run else 4,
        num_workers=cfg["train"].get("num_workers", 2),
        num_classes=num_classes,
        imbalance_strategy=cfg["data"].get("imbalance_strategy", "weighted_sampler"),
        aug_cfg=cfg["data"].get("augmentation", {})
    )

    model = build_model(
        backbone_name=cfg["model"]["backbone"],
        num_classes=num_classes,
        pretrained=cfg["model"].get("pretrained", True),
        drop_rate=cfg["model"].get("drop_rate", 0.2)
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg["train"]["lr"]), weight_decay=float(cfg["train"].get("weight_decay", 0.01)))
    scaler = GradScaler(device.type, enabled=cfg["train"].get("use_amp", True) and device.type == "cuda")

    epochs = cfg["train"]["epochs"] if not dry_run else 1
    best_f1 = 0.0
    patience = cfg["train"].get("early_stopping_patience", 5)
    patience_counter = 0

    save_dir = Path("results/checkpoints")
    save_dir.mkdir(parents=True, exist_ok=True)
    # Tên checkpoint phân biệt theo backbone, seed và số lớp (3/4 class)
    # để các benchmark chạy song song không ghi đè lẫn nhau.
    best_ckpt_path = save_dir / f"best_{cfg['model']['backbone']}_seed{seed}_cls{num_classes}.pt"

    max_b = 5 if dry_run else None
    print(f"[INFO] Starting training for {epochs} epochs...")
    for epoch in range(1, epochs + 1):
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, scaler, device, use_amp=cfg["train"].get("use_amp", True), max_batches=max_b)
        val_metrics = evaluate_model(model, val_loader, criterion, device, max_batches=max_b)

        print(f"Epoch [{epoch:02d}/{epochs:02d}] | Train Loss: {train_metrics['loss']:.4f} | Train F1: {train_metrics['f1_macro']:.4f} | Val Loss: {val_metrics['loss']:.4f} | Val F1: {val_metrics['f1_macro']:.4f} | Val AUC: {val_metrics['auc_macro']:.4f}")

        if val_metrics["f1_macro"] > best_f1:
            best_f1 = val_metrics["f1_macro"]
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_metrics": val_metrics,
                "config": cfg
            }, best_ckpt_path)
            print(f"  --> Saved new best checkpoint to {best_ckpt_path}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"[INFO] Early stopping triggered at epoch {epoch}")
                break

    # Final evaluation on Internal Test set
    if best_ckpt_path.exists():
        checkpoint = torch.load(best_ckpt_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"\n[INFO] Loaded best model checkpoint from epoch {checkpoint['epoch']} for final testing.")

    test_metrics = evaluate_model(model, test_loader, criterion, device, max_batches=max_b)
    print("\n--- Internal Test Results ---")
    for k, v in test_metrics.items():
        print(f"  {k}: {v:.4f}")

    if ext_loader is not None:
        ext_metrics = evaluate_model(model, ext_loader, criterion, device, max_batches=max_b)
        print("\n--- External Test Results (Montgomery Held-Out Source) ---")
        for k, v in ext_metrics.items():
            print(f"  {k}: {v:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--dry-run", action="store_true", help="Run 1 epoch with small batch for testing")
    parser.add_argument("--backbone", default=None, help="Override backbone name")
    parser.add_argument("--seed", type=int, default=None, help="Override random seed")
    args = parser.parse_args()
    run_training(args.config, dry_run=args.dry_run, backbone_override=args.backbone, seed_override=args.seed)
