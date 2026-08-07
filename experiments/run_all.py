import os
import sys
import yaml
import copy
import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from tqdm import tqdm

from src.utils import set_seed, compute_metrics
from src.datasets import build_dataloaders
from src.models import build_model
from src.train import train_epoch, evaluate_model

def run_experiment(backbone, seed, cfg, device, epochs=15):
    set_seed(seed)
    num_classes = cfg["data"].get("num_classes", 4)
    splits_dir = cfg["data"]["splits_dir"]

    train_loader, val_loader, test_loader, ext_loader, label_map = build_dataloaders(
        csv_dir=splits_dir,
        image_size=cfg["data"].get("image_size", 224),
        batch_size=cfg["train"]["batch_size"],
        num_workers=cfg["train"].get("num_workers", 2),
        num_classes=num_classes,
        imbalance_strategy=cfg["data"].get("imbalance_strategy", "weighted_sampler"),
        aug_cfg=cfg["data"].get("augmentation", {})
    )

    model = build_model(
        backbone_name=backbone,
        num_classes=num_classes,
        pretrained=cfg["model"].get("pretrained", True),
        drop_rate=cfg["model"].get("drop_rate", 0.2)
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg["train"]["lr"]), weight_decay=float(cfg["train"].get("weight_decay", 0.01)))
    scaler = torch.amp.GradScaler('cuda', enabled=cfg["train"].get("use_amp", True) and device.type == "cuda")

    best_f1 = 0.0
    patience = cfg["train"].get("early_stopping_patience", 5)
    patience_counter = 0

    save_dir = Path("results/checkpoints")
    save_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt_path = save_dir / f"best_{backbone}_seed{seed}.pt"

    for epoch in range(1, epochs + 1):
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, scaler, device, use_amp=cfg["train"].get("use_amp", True))
        val_metrics = evaluate_model(model, val_loader, criterion, device)

        if val_metrics["f1_macro"] > best_f1:
            best_f1 = val_metrics["f1_macro"]
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_metrics": val_metrics
            }, best_ckpt_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    # Load best checkpoint for test evaluation
    if best_ckpt_path.exists():
        checkpoint = torch.load(best_ckpt_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])

    test_metrics = evaluate_model(model, test_loader, criterion, device)
    ext_metrics = evaluate_model(model, ext_loader, criterion, device) if ext_loader is not None else None

    return test_metrics, ext_metrics

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--backbones", nargs="+", default=["resnet18", "densenet121", "efficientnet_b0"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 7, 123])
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Multi-backbone experiment runner starting on device: {device}")

    all_results = []
    for backbone in args.backbones:
        print(f"\n========================================================")
        print(f"  BENCHMARKING BACKBONE: {backbone.upper()}")
        print(f"========================================================")
        
        seed_test_metrics = []
        seed_ext_metrics = []

        for seed in args.seeds:
            print(f"\n[RUN] Backbone: {backbone} | Seed: {seed}")
            t_met, e_met = run_experiment(backbone, seed, cfg, device, epochs=args.epochs)
            seed_test_metrics.append(t_met)
            if e_met:
                seed_ext_metrics.append(e_met)
            print(f"  Result (Internal Test): Acc={t_met['accuracy']:.4f}, F1={t_met['f1_macro']:.4f}, AUC={t_met['auc_macro']:.4f}")

        # Compute mean +- std across seeds
        def aggregate_metrics(metrics_list):
            agg = {}
            for k in metrics_list[0].keys():
                vals = [m[k] for m in metrics_list]
                agg[f"{k}_mean"] = np.mean(vals)
                agg[f"{k}_std"] = np.std(vals)
            return agg

        test_agg = aggregate_metrics(seed_test_metrics)
        ext_agg = aggregate_metrics(seed_ext_metrics) if seed_ext_metrics else None

        all_results.append({
            "backbone": backbone,
            "test_agg": test_agg,
            "ext_agg": ext_agg
        })

    # Build Markdown Summary Report
    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)
    summary_path = results_dir / "benchmark_summary.md"

    md = f"# Multi-Backbone Benchmark Results (Mean ± Std over {len(args.seeds)} Seeds)\n\n"
    md += f"- **Tested Backbones:** {', '.join(args.backbones)}\n"
    md += f"- **Random Seeds:** {args.seeds}\n"
    md += f"- **Training Epochs per Run:** {args.epochs}\n\n"

    md += "## 1. Internal Test Set Performance\n\n"
    md += "| Backbone | Accuracy | Precision (Macro) | Recall (Macro) | F1 Score (Macro) | AUC (Macro) |\n"
    md += "|---|---|---|---|---|---|\n"

    for res in all_results:
        b = res["backbone"]
        t = res["test_agg"]
        md += f"| **{b}** | {t['accuracy_mean']:.4f} ± {t['accuracy_std']:.4f} | {t['precision_macro_mean']:.4f} ± {t['precision_macro_std']:.4f} | {t['recall_macro_mean']:.4f} ± {t['recall_macro_std']:.4f} | **{t['f1_macro_mean']:.4f} ± {t['f1_macro_std']:.4f}** | {t['auc_macro_mean']:.4f} ± {t['auc_macro_std']:.4f} |\n"

    if all_results[0]["ext_agg"]:
        md += "\n## 2. External Test Set Performance (Held-Out Montgomery Source)\n\n"
        md += "| Backbone | Accuracy | Precision (Macro) | Recall (Macro) | F1 Score (Macro) | AUC (Macro) |\n"
        md += "|---|---|---|---|---|---|\n"
        for res in all_results:
            b = res["backbone"]
            e = res["ext_agg"]
            md += f"| **{b}** | {e['accuracy_mean']:.4f} ± {e['accuracy_std']:.4f} | {e['precision_macro_mean']:.4f} ± {e['precision_macro_std']:.4f} | {e['recall_macro_mean']:.4f} ± {e['recall_macro_std']:.4f} | **{e['f1_macro_mean']:.4f} ± {e['f1_macro_std']:.4f}** | {e['auc_macro_mean']:.4f} ± {e['auc_macro_std']:.4f} |\n"

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(md)

    print(f"\n[SUCCESS] Wrote benchmark summary report to {summary_path}")

if __name__ == "__main__":
    main()
