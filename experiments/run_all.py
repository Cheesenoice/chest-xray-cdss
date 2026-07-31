import os
import sys
import subprocess
import yaml
import pandas as pd
from pathlib import Path
from datetime import datetime


def run_experiment(config_path, backbone, seed, dry_run=False):
    cmd = [
        sys.executable, "-m", "src.train",
        "--config", str(config_path),
        "--backbone", backbone,
        "--seed", str(seed)
    ]
    if dry_run:
        cmd.append("--dry-run")
    print(f"\n{'='*60}")
    print(f"[EXPERIMENT] backbone={backbone}, seed={seed}, dry_run={dry_run}")
    print(f"{'='*60}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print(result.stderr[-2000:])
    return result.returncode == 0


def run_all(config_path="configs/default.yaml", dry_run=False):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    backbones = cfg.get("backbones", ["resnet18", "densenet121", "efficientnet_b0"])
    seeds = cfg.get("seeds", [42, 7, 123])

    results_log = []
    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for backbone in backbones:
        successes = 0
        for seed in seeds:
            ok = run_experiment(config_path, backbone, seed, dry_run=dry_run)
            results_log.append({
                "backbone": backbone,
                "seed": seed,
                "success": ok
            })
            if ok:
                successes += 1
        print(f"[SUMMARY] {backbone}: {successes}/{len(seeds)} seeds succeeded")

    log_path = results_dir / f"benchmark_log_{timestamp}.csv"
    log_df = pd.DataFrame(results_log)
    log_df.to_csv(log_path, index=False)
    print(f"\n[SUCCESS] Benchmark log saved to {log_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    run_all(args.config, dry_run=args.dry_run)
