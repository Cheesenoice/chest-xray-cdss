import yaml
import pandas as pd
from pathlib import Path

def run_data_checks(config_path="configs/default.yaml"):
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)
    splits_dir = Path(cfg["data"]["splits_dir"])

    train_path = splits_dir / "train.csv"
    if not train_path.exists():
        print(f"[ERROR] Splits not found in {splits_dir}. Run split.py first.")
        return

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(splits_dir / "val.csv")
    test_df = pd.read_csv(splits_dir / "test.csv")
    ext_df = pd.read_csv(splits_dir / "external_test.csv")

    all_df = pd.concat([train_df, val_df, test_df, ext_df], ignore_index=True)

    report = "# Data Quality & Integrity Verification Report\n\n"
    report += f"Total Images Verified: {len(all_df)}\n"
    report += f"Unique Patients: {all_df['patient_id'].nunique()}\n"
    report += f"Unique Sources: {all_df['source'].unique().tolist()}\n\n"

    report += "## Image Dimension Statistics\n"
    report += f"- Min Resolution: {all_df['width'].min()}x{all_df['height'].min()}\n"
    report += f"- Max Resolution: {all_df['width'].max()}x{all_df['height'].max()}\n"
    report += f"- Median Resolution: {all_df['width'].median():.0f}x{all_df['height'].median():.0f}\n\n"

    report += "## Domain Breakdown (Pediatric vs Adult Confound Check)\n"
    report += "```\n"
    report += str(pd.crosstab(all_df["label"], all_df["domain"]))
    report += "\n```\n\n"

    out_path = results_dir / "data_quality.md"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"[SUCCESS] Data quality report saved to {out_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    run_data_checks(args.config)
