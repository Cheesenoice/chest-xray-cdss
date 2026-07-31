import yaml
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import GroupShuffleSplit

def create_splits(config_path="configs/default.yaml"):
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    processed_dir = Path(cfg["data"]["processed_dir"])
    splits_dir = Path(cfg["data"]["splits_dir"])
    splits_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = processed_dir / "manifest.csv"
    if not manifest_path.exists():
        print(f"[ERROR] Manifest file {manifest_path} not found. Run prepare.py first.")
        return

    df = pd.read_csv(manifest_path)
    seed = cfg.get("seed", 42)
    num_classes = cfg["data"].get("num_classes", 4)
    external_sources = cfg["data"].get("external_sources", ["montgomery"])

    # Define active classes
    if num_classes == 3:
        active_labels = ["normal", "bacterial_pneumonia", "viral_pneumonia"]
    else:
        active_labels = ["normal", "bacterial_pneumonia", "viral_pneumonia", "tuberculosis"]

    # Separate external test set (held-out sources)
    ext_df = df[df["source"].isin(external_sources) & df["label"].isin(active_labels)].copy()
    internal_df = df[(~df["source"].isin(external_sources)) & df["label"].isin(active_labels)].copy()

    print(f"[INFO] Total internal dataset size: {len(internal_df)} images across {internal_df['patient_id'].nunique()} patient groups.")
    print(f"[INFO] External test dataset size: {len(ext_df)} images ({external_sources}).")

    # Patient-level split: GroupShuffleSplit to split Train / (Val + Test)
    val_ratio = cfg["data"]["splits"]["val"]
    test_ratio = cfg["data"]["splits"]["test"]
    val_test_ratio = val_ratio + test_ratio

    gss1 = GroupShuffleSplit(n_splits=1, test_size=val_test_ratio, random_state=seed)
    train_idx, val_test_idx = next(gss1.split(internal_df, groups=internal_df["patient_id"]))

    train_df = internal_df.iloc[train_idx].copy()
    val_test_df = internal_df.iloc[val_test_idx].copy()

    # Split (Val + Test) into Val and Test
    relative_test_ratio = test_ratio / val_test_ratio
    gss2 = GroupShuffleSplit(n_splits=1, test_size=relative_test_ratio, random_state=seed)
    val_idx, test_idx = next(gss2.split(val_test_df, groups=val_test_df["patient_id"]))

    val_df = val_test_df.iloc[val_idx].copy()
    test_df = val_test_df.iloc[test_idx].copy()

    # Strict Leakage Checks (ASSERT)
    train_patients = set(train_df["patient_id"])
    val_patients = set(val_df["patient_id"])
    test_patients = set(test_df["patient_id"])
    ext_patients = set(ext_df["patient_id"])

    train_val_intersect = train_patients.intersection(val_patients)
    train_test_intersect = train_patients.intersection(test_patients)
    val_test_intersect = val_patients.intersection(test_patients)
    ext_internal_intersect = ext_patients.intersection(train_patients | val_patients | test_patients)

    assert len(train_val_intersect) == 0, f"DATA LEAKAGE: Patient overlap train-val: {train_val_intersect}"
    assert len(train_test_intersect) == 0, f"DATA LEAKAGE: Patient overlap train-test: {train_test_intersect}"
    assert len(val_test_intersect) == 0, f"DATA LEAKAGE: Patient overlap val-test: {val_test_intersect}"
    assert len(ext_internal_intersect) == 0, f"DATA LEAKAGE: Patient overlap internal-external: {ext_internal_intersect}"

    # Strict MD5 Hash Non-Overlap Check
    train_md5 = set(train_df["md5"])
    val_md5 = set(val_df["md5"])
    test_md5 = set(test_df["md5"])
    ext_md5 = set(ext_df["md5"])

    assert len(train_md5.intersection(val_md5)) == 0, "DATA LEAKAGE: MD5 overlap train-val"
    assert len(train_md5.intersection(test_md5)) == 0, "DATA LEAKAGE: MD5 overlap train-test"
    assert len(val_md5.intersection(test_md5)) == 0, "DATA LEAKAGE: MD5 overlap val-test"
    assert len(train_md5.intersection(ext_md5)) == 0, "DATA LEAKAGE: MD5 overlap train-external"

    print("[SUCCESS] All Data Leakage Assertions Passed Cleanly!")

    # Save CSVs
    train_df.to_csv(splits_dir / "train.csv", index=False)
    val_df.to_csv(splits_dir / "val.csv", index=False)
    test_df.to_csv(splits_dir / "test.csv", index=False)
    if len(ext_df) > 0:
        ext_df.to_csv(splits_dir / "external_test.csv", index=False)
    elif (splits_dir / "external_test.csv").exists():
        (splits_dir / "external_test.csv").unlink()

    print(f"[INFO] Splits saved to {splits_dir}/")

    # Generate Markdown Report for paper/documentation
    report_path = splits_dir / "split_report.md"
    summary_table = []
    splits_dict = {"Train": train_df, "Validation": val_df, "Internal Test": test_df, "External Test": ext_df}

    report_md = f"# Dataset Split Summary Report\n\n"
    report_md += f"- **Num Classes:** {num_classes}\n"
    report_md += f"- **Random Seed:** {seed}\n"
    report_md += f"- **Patient Leakage:** ZERO (verified via GroupShuffleSplit & Assertions)\n"
    report_md += f"- **MD5 Hash Leakage:** ZERO (verified via MD5 set intersections)\n\n"

    report_md += "## Image & Patient Distribution\n\n"
    report_md += "| Split | " + " | ".join(active_labels) + " | Total Images | Unique Patients |\n"
    report_md += "|---|" + "|".join(["---"] * (len(active_labels) + 2)) + "|\n"

    for s_name, s_df in splits_dict.items():
        counts = s_df["label"].value_counts().to_dict()
        row_str = f"| **{s_name}** | "
        row_str += " | ".join([str(counts.get(lbl, 0)) for lbl in active_labels])
        row_str += f" | **{len(s_df)}** | {s_df['patient_id'].nunique()} |\n"
        report_md += row_str

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_md)

    print(f"[SUCCESS] Wrote split report to {report_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    create_splits(args.config)
