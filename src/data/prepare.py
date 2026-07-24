import os
import re
import hashlib
import pandas as pd
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import yaml

def get_md5(filepath):
    hash_md5 = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def parse_kermany(raw_dir):
    records = []
    kermany_dir = raw_dir / "kermany"
    if not kermany_dir.exists():
        print(f"[WARN] Kermany directory {kermany_dir} does not exist.")
        return records

    image_paths = [
        p for p in list(kermany_dir.rglob("*.jpeg")) + list(kermany_dir.rglob("*.png")) + list(kermany_dir.rglob("*.jpg"))
        if "__MACOSX" not in str(p) and not p.name.startswith("._")
    ]
    print(f"[INFO] Parsing {len(image_paths)} Kermany images...")

    normal_idx = 0
    for p in tqdm(image_paths, desc="Kermany"):
        filename = p.name
        parent_dir = p.parent.name.upper()

        if "NORMAL" in parent_dir or filename.startswith("IM-") or "NORMAL" in filename:
            label = "normal"
            # Kermany normal doesn't have explicit patient IDs, use hash of stem or unique index
            patient_id = f"kermany_norm_{p.stem}"
        elif "PNEUMONIA" in parent_dir or "bacteria" in filename or "virus" in filename:
            if "bacteria" in filename.lower():
                label = "bacterial_pneumonia"
            elif "virus" in filename.lower():
                label = "viral_pneumonia"
            else:
                label = "bacterial_pneumonia" # fallback

            # Extract patient ID: person{N}
            match = re.search(r"person(\d+)", filename, re.IGNORECASE)
            if match:
                patient_id = f"person{match.group(1)}"
            else:
                patient_id = f"kermany_pneu_{p.stem}"
        else:
            continue

        records.append({
            "filepath": str(p.resolve()),
            "source": "kermany",
            "label": label,
            "patient_id": patient_id,
            "domain": "pediatric"
        })

    return records

def parse_pulmonary(raw_dir):
    records = []
    pulm_dir = raw_dir / "pulmonary_abnormalities"
    if not pulm_dir.exists():
        print(f"[WARN] Pulmonary abnormalities directory {pulm_dir} does not exist.")
        return records

    image_paths = [
        p for p in list(pulm_dir.rglob("*.png")) + list(pulm_dir.rglob("*.jpg")) + list(pulm_dir.rglob("*.jpeg"))
        if "__MACOSX" not in str(p) and not p.name.startswith("._")
    ]
    print(f"[INFO] Parsing {len(image_paths)} Pulmonary Abnormalities images...")

    for p in tqdm(image_paths, desc="Pulmonary"):
        filename = p.name
        # Shenzhen pattern: CHNCXR_xxxx_0.png or CHNCXR_xxxx_1.png
        if filename.startswith("CHNCXR"):
            parts = p.stem.split("_")
            if len(parts) >= 3:
                pid = f"CHN_{parts[1]}"
                lbl_code = parts[2]
                label = "tuberculosis" if lbl_code == "1" else "normal"
                records.append({
                    "filepath": str(p.resolve()),
                    "source": "shenzhen",
                    "label": label,
                    "patient_id": pid,
                    "domain": "adult"
                })
        # Montgomery pattern: MCUCXR_xxxx_0.png or MCUCXR_xxxx_1.png
        elif filename.startswith("MCUCXR"):
            parts = p.stem.split("_")
            if len(parts) >= 3:
                pid = f"MCU_{parts[1]}"
                lbl_code = parts[2]
                label = "tuberculosis" if lbl_code == "1" else "normal"
                records.append({
                    "filepath": str(p.resolve()),
                    "source": "montgomery",
                    "label": label,
                    "patient_id": pid,
                    "domain": "adult"
                })

    return records

def prepare_manifest(config_path="configs/default.yaml"):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    raw_dir = Path(cfg["data"]["raw_dir"])
    processed_dir = Path(cfg["data"]["processed_dir"])
    processed_dir.mkdir(parents=True, exist_ok=True)

    records = []
    records.extend(parse_kermany(raw_dir))
    records.extend(parse_pulmonary(raw_dir))

    if not records:
        print("[ERROR] No images found! Ensure datasets are downloaded to data/raw/")
        return

    df = pd.DataFrame(records)
    print(f"[INFO] Total parsed images: {len(df)}")

    # Add metadata: width, height, md5
    widths, heights, md5s = [], [], []
    valid_mask = []

    print("[INFO] Computing MD5 hashes and checking image integrity...")
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Metadata"):
        try:
            with Image.open(row["filepath"]) as img:
                w, h = img.size
                widths.append(w)
                heights.append(h)
                md5s.append(get_md5(row["filepath"]))
                valid_mask.append(True)
        except Exception as e:
            print(f"[WARN] Corrupt image {row['filepath']}: {e}")
            widths.append(None)
            heights.append(None)
            md5s.append(None)
            valid_mask.append(False)

    df = df[valid_mask].copy()
    df["width"] = widths
    df["height"] = heights
    df["md5"] = md5s

    # Deduplication
    if cfg["data"].get("dedup", True):
        initial_len = len(df)
        df = df.drop_duplicates(subset=["md5"], keep="first").copy()
        print(f"[INFO] Deduplication: removed {initial_len - len(df)} duplicate images (by MD5).")

    manifest_path = processed_dir / "manifest.csv"
    df.to_csv(manifest_path, index=False)
    print(f"[SUCCESS] Saved manifest to {manifest_path} ({len(df)} records).")

    print("\n--- Manifest Summary ---")
    print(df.groupby(["source", "label"]).size())

if __name__ == "__main__":
    prepare_manifest()
