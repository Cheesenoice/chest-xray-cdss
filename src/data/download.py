import os
import sys
import shutil
import zipfile
import yaml
from pathlib import Path

def download_datasets(config_path="configs/default.yaml"):
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    
    raw_dir = Path(cfg["data"]["raw_dir"])
    raw_dir.mkdir(parents=True, exist_ok=True)
    
    # Kaggle API token setting from env if available
    token = os.environ.get("KAGGLE_API_TOKEN")
    if token and not os.environ.get("KAGGLE_KEY"):
        # Set KAGGLE_API_TOKEN
        os.environ["KAGGLE_API_TOKEN"] = token

    datasets = {
        "kermany": "paultimothymooney/chest-xray-pneumonia",
        "pulmonary_abnormalities": "kmader/pulmonary-chest-xray-abnormalities"
    }

    for name, slug in datasets.items():
        dest = raw_dir / name
        if dest.exists() and any(dest.iterdir()):
            print(f"[INFO] Dataset '{name}' already exists at {dest}. Skipping.")
            continue
        
        print(f"[INFO] Downloading dataset '{name}' ({slug})...")
        dest.mkdir(parents=True, exist_ok=True)
        
        # Try kaggle CLI
        cmd = f'kaggle datasets download -d {slug} -p "{dest}" --unzip'
        ret = os.system(cmd)
        if ret != 0:
            print(f"[WARN] Kaggle CLI failed for {slug}. Trying kagglehub...")
            try:
                import kagglehub
                path = kagglehub.dataset_download(slug)
                print(f"[INFO] Downloaded to kagglehub cache: {path}")
                # Copy or symlink files to raw_dir / name
                for item in Path(path).glob("*"):
                    target = dest / item.name
                    if item.is_dir():
                        shutil.copytree(item, target, dirs_exist_ok=True)
                    else:
                        shutil.copy2(item, target)
            except Exception as e:
                print(f"[ERROR] Failed to download {slug}: {e}")
                sys.exit(1)
        
        print(f"[SUCCESS] Downloaded '{name}' to {dest}")

if __name__ == "__main__":
    download_datasets()
