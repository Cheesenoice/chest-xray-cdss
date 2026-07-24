import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image

LABEL_MAP_4CLASS = {
    "normal": 0,
    "bacterial_pneumonia": 1,
    "viral_pneumonia": 2,
    "tuberculosis": 3
}

LABEL_MAP_3CLASS = {
    "normal": 0,
    "bacterial_pneumonia": 1,
    "viral_pneumonia": 2
}

def get_transforms(image_size=224, is_train=True, aug_cfg=None):
    if is_train and aug_cfg:
        transforms_list = [
            A.Resize(image_size, image_size),
        ]
        if aug_cfg.get("horizontal_flip", True):
            transforms_list.append(A.HorizontalFlip(p=0.5))
        if aug_cfg.get("rotate_deg", 0) > 0:
            transforms_list.append(A.Rotate(limit=aug_cfg["rotate_deg"], p=0.5))
        if aug_cfg.get("brightness_contrast", 0) > 0:
            bc = aug_cfg["brightness_contrast"]
            transforms_list.append(A.RandomBrightnessContrast(brightness_limit=bc, contrast_limit=bc, p=0.5))
        
        transforms_list.extend([
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
        return A.Compose(transforms_list)
    else:
        return A.Compose([
            A.Resize(image_size, image_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])

class ChestXRayDataset(Dataset):
    def __init__(self, df, label_map=LABEL_MAP_4CLASS, transform=None):
        self.df = df.reset_index(drop=True)
        self.label_map = label_map
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row["filepath"]

        # Read image using OpenCV (BGR) -> RGB
        img = cv2.imread(img_path)
        if img is None:
            # Fallback to PIL
            img = np.array(Image.open(img_path).convert("RGB"))
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transform:
            augmented = self.transform(image=img)
            img_tensor = augmented["image"]
        else:
            img_tensor = torch.tensor(img).permute(2, 0, 1).float() / 255.0

        label_idx = self.label_map[row["label"]]
        return img_tensor, label_idx, row["filepath"]

def build_dataloaders(csv_dir, image_size=224, batch_size=32, num_workers=2, num_classes=4, imbalance_strategy="weighted_sampler", aug_cfg=None):
    label_map = LABEL_MAP_4CLASS if num_classes == 4 else LABEL_MAP_3CLASS
    
    train_df = pd.read_csv(os.path.join(csv_dir, "train.csv"))
    val_df = pd.read_csv(os.path.join(csv_dir, "val.csv"))
    test_df = pd.read_csv(os.path.join(csv_dir, "test.csv"))
    ext_df = pd.read_csv(os.path.join(csv_dir, "external_test.csv")) if os.path.exists(os.path.join(csv_dir, "external_test.csv")) else None

    train_ds = ChestXRayDataset(train_df, label_map, get_transforms(image_size, is_train=True, aug_cfg=aug_cfg))
    val_ds = ChestXRayDataset(val_df, label_map, get_transforms(image_size, is_train=False))
    test_ds = ChestXRayDataset(test_df, label_map, get_transforms(image_size, is_train=False))
    ext_ds = ChestXRayDataset(ext_df, label_map, get_transforms(image_size, is_train=False)) if ext_df is not None else None

    # Handle class imbalance for training sampler
    sampler = None
    if imbalance_strategy == "weighted_sampler":
        targets = [label_map[lbl] for lbl in train_df["label"]]
        class_counts = pd.Series(targets).value_counts().sort_index().values
        class_weights = 1.0 / class_counts
        sample_weights = torch.tensor([class_weights[t] for t in targets], dtype=torch.float)
        sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=(sampler is None),
        num_workers=num_workers,
        pin_memory=True
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    ext_loader = DataLoader(ext_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True) if ext_ds is not None else None

    return train_loader, val_loader, test_loader, ext_loader, label_map
