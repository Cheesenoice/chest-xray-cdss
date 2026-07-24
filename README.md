# Explainable Deep Learning Clinical Decision Support System for Chest X-ray Screening

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC_BY_4.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)

> **Graduation Thesis Project (2026)**  
> **Topic (EN):** *An Explainable Deep Learning Clinical Decision Support System for Chest X-ray Screening*  
> **Topic (VN):** *Hệ thống hỗ trợ sàng lọc và quyết định lâm sàng từ ảnh X-quang ngực bằng học sâu có giải thích.*

---

## 📌 Project Overview

This repository implements a **reproducible, explainable, and externally-validated deep learning system** for multi-class chest X-ray screening:
- **Classes Supported:** `Normal`, `Bacterial Pneumonia`, `Viral Pneumonia`, `Tuberculosis (TB)`
- **Explainability:** Grad-CAM saliency heatmaps overlaid on thoracic regions.
- **External Validation:** 100% held-out Montgomery dataset for honest generalization benchmarking.
- **Data Leakage Prevention:** Strict patient-level (`patient_id`) and MD5 hash non-overlap verification across splits.

---

## 📁 Repository Structure

```text
chest-xray-cdss/
├── CLAUDE_EN.md              # Context & master plan document
├── DATA.md                   # Data pipeline specification
├── README.md                 # Project README
├── requirements.txt          # Dependencies
├── configs/
│   └── default.yaml          # Hyperparameters, data paths, random seeds
├── data/
│   ├── raw/                  # Raw downloaded datasets (.gitkeep)
│   └── processed/            # Manifest & patient-level split CSVs
│       └── splits/
├── src/
│   ├── data/
│   │   ├── download.py       # Kaggle dataset downloader
│   │   ├── prepare.py        # Manifest generator & MD5 deduplication
│   │   ├── split.py          # Zero-leakage patient-level splitter
│   │   └── checks.py         # Data quality audit
│   ├── datasets.py           # PyTorch Dataset & DataLoader
│   ├── models.py             # Backbone model factory (timm: DenseNet-121, ResNet-18)
│   ├── train.py              # End-to-end training & evaluation loop
│   └── utils.py              # Seed management & metric computation
├── notebooks/                # Publication-grade EDA Jupyter Notebooks
│   ├── 01_eda_kermany.ipynb
│   ├── 02_eda_pulmonary.ipynb
│   └── 03_eda_consolidated.ipynb
└── results/                  # Split reports and data quality output
```

---

## ⚡ Quick Start

### 1. Setup Virtual Environment
```bash
python -m venv .venv
# PowerShell:
.\.venv\Scripts\Activate.ps1
# Install dependencies:
pip install -r requirements.txt
```

### 2. Download Datasets
```bash
python -m src.data.download
```

### 3. Data Processing & Leakage Prevention Split
```bash
python -m src.data.prepare
python -m src.data.split
python -m src.data.checks
```

### 4. Train Model
```bash
# Dry-run test (5 batches):
python -m src.train --config configs/default.yaml --dry-run

# Full training:
python -m src.train --config configs/default.yaml
```

---

## 📊 Dataset Split Summary (Zero Patient Leakage)

| Split | Normal | Bacterial Pneumonia | Viral Pneumonia | Tuberculosis | Total Images | Unique Patients |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Train** | 1,325 | 1,966 | 1,060 | 235 | **4,586** | 2,740 |
| **Validation** | 270 | 411 | 226 | 59 | **966** | 587 |
| **Internal Test** | 310 | 383 | 199 | 42 | **934** | 588 |
| **External Test (Montgomery)** | 240 | 0 | 0 | 174 | **414** | 138 |

---

## 📜 Citations & Licenses

- **Kermany et al. 2018:** *Identifying Medical Diagnoses and Treatable Diseases by Image-Based Deep Learning*, Cell. (CC BY 4.0)
- **Jaeger et al. 2014 & Candemir et al. 2014:** *Shenzhen & Montgomery Chest X-ray Sets*, NLM/NIH.
