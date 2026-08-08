# Chest X-ray CDSS — Explainable Deep Learning Clinical Decision Support System

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

A **reproducible benchmark** comparing multiple deep learning backbones (ResNet-18, DenseNet-121, EfficientNet-B0) for chest X-ray multi-class classification, with Grad-CAM explainability, external validation on held-out sources, and a classical ML baseline (HOG/LBP + SVM).

## Table of Contents

- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Data Pipeline](#data-pipeline)
- [Training & Evaluation](#training--evaluation)
- [Results](#results)
- [Web Demo](#web-demo)
- [Paper / Citation](#paper--citation)
- [License & Attribution](#license--attribution)

## Quick Start

### Option A: Run Interactive Web Demo Immediately (Pre-trained Weights Included)
```bash
# 1. Clone repo
git clone https://github.com/Cheesenoice/chest-xray-cdss.git
cd chest-xray-cdss

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch Web App Demo (Pre-trained weights included!)
streamlit run app/app.py
```

### Option B: Download Datasets & Reproduce Experiments from Scratch
```bash
# 1. Download datasets (Automated via Kaggle API / kagglehub)
python -m src.data.download

# 2. Prepare manifest & execute zero-leakage patient-level split
python -m src.data.prepare
python -m src.data.split
python -m src.data.checks

# 3. Run Scientific Audit Verification (100% Zero Leakage Test)
python -m src.audit_pipeline

# 4. Run Classical ML Baseline (HOG + SVM)
python -m src.baseline_classical

# 5. Run Deep Learning Multi-Backbone Benchmark (ResNet18, DenseNet121, EfficientNetB0 x 3 seeds)
python -m experiments.run_all --config configs/default.yaml --epochs 15
```

## Project Structure

```
├── configs/          # YAML configuration (hyperparams, data paths, seeds)
├── src/
│   ├── data/         # Data pipeline: download → prepare → split → checks
│   ├── datasets.py   # PyTorch Dataset + augmentation (albumentations)
│   ├── models.py     # Backbone factory (timm)
│   ├── train.py      # Training loop (AMP, early stopping, checkpointing)
│   ├── evaluate.py   # Metrics, bootstrap CI, confusion matrix, ROC curves
│   ├── explain.py    # Grad-CAM heatmap overlay
│   ├── baseline_classical.py  # HOG/LBP + SVM/LogReg baseline
│   └── utils.py      # Seed management, metric computation
├── experiments/      # Multi-backbone × multi-seed runner
├── app/              # Streamlit web demo + PDF report generation
├── notebooks/        # EDA notebooks
├── results/          # Metric tables, confusion matrices, ROC plots, logs
├── paper/            # IMRaD outline + figures
└── model_card.md     # Model card with intended use and limitations
```

## Data Pipeline

Leakage-free patient-level split with strict assertions:

1. **download.py** — Download Kermany + Shenzhen/Montgomery via Kaggle API
2. **prepare.py** — Parse filenames → labels + patient IDs, compute MD5, deduplicate
3. **split.py** — `GroupShuffleSplit` at patient level, hold out Montgomery → external test
4. **checks.py** — Verify MD5 overlap, dimension stats, domain confound report

## Training & Evaluation

- **Backbones:** ResNet-18, DenseNet-121, EfficientNet-B0 (timm, ImageNet pretrained)
- **Optimizer:** AdamW (lr=1e-4), cosine scheduler, early stopping
- **AMP:** Mixed-precision training for speed
- **Class imbalance:** WeightedRandomSampler
- **Metrics:** Accuracy, Precision, Recall, F1 (macro), AUC (macro), per-class metrics
- **Reproducibility:** Mean ± std over 3 seeds + 95% CI bootstrap

## Web Demo

```bash
streamlit run app/app.py
```

Upload a chest X-ray → diagnosis probability → Grad-CAM heatmap → triage alert.

## Paper / Citation

See `paper/outline.md` for IMRaD structure. Preprint available at [arXiv/medRxiv link TBD].

## License & Attribution

- Code: MIT License
- Kermany dataset: CC BY 4.0 ([Kermany et al. 2018](https://doi.org/10.1016/j.cell.2018.02.010))
- Shenzhen/Montgomery: Public domain (NLM/NIH) — [Jaeger et al. 2014](https://pubmed.ncbi.nlm.nih.gov/24142925/), [Candemir et al. 2014](https://pubmed.ncbi.nlm.nih.gov/24142882/)
- TB dataset: Rahman et al. — [Reliable Tuberculosis Detection](https://www.kaggle.com/datasets/tawsifurrahman/tuberculosis-tb-chest-xray-dataset)

> **Disclaimer:** This system is for clinical decision support and research only. It does NOT replace a physician's diagnosis.
