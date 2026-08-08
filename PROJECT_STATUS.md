# Project Status & Progress Tracker

> This file is the **single source of truth** for AI agents and humans working on this repository.
> Read this first before starting any task. Update it after completing any step.

Last updated: **2026-07-31**

---

## Project Overview

**Title:** MedVision AI: An Explainable Deep Learning Clinical Decision Support Platform for Chest X-Ray Screening

**Goal:** Production-grade Clinical Decision Support Platform + Reproducible multi-class benchmark (`Normal`, `Bacterial Pneumonia`, `Viral Pneumonia`, `Tuberculosis`) comparing `ResNet-18`, `DenseNet-121`, `EfficientNet-B0` vs. Classical `HOG+SVM` baseline with:
- Zero-leakage patient-level splits (`GroupShuffleSplit` on `patient_id`)
- 100% Held-Out External Site Validation (Montgomery County, USA)
- Grad-CAM pathological heatmap localization (`denseblock4`)
- Patient Vitals & Clinical Risk Triage Engine (SpO2, Temp, Dyspnea, Priority Alert)
- Automated 1-Click PDF Diagnostic Report Export (`app/report.py`)
- Preliminary AI Radiology Draft Findings Generator
- Interactive Streamlit 4-Tab Web Application (`app/app.py`) + Plotly Analytics
- IMRaD Preprint Paper (`paper/paper_draft.md`) + Model Card (`model_card.md`)

---

## 4-Week Roadmap

| Phase | Milestone | Status | Output / Artifacts |
|---|---|---|---|
| **Phase 1** | Data pipeline + Patient-level split + EDA | ✅ Complete | `manifest.csv`, `split_report.md`, `01_eda_kermany.ipynb`, `02_eda_pulmonary.ipynb` |
| **Phase 2** | Classical ML Baseline + Multi-backbone Benchmark (3 seeds) + Grad-CAM | ✅ Complete | `classical_baseline.md` (Acc: 83.51% int / 10.87% ext), `benchmark_summary.md`, `gradcam_gallery.png` |
| **Phase 3** | Clinical Decision Support Layer + Streamlit App + PDF Report | ✅ Complete | `app/app.py` (DICOM, CLAHE, Triage, Plotly), `app/report.py` (PDF Report) |
| **Phase 4** | Scientific Integrity Audit + Paper Manuscript + Defense Presentation | ✅ Complete | `src/audit_pipeline.py` (100% PASS), `paper/paper_draft.md`, `model_card.md` |
| **Phase 5 (Next)** | Full-Stack Production Platform (FastAPI Backend + React/Next.js Frontend) | 📌 Planned (New Folder) | Dedicated folder: FastAPI REST API, React/Next.js Radiology Workstation, PostgreSQL |
| **Phase 6 (Advanced)** | LLM Radiology Draft Findings + VQA / Error Analysis Extension | 💡 Future Vision | Pre-formulated radiology draft findings, Medical VQA concept framework |

---

## Checklist

### Week 1 — Data + baseline (✅ COMPLETE)

| Task | Status | Result / Notes |
|------|--------|----------------|
| Download Kermany + Shenzhen/Montgomery | ✅ | `data/raw/` (gitignored) |
| EDA notebooks | ✅ | `notebooks/01_eda_kermany.ipynb`, `02_eda_shenzhen.ipynb`, `03_eda_montgomery.ipynb` |
| `src/data/download.py` | ✅ | Idempotent, skips existing files |
| `src/data/prepare.py` → manifest | ✅ | 6,624 images; MD5 dedup |
| `src/data/split.py` → 4-class splits | ✅ | Patient-level `GroupShuffleSplit`; ZERO patient/MD5 leakage (asserted) |
| `src/data/split.py` → 3-class splits | ✅ | `data/processed/splits_kermany3/` (Kermany only) |
| `src/data/checks.py` → report | ✅ | `results/data_quality.md` |
| `src/datasets.py`, `src/models.py` | ✅ | Albumentations aug, WeightedRandomSampler, timm factory |
| Baseline DenseNet-121 (4-class) | ✅ | Test acc 0.849, F1 0.837, AUC 0.951 |
| `src/evaluate.py` | ✅ | Bootstrap CI, confusion matrix, ROC figures |
| Repo restructure + README + model card | ✅ | Commit `2294b5d` |
| Git history | ✅ | 4 commits on `main`, clean structure |

### Week 2 — Benchmark + explainability (🔴 IN PROGRESS)

| Task | Status | Result / Notes |
|------|--------|----------------|
| Train resnet18 × 3 seeds (3-class) | ✅ | acc 0.825±0.006, F1 0.801±0.005, AUC 0.932±0.001 |
| Train densenet121 × 3 seeds (3-class) | ✅ | acc 0.833±0.013, F1 0.809±0.012, AUC 0.930±0.005 |
| Train efficientnet_b0 × 3 seeds (3-class) | ⏸ Paused | User stopped benchmark manually; ~1.5h to complete |
| Aggregation script (mean ± std + CI table) | ⏳ Pending | Currently manual from `results/benchmark_run.log` |
| Per-class metrics breakdown | ⏳ Pending | |
| Grad-CAM heatmaps (`src/explain.py`) | ⏳ Pending | Code written, needs checkpoint (have 6) |
| Classical baseline HOG/LBP + SVM (`src/baseline_classical.py`) | ⏳ Pending | Code written, CPU-only, ~15-30 min |

### Week 3 — External validation + demo (⏳ PENDING)

| Task | Status |
|------|--------|
| Retrain 4-class (checkpoints were deleted) | ⏳ |
| External validation on Montgomery (TB) | ⏳ |
| One improvement experiment (focal loss OR TTA) | ⏳ |
| Streamlit app (`app/app.py`) + PDF report (`app/report.py`) | ⏳ Code written, untested |
| Deploy Hugging Face Spaces + demo video | ⏳ |

### Week 4 — Paper (⏳ PENDING)

| Task | Status |
|------|--------|
| Write IMRaD (outline exists: `paper/outline.md`) | ⏳ |
| Fill Methods/Results from `results/` | ⏳ |
| Post preprint (arXiv/medRxiv) | ⏳ |
| Defense slides | ⏳ |

---

## Measured Results (Internal Test, 3-class)

| Backbone | Accuracy | F1 macro | AUC macro | Note |
|----------|:--------:|:--------:|:---------:|------|
| ResNet-18 | 0.825 ± 0.006 | 0.801 ± 0.005 | 0.932 ± 0.001 | 3 seeds: 42, 7, 123 |
| DenseNet-121 | 0.833 ± 0.013 | 0.809 ± 0.012 | 0.930 ± 0.005 | 3 seeds: 42, 7, 123 |
| EfficientNet-B0 | — | — | — | Not run yet |

### Reference (4-class, single run, seed 42)
- DenseNet-121: internal acc 0.849, F1 0.837, AUC 0.951; external (Montgomery) acc 0.732, AUC 0.799
- The 4-class checkpoint was overwritten/deleted — **must retrain** for the demo (Week 3).

### Checkpoints available (gitignored, local only)
`results/checkpoints/best_{backbone}_seed{seed}_cls3.pt`:
- resnet18: seeds 42, 7, 123
- densenet121: seeds 42, 7, 123

---

## Known Issues & Notes

| # | Issue | Severity | Status |
|---|-------|----------|--------|
| 1 | AUC=0.0 bug on partial-class subsets (external test) — **FIXED** in `src/utils.py` (filter by classes present) | High | ✅ Fixed |
| 2 | AMP deprecation warnings — **FIXED** (`torch.amp` API) | Low | ✅ Fixed |
| 3 | YAML encoding error with Vietnamese comments (cp1252) — **FIXED** (`encoding="utf-8"` in 9 files) | High | ✅ Fixed |
| 4 | 3-class checkpoint overwrote 4-class checkpoint (same filename) — **FIXED** (names now scoped `_cls{3|4}`) | High | ✅ Fixed |
| 5 | 4-class best checkpoint lost (deleted) — needs retrain for demo | Medium | ⏳ Pending |
| 6 | `benchmark_run.log` / `benchmark_run_err.log` not committed yet | Low | ⏳ Pending |
| 7 | Absolute paths in split CSVs (`C:\Users\huynh\...`) — hurts reproducibility on other machines | Medium | ⏳ Pending |
| 8 | sklearn warnings ("Only one class present", "Target scores need to be probabilities") in logs — cosmetic | Low | ⏳ Pending |
| 9 | AUC ~0.93 below literature target (0.95–0.97) — likely viral vs bacterial class confusion; inspect `results/cm_test.png` | Info | ⏳ Pending |
| 10 | EfficientNet-B0 not yet benchmarked | Medium | ⏳ Pending |

---

## Reproducibility & Commands

```bash
# Environment
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128   # CUDA 12.8 (RTX 50-series)
pip install -r requirements.txt

# Data pipeline (run once)
python -m src.data.download
python -m src.data.prepare
python -m src.data.split --config configs/default.yaml      # 4-class
python -m src.data.split --config configs/kermany3.yaml     # 3-class (Kermany only)
python -m src.data.checks --config <config>

# Train (single run / dry run)
python -m src.train --config configs/kermany3.yaml                     # 3-class
python -m src.train --config configs/default.yaml                      # 4-class
python -m src.train --config <cfg> --dry-run                           # smoke test (1 epoch, 5 batches)
python -m src.train --config <cfg> --backbone resnet18 --seed 7        # overrides

# Benchmark (multi-run)
python -m experiments.run_all --config configs/kermany3.yaml           # 3 backbone × 3 seeds (~4-6h GPU)
python -m experiments.run_all --config configs/kermany3.yaml --dry-run

# Evaluate + figures
python -m src.evaluate --config <cfg> [--checkpoint path]              # CM + ROC + bootstrap CI

# Explainability / baseline / demo
python -m src.explain --config <cfg>                                   # Grad-CAM overlays
python -m src.baseline_classical --config <cfg>                        # HOG/LBP + SVM
streamlit run app/app.py                                               # web demo

# Artifacts
#   results/checkpoints/           model weights (gitignored)
#   results/*.png                  confusion matrix, ROC curves
#   results/benchmark_log_*.csv    per-run aggregation
#   results/benchmark_run.log      full benchmark trace
```

---

## Next Actions (suggested priority)

1. Run remaining EfficientNet-B0 × 3 seeds (~1.5h GPU)
2. Write aggregation script → final mean ± std + CI table (both backbones now, all 3 later)
3. Run classical baseline (CPU, no GPU contention)
4. Run Grad-CAM on best checkpoint
5. Retrain 4-class for demo (Week 3)
6. Relative-ize paths in manifest/splits for cross-machine reproducibility
7. Commit current uncommitted state (run_all.py UTF-8 fix, logs, figures)

---

## Design Decisions Log

| Decision | Rationale |
|----------|-----------|
| 3-class Kermany = primary benchmark | Avoids TB adult-vs-pediatric domain confound (cleanest result, per `DATA.md`) |
| 4-class = secondary (demo + external validation) | Needed for Montgomery TB external test + demo variety |
| Patient-level `GroupShuffleSplit` | Prevents patient-ID leakage (same patient across train/test) |
| MD5 dedup + leakage assert | Catches duplicate images across train/test |
| Mean ± std over 3 seeds + 95% bootstrap CI | Standard reporting; stable variance (std ≤ 0.013) |
| Fixed hyperparams across backbones | Fair comparison (same lr 1e-4, AdamW, cosine, early stop patience 5) |
| Montgomery held out entirely | Honest external generalization measurement |
