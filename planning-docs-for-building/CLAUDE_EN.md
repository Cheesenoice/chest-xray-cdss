# CLAUDE.md — Context & Plan for Graduation Thesis Project

> This file is the central context of the project. Claude Code MUST read this thoroughly before
> writing any line of code. It describes: who the user is, the circumstances, the chosen topic,
> mandatory scientific standards, technical architecture, repo structure, 4-week timeline, and
> pitfalls to avoid. When there is a conflict between ad-hoc requests and this file, prioritize
> this file and ask the user for clarification.

---

## 0. TL;DR for Claude Code

We are building an **Explainable Deep Learning Clinical Decision Support System for Chest
X-ray Screening**. This serves dual purposes: (1) a **polished demo product** to present to the
graduation defense committee, and (2) **rigorous experimental research** to write a publishable
paper/preprint.

The research contribution is NOT "invent a new model architecture." Instead, it is a
**reproducible, explainable, and honestly externally-validated benchmark**. This is the
feasible research gap within 1 month and exactly what medical-AI reviewers value.

Final deliverables: (1) clean, runnable, reproducible code repository; (2) model achieving
strong metrics; (3) web app demo (upload X-ray → diagnosis → Grad-CAM heatmap → auto-report);
(4) a preprint + paper draft.

---

## 1. User Context (Read to Understand Constraints)

- This is a **bachelor-level graduation thesis**, built in 2026, the era of AI agents.
- The user **learns while building**; most code and logic are AI-assisted (Claude Code generates).
  => Code must be **clear, well-commented, easy to understand so the user can learn from it**.
  Avoid clever/opaque code. Prioritize readability over extreme optimization.
- **Timeline: exactly 1 MONTH (4 weeks).** Every technical decision must prioritize feasibility
  within this window. No novel architectures from scratch, no 3D volumetric segmentation, no
  large EHR pipelines.
- **Compute resources:** Free Kaggle GPU (T4/P100) + free powerful university GPU server.
  Dataset must be small enough to train on these resources in minutes to hours.
- **Dual objective:**
  1. Have a **polished demo product + strong evidence metrics** to submit and defend to the
     committee.
  2. Have a **scientifically rigorous paper/preprint** to apply for master's scholarships in
     China (CSC) or South Korea (GKS). High venue ranking is NOT required — a publicly
     available, legitimate, citable output is sufficient.
- **Realistic paper expectations:** within 1 month you will CERTAINLY write and submit a
  preprint, but will NOT have time for full peer-review acceptance. Thus the mandatory output =
  **preprint (arXiv/medRxiv) + public code repository + live demo**; conference/journal
  submission is "nice to have."
- Everything must be done **to proper scientific research standards** — this is a hard
  requirement, see section 4.

---

## 2. Chosen Topic (Final)

**Name (VN):** Hệ thống hỗ trợ sàng lọc và quyết định lâm sàng từ ảnh X-quang ngực bằng học
sâu có giải thích.

**Name (EN, for paper):** *An Explainable Deep Learning Clinical Decision Support System for
Chest X-ray Screening.*

**Problem:** Multi-class classification of chest X-ray images. Default configuration is **4
classes**: `Normal` / `Bacterial pneumonia` / `Viral pneumonia` / `Tuberculosis (TB)`.

Pipeline must be **flexible and configurable** by number of classes (able to run 3-class
Normal/Bacterial/Viral, or 4-class with TB) via config file, for scientific reasons explained
in section 4.3.

**Clinical narrative (to answer committee questions):** In primary-care facilities lacking
radiologists, a triage system flags high-suspicion cases (red) for urgent referral, with
Grad-CAM heatmaps for physician verification. This is **clinical decision support**, not
replacement of physician judgment.

---

## 3. Research Framing (research gap / contribution)

Do NOT frame this as "we propose a novel model." Frame it as a **rigorous reproducible
benchmark study**, with 4 pillars — these are the contributions and also what peer-reviewed
medical-AI papers typically lack:

1. **Reproducibility:** compare 3–4 backbones under IDENTICAL conditions (same splits, same
   augmentation, same schedule), report **mean ± standard deviation over ≥3 random seeds**.
2. **Explainability:** Grad-CAM (optionally Grad-CAM++ or Score-CAM) overlay on images; verify
   qualitatively that the model attends to the correct lung regions of pathology.
3. **External validation:** evaluate on a **completely held-out source** that the model never
   saw during training, to honestly measure generalization gap. This is the most persuasive
   contribution and answers "is the model overfitting?"
4. **Comparison with classical methods:** one classical ML baseline (hand-crafted features HOG/
   LBP + SVM or Logistic Regression) to satisfy the thesis requirement "compare with traditional
   methods."

One accuracy-improvement intervention (choose one): class-balanced loss / focal loss, or
test-time augmentation, or ensembling — report the performance delta vs. baseline.

---

## 4. MANDATORY SCIENTIFIC STANDARDS (Read Carefully — Violating These Ruins the Paper)

### 4.1. Data Leakage Prevention — Priority #1
- **Split data at PATIENT/SOURCE level, NOT image level.** Images from the same patient must
  never appear in both train and test. This is the #1 leakage error in medical imaging; reviewers
  always check. If the dataset lacks patient IDs, clearly state this limitation and split by
  image source (at source level where possible).
- Remove duplicate images before splitting.

### 4.2. Three-way Split & Evaluation Protocol
- Split into **train / validation / test**. Tune ONLY on validation. **Touch test set EXACTLY
  ONCE** at the very end; do not use it for model selection.
- Fix the split ratio and log it in config; record number of images per class per split.

### 4.3. Domain Confound Warning (Critical for 4-class Config)
- Kermany dataset = **pediatric X-rays** (Guangzhou); TB datasets (Shenzhen/Montgomery) =
  **adults**. If merging into 4 classes, the model MAY distinguish TB from pneumonia based on
  age/anatomy rather than pathology => confound.
- Mitigation strategy (MANDATORY to write in paper as a limitation + motivation for external
  validation): (a) still build 4-class for the demo product, BUT (b) explicitly acknowledge the
  domain difference, and (c) design external validation on **held-out sources** to honestly
  expose the issue. Additionally, run in parallel a clean 3-class benchmark ONLY on Kermany
  (Normal/Bacterial/Viral) as the most reliable result.

### 4.4. Metric Reporting
- Mandatory: **Accuracy, Precision, Recall, F1 (macro), AUC (macro / one-vs-rest)**.
- For imbalanced data, also report **balanced accuracy** and **per-class metrics** (not just
  overall accuracy).
- Report **mean ± std over ≥3 seeds** (e.g., seeds 42, 7, 123).
- Report **95% confidence intervals (bootstrap)** for the primary metric.
- Plot **confusion matrix** and **ROC curve** on test set.
- If time permits: **calibration** (reliability diagram + ECE).

### 4.5. Reproducibility & Ethics
- Fix all random seeds (Python, NumPy, PyTorch, cuDNN deterministic if possible).
- Document hardware, number of epochs, training time, library versions (`requirements.txt` or
  `environment.yml` with pinned versions).
- Use only publicly licensed data; **cite the original dataset paper** for each source; note the
  license (some datasets like HAM10000/ISIC are CC BY-NC — non-commercial use only).
- Export a short **model card** (purpose, data, limitations, not for self-diagnosis).
- Clearly state: this system is for clinical decision support, NOT a substitute for physician
  diagnosis.

---

## 5. Data (Datasets)

> Kaggle dataset slugs may change — Claude Code MUST verify current URLs on Kaggle before
> downloading and prioritize using Kaggle API (`kaggle datasets download`) or `kagglehub`.

**Primary source — pneumonia (pediatric):**
- Kermany "Chest X-Ray Images (Pneumonia)". Reference slug:
  `paultimothymooney/chest-xray-pneumonia`. ~5,856 images, classes Normal/Pneumonia; bacterial
  vs. viral labels inferred from filename (string "bacteria"/"virus"). License CC BY 4.0. Small
  and trains quickly.

**Source for TB class (adults) — used for 4-class config:**
- TB Chest X-ray Database. Reference slug: `tawsifurrahman/tuberculosis-tb-chest-xray-dataset`,
  or Shenzhen + Montgomery from `kmader/pulmonary-chest-xray-abnormalities`. Images labeled
  TB / normal.

**External validation (held-out source — NOT used for training):**
- Choose a different pneumonia source than Kermany to test generalization (e.g., another
  pneumonia/COVID dataset on Kaggle), OR with TB: train on Shenzhen and hold out all of
  Montgomery for external testing (or vice versa). Document this strategy clearly.

Alternatives (only if user requests a topic change — already deliberated, default DO NOT
change): HAM10000 dermatology; APTOS 2019 diabetic retinopathy; MIT-BIH ECG arrhythmia.
**Default: proceed with chest X-ray.**

---

## 6. Technical Architecture

**Language / Framework:** Python 3.10+, PyTorch (preferred) + timm (pretrained backbone library),
scikit-learn (classical baseline + metrics), albumentations (augmentation), pytorch-grad-cam
(Grad-CAM), matplotlib/seaborn (plotting).

**Model family (transfer learning, ImageNet pretrained — trains quickly on free GPU):**
minimum comparison:
- `ResNet-18` (lightweight baseline)
- `DenseNet-121` (well-benchmarked on chest X-ray)
- `EfficientNet-B0` (parameter-efficient)
- (optional) `Swin-Tiny` or `ViT-Small` for Transformer diversity per thesis requirements.

Images resized to 224×224, normalized by ImageNet mean/std. Reasonable augmentation (horizontal
flip, gentle rotation, brightness/contrast jitter) — NO augmentation that distorts pathology.

**Training config (in config file, not hardcoded):** AdamW optimizer, lr ~1e-4 with fine-tuning,
cosine or ReduceLROnPlateau scheduler, early stopping on validation macro-F1, mixed precision
(AMP) for speed. Handle class imbalance with class weights or WeightedRandomSampler.

**Explainability:** Grad-CAM on final conv layer; export heatmap overlays. Save several correct
and incorrect examples for the paper.

**Classical baseline:** extract HOG or LBP features → SVM/LogReg (scikit-learn). Evaluate on same
test set for comparison.

**Demo product (web app):**
- Backend: FastAPI serving inference + Grad-CAM generation, or integrate directly into Streamlit/
  Gradio for speed.
- Frontend: **Streamlit** or **Gradio** (prefer speed) — workflow: upload X-ray → predict →
  display class probabilities → overlay Grad-CAM heatmap → triage alert (red/yellow/green) →
  export PDF report button (reportlab/fpdf) → (optional) case history dashboard.
- Deployment: **Hugging Face Spaces** (free) for a live shareable link for committee to test.
  Include local run instructions in README.

---

## 7. Repo Structure (Recommended)

```
chest-xray-cdss/
├── CLAUDE.md                 # this file
├── README.md                 # description, installation, usage, results, demo link
├── requirements.txt          # pinned versions
├── configs/
│   └── default.yaml          # num classes, data paths, hyperparams, seed list
├── data/                     # DO NOT commit data; script to download only + .gitignore
│   └── README.md
├── src/
│   ├── data/
│   │   ├── download.py       # download datasets via kaggle API
│   │   ├── prepare.py        # merge sources, assign labels, remove duplicates
│   │   └── split.py          # split at patient/source level, export split CSV
│   ├── datasets.py           # Dataset/DataLoader + augmentation
│   ├── models.py             # backbone factory (timm)
│   ├── train.py              # training loop, logging, early stopping, checkpointing
│   ├── evaluate.py           # metrics, bootstrap CI, confusion matrix, ROC
│   ├── explain.py            # Grad-CAM overlay
│   ├── baseline_classical.py # HOG/LBP + SVM
│   └── utils.py              # seed management, logging, reproducibility helpers
├── experiments/
│   └── run_all.py            # run multiple backbones × multiple seeds, aggregate results
├── results/                  # metric tables, figures, logs (commit these)
├── app/
│   ├── app.py                # Streamlit/Gradio demo
│   └── report.py             # PDF report generation
├── notebooks/                # EDA, quick experimentation
├── model_card.md
└── paper/
    ├── outline.md            # IMRaD structure (see section 10)
    └── figures/
```

Convention: **do not commit data and heavy checkpoints** (use `.gitignore`); commit config,
code, result tables/figures, logs.

---

## 8. Performance Targets (as Reference Points)

Based on literature (use as benchmark, not absolute targets):
- Pneumonia binary/multi-class on Kermany: accuracy ~90–95%, AUC ~0.95–0.97.
- With correct patient-level splits + external validation, **expect internal metrics HIGH and
  external metrics LOWER** (e.g., internal AUC ~0.96, external ~0.85–0.90). **THIS is the
  honest result and deserving of praise** — do not "adjust" for appearance.
- If best model < 85% accuracy or AUC < 0.90 on internal test => suspect error (leakage,
  wrong augmentation, mislabeling) => debug, DO NOT switch topics.

Warning: the ~99% accuracy figures in many papers typically result from leaky splits or
duplicate images. Do not aim for that or claim it with proper splitting.

---

## 9. 4-Week Timeline (Orientation for Workflow)

**Week 1 — Data + baseline + repo setup.**
Initialize repo, config, seed. Download Kermany (+ TB). EDA. Split train/val/test using CORRECT
standards from section 4. Write Dataset/DataLoader. Train one DenseNet-121 baseline end-to-end
to get first results. Initialize Git + README + requirements.

**Week 2 — Benchmarking + tuning + explainability.**
Train 3–4 backbones × 3 seeds under identical conditions. Log full metrics + std. Add Grad-CAM.
Write classical baseline (HOG/LBP+SVM). Begin professor outreach in parallel.

**Week 3 — External validation + demo product.**
Run best model on held-out source, measure generalization gap. Try 1 accuracy-improvement
intervention, report delta. Build web app (upload→predict→Grad-CAM→report), deploy to Hugging
Face Spaces. Record demo video.

**Week 4 — Writing + preprint + submission.**
Write paper IMRaD (AI assists, but USER verifies all numbers & citations). Post preprint to
arXiv/medRxiv. Submit to workshop/journal if deadline permits. Tag repo release (seed + env).
Prepare defense slides + technical specification sheet.

---

## 10. Paper Structure (IMRaD Template for Reference)

- **Title / Abstract:** state the problem, contributions (reproducible benchmark + XAI + external
  validation), main findings.
- **Introduction:** context of AI for chest X-ray, the gap (single-dataset over-reliance, lack
  of external validation, lack of explainability), our contributions.
- **Related Work:** brief survey of CNN/Transformer approaches to chest X-ray.
- **Methods:** dataset + license + splitting strategy (emphasize patient/source level), backbones,
  training config, Grad-CAM, classical baseline, metrics + CI.
- **Results:** complete metric tables (mean±std, CI), confusion matrix, ROC, Grad-CAM figures,
  external validation, improvement intervention delta.
- **Discussion:** clinical significance, honest discussion of external-validation drop, comparison
  with published benchmarks.
- **Limitations:** domain confound (pediatric vs. adult), 1–2 data sources only, no real clinical
  deployment tested.
- **Conclusion + Reproducibility statement** (code link, seeds, environment).

Venues: preprint to arXiv (eess.IV / cs.CV) or medRxiv (mandatory). Optional submission (if
deadline permits): MICCAI satellite workshop (e.g. AMAI), IEEE ICHI/Healthcom, or DOAJ/Scopus-
indexed open-access journal. **Avoid predatory journals** (verify on DOAJ/Scopus/WoS, use
Think-Check-Submit checklist; be suspicious of "accept in 3 days" promises).

---

## 11. How Claude Code Should Operate (Conventions)

- **Build incrementally; run at each step.** Prioritize getting an end-to-end pipeline with
  results early (even if partial), then expand. Never write a massive block before running.
- **Test on small data first** (hundreds of images, 1–2 epochs) to catch bugs fast, then full
  run.
- **Code should be readable, with docstrings + comments** to help the user understand. Explain
  "why," not just "what."
- **Every constant/hyperparameter goes in `configs/`**, not hardcoded scattered throughout.
- **Seed & reproducibility as defaults**, not add-on features.
- **Print clear logs:** size of each split, number of images per class per split, metrics per
  epoch.
- When a step has risk (easy leakage, easy mislabeling), **pause and explain to the user**
  before proceeding.
- Before downloading data or using Kaggle slugs, **verify the slug still exists**; if API
  requires token, guide the user to set up `kaggle.json`.
- Prefer stable, common libraries (timm, pytorch-grad-cam, albumentations) over reinventing.

---

## 12. Pitfalls to Avoid (Summary)

- Split at image level (leakage) → always split at patient/source level.
- Use test set for tuning → touch test set only once.
- Duplicate images across splits.
- Merge pediatric (pneumonia) + adult (TB) data without disclosing domain confound.
- Claim ~99% accuracy like leaky papers.
- Attempt 3D BraTS segmentation / MIMIC credentialed / federated learning complexity — CANNOT
  fit in 1 month.
- Submit to predatory journals.
- Report only overall accuracy on imbalanced data.

---

## 13. Definition of Done

- [ ] Reproducible pipeline: clone repo + set up data + run one command → metrics match.
- [ ] Complete metric tables (mean±std, CI) + confusion matrix + ROC + Grad-CAM figures.
- [ ] External validation with honest performance drop and discussion.
- [ ] Classical baseline for comparison.
- [ ] Working web app demo + deployed on Hugging Face Spaces + demo video.
- [ ] Model card + reproducibility statement.
- [ ] Preprint ready to post + paper draft in IMRaD form.
- [ ] Defense slides + technical specification sheet for committee.
