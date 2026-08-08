# MedVision AI — Intelligent Radiology Decision Support Platform
## Product Specification & UI Workflow Documentation

> **Project Title (Graduation Thesis):**  
> *Xây dựng Hệ thống AI Đa mô hình Hỗ trợ Phát hiện Bất thường, Giải thích Kết quả và Hỗ trợ Quyết định Lâm sàng từ Ảnh X-quang Ngực*  
> *(An Explainable Deep Learning Clinical Decision Support System for Chest X-Ray Screening)*

---

## 1. System Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          USER / CLINICIAN INTERFACE                         │
│                    (Radiologist / Primary-Care Physician)                   │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                        1. Upload DICOM / JPEG / PNG Scan
                        2. Patient Metadata (Age, SpO2, Temp, Symptoms)
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                       MEDVISION AI PLATFORM (STREAMLIT)                     │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
       ┌───────────────────────────────┼───────────────────────────────┐
       ▼                               ▼                               ▼
┌──────────────┐              ┌────────────────┐              ┌────────────────┐
│   MODULE 1   │              │    MODULE 2    │              │    MODULE 3    │
│ Medical Image│              │  AI Diagnosis  │              │ Explainable AI │
│  Management  │              │     Engine     │              │ (Grad-CAM XAI) │
└──────┬───────┘              └───────┬────────┘              └───────┬────────┘
       │                              │                               │
       └──────────────────────────────┼───────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                   MODULE 4: CLINICAL DECISION SUPPORT LAYER                 │
│  - Triage Priority (RED / YELLOW / GREEN)                                    │
│  - Risk Scoring Engine (Combines AI Probability + SpO2 + Temp + Symptoms)   │
│  - Automated 1-Click PDF Report Generator (app/report.py)                   │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                 MODULE 5: MODEL EVALUATION & RESEARCH DASHBOARD             │
│  - Plotly Hospital Analytics (6,900 Scans / 3,915 Patients)                 │
│  - Benchmark Table (DenseNet121 vs ResNet18 vs EfficientNet0 vs HOG+SVM)   │
│  - Error Analysis (Cases model got wrong)                                   │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Detailed UI Workflows & Screen Layouts

### Screen 1: Medical Image Management & Case Setup
- **Supported File Formats:** DICOM (`.dcm` via `pydicom`), PNG, JPEG.
- **Patient Profile Fields:** Patient ID, Full Name, Age, Gender, Body Temperature (°C), Oxygen Saturation ($SpO_2$), Clinical Symptoms (Dyspnea, Cough, Fever).
- **Preset Test Scans Selector:** Instant 1-click loading of real test scans for graduation defense presentation.

```
+-----------------------------------------------------------------------------+
| 👤 PATIENT METADATA                                                         |
| Patient ID: [ PAT-2026-0089 ]   Name: [ Anonymous Patient ]                 |
| Age: [ 34 ]   Gender: [ Male ]   SpO2: [ 92% ]   Temp: [ 38.5 °C ]          |
| Symptoms: [x] Dyspnea   [x] High Fever   [ ] Chest Pain                     |
+-----------------------------------------------------------------------------+
| 📁 MEDICAL IMAGE INGESTION                                                  |
| Drag and drop file here: [ DICOM / JPEG / PNG ]                             |
| [x] Enable CLAHE Adaptive Histogram Contrast Enhancement                    |
+-----------------------------------------------------------------------------+
```

---

### Screen 2: AI Diagnosis Engine & Pathology Detection
- **Multi-Class Detection Engine:**
  - `Normal` (Low Risk)
  - `Bacterial Pneumonia` (High Suspicion)
  - `Viral Pneumonia` (Moderate Suspicion)
  - `Tuberculosis` (High Suspicion)
- **Model Selection:** `DenseNet-121` (Best Generalization), `ResNet-18`, `EfficientNet-B0`.

```
+-----------------------------------------------------------------------------+
| 🔬 AI DIFFERENTIAL PATHOLOGY PROBABILITIES                                  |
| Bacterial Pneumonia:  [████████████████████████░░░░]  84.2%                 |
| Normal:               [███░░░░░░░░░░░░░░░░░░░░░░░░░]  11.5%                 |
| Viral Pneumonia:      [█░░░░░░░░░░░░░░░░░░░░░░░░░░░]   3.1%                 |
| Tuberculosis:         [█░░░░░░░░░░░░░░░░░░░░░░░░░░░]   1.2%                 |
+-----------------------------------------------------------------------------+
```

---

### Screen 3: Explainable AI & Pathological Region Localization
- **Grad-CAM Heatmap Overlay:** Visual attention maps over target convolutional layer (`denseblock4`).
- **Pathological Feature Highlight:** Direct spatial alignment over affected lung opacities and infiltrates.

```
+------------------------------------+------------------------------------+
| 📸 ORIGINAL CHEST X-RAY            | 🔥 GRAD-CAM PATHOLOGICAL HEATMAP   |
|                                    |         ░░▒▒▓▓████▓▓▒▒             |
|                                    |         ░▒▒▓▓██████▓▓▒             |
|                                    |         [ Highlighted Opacity ]      |
| Resolution: 224x224 px             | Target Layer: denseblock4          |
+------------------------------------+------------------------------------+
```

---

### Screen 4: Clinical Decision Support & Priority Triage Layer
- **Triage Alert Status:**
  - 🚨 **RED ALERT (High Suspicion):** Bacterial Pneumonia / Tuberculosis -> Immediate Specialist Referral.
  - ⚠️ **YELLOW ALERT (Moderate Suspicion):** Viral Pneumonia -> Secondary Monitoring & Isolation.
  - ✅ **GREEN (Low Risk):** Normal Chest X-Ray.
- **Combined Risk Evidence:** Merges AI prediction probability + Reduced $SpO_2$ + Fever symptoms.
- **1-Click PDF Report Export:** Generates printable diagnostic PDF (`app/report.py`) with physician sign-off block.

```
+-----------------------------------------------------------------------------+
| 🚨 URGENT REFERRAL: BACTERIAL PNEUMONIA (84.2% Confidence)                  |
| Priority: Immediate Radiologist Review Required                             |
| Supporting Evidence:                                                        |
|  • High probability bacterial pneumonia opacity (84.2%)                    |
|  • Reduced oxygen saturation (SpO2: 92%)                                    |
|  • Reported dyspnea and elevated body temperature (38.5°C)                  |
|                                                                             |
| 📄 [ Export Diagnostic PDF Report ]                                         |
+-----------------------------------------------------------------------------+
```

---

### Screen 5: Research Benchmark & Evaluation Dashboard
- **Completed Multi-Seed Empirical Results (Mean ± Std over 3 Seeds):**

| Architecture | Internal Accuracy | Internal F1 (Macro) | Internal AUC | External AUC (Montgomery OOD) |
|---|:---: |:---: |:---: |:---: |
| **ResNet-18** | **84.98% ± 0.33%** | **84.09% ± 0.22%** | **0.9538 ± 0.0020** | 0.7606 ± 0.0072 |
| **DenseNet-121** | 84.55% ± 0.22% | 83.22% ± 0.55% | 0.9512 ± 0.0024 | **0.8296 ± 0.0613 (BEST)** |
| **EfficientNet-B0** | 83.83% ± 0.91% | 82.69% ± 0.88% | 0.9493 ± 0.0150 | 0.7208 ± 0.0360 |
| **HOG + SVM Baseline** | 83.51% | 79.40% | 0.9470 | **0.6052 (COLLAPSE)** |

- **Error Analysis Section:** Inspection tool for cases misclassified by models during testing to answer committee questions (*"Where does your model fail?"*).

---

## 3. Scientific Integrity Audit Verification

- **Patient Leakage Audit (`src/audit_pipeline.py`):** **100% PASS**
  - Train (2,740 patients), Val (587 patients), Test (588 patients), External (138 patients) have **zero patient ID overlap**.
- **MD5 Hash Audit:** **100% PASS** (5,888 duplicate scans removed).
- **External Site Isolation:** **100% PASS** (Montgomery set 100% held-out).
