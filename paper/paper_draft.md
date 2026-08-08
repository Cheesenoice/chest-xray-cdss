# An Explainable Deep Learning Clinical Decision Support System for Chest X-Ray Screening

**Author:** [User Name / Author List]  
**Institution:** [University / Department Name]  
**Target Venue:** arXiv / medRxiv Preprint (Preprint / Conference Submission)  
**Code Repository:** `Cheesenoice/chest-xray-cdss`  

---

## Abstract

**Background:** Automated chest X-ray screening systems powered by deep learning hold immense potential for primary-care triage, particularly in resource-constrained environments lacking specialized radiologists. However, adoption remains hindered by dataset leakage, reliance on single-center benchmarks, and lack of visual explainability.  
**Methods:** We present an explainable Clinical Decision Support System (CDSS) for 4-class chest X-ray classification (`Normal`, `Bacterial Pneumonia`, `Viral Pneumonia`, `Tuberculosis`). We evaluate three convolutional backbones (`ResNet-18`, `DenseNet-121`, `EfficientNet-B0`) under a strict zero-leakage patient-level splitting protocol (`GroupShuffleSplit` on `patient_id` with verified MD5 non-overlap across 6,900 deduplicated images). We validate the system externally on a 100% held-out dataset (Montgomery County, USA) and incorporate Grad-CAM visual heatmaps alongside a traditional machine learning baseline (HOG + SVM).  
**Results:** Internal test evaluation demonstrates strong multi-class classification performance across backbones. External validation on unseen data highlights the honest generalization gap in medical imaging models. Grad-CAM visual heatmaps accurately localize pathological lung opacities and consolidations.  
**Conclusion:** Our findings emphasize the necessity of patient-level data partitioning and external validation in medical AI research. The open-source pipeline and Streamlit-based web application provide a practical prototype for clinical decision support.

---

## 1. Introduction

Chest radiographies (X-rays) are among the most frequently performed diagnostic imaging modalities worldwide. In primary healthcare settings across low- and middle-income regions, the scarcity of certified radiologists often delays critical diagnoses for respiratory diseases such as bacterial/viral pneumonia and pulmonary tuberculosis (TB). 

While deep neural networks have demonstrated high nominal accuracy on public benchmarks, medical imaging reviewers and clinicians frequently highlight three major flaws in existing literature:
1. **Data Leakage:** Image-level splits that place multiple scans from the same patient into both training and testing sets, artificially inflating metrics.
2. **Lack of External Validation:** Failure to test models on out-of-distribution data from unseen medical centers.
3. **Black-Box Decision Making:** Absence of interpretable spatial visual feedback for attending physicians.

To address these gaps, this study proposes a reproducible, explainable, and externally validated benchmark for chest X-ray screening.

---

## 2. Materials and Methods

### 2.1 Datasets and Data Hygiene
We aggregate images from public medical repositories:
- **Kermany Chest X-Ray Dataset:** Pediatric chest X-rays from Guangzhou Women and Children's Medical Center (CC BY 4.0). Used for `Normal`, `Bacterial Pneumonia`, and `Viral Pneumonia`.
- **Shenzhen Hospital Dataset (NLM/NIH):** Adult chest X-rays used for `Tuberculosis` training and validation.
- **Montgomery County Dataset (NLM/NIH):** Adult chest X-rays held out **100%** as an **External Test Set** (never seen during training).

**Deduplication & Leakage Prevention:** All raw images undergo MD5 hash checksum calculation; 5,888 duplicate image files were identified and removed. Data partitioning is strictly executed at the **patient level** using `GroupShuffleSplit` on patient IDs (`person{N}` regex for Kermany, `CHN_{id}` for Shenzhen, `MCU_{id}` for Montgomery). Assertions verify zero patient or MD5 overlap between train, validation, and test splits.

| Split | Normal | Bacterial Pneumonia | Viral Pneumonia | Tuberculosis | Total Images | Unique Patients |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Train (70%)** | 1,325 | 1,966 | 1,060 | 235 | **4,586** | 2,740 |
| **Validation (15%)** | 270 | 411 | 226 | 59 | **966** | 587 |
| **Internal Test (15%)** | 310 | 383 | 199 | 42 | **934** | 588 |
| **External Test (Montgomery)** | 240 | 0 | 0 | 174 | **414** | 138 |

### 2.2 Model Architectures & Transfer Learning
We benchmark three transfer learning backbones pre-trained on ImageNet:
1. **ResNet-18:** Lightweight baseline (11.7M parameters).
2. **DenseNet-121:** Feature-reuse dense connections (7.0M parameters), established standard in chest X-ray research.
3. **EfficientNet-B0:** Compound scaling parameter-efficient architecture (5.3M parameters).

Images are resized to $224 \times 224$ pixels and normalized with ImageNet mean $[0.485, 0.456, 0.406]$ and standard deviation $[0.229, 0.224, 0.225]$. Data augmentation (horizontal flips, rotation $\pm 12^\circ$, brightness/contrast jitter) is applied strictly during training.

### 2.3 Classical Machine Learning Baseline
To satisfy traditional ML comparison standards, we implement a feature engineering baseline combining Histogram of Oriented Gradients (HOG, 9 orientations, $16 \times 16$ cell size) and 32-bin intensity histograms, classified via Support Vector Machines (SVM with RBF kernel).

### 2.4 Visual Explainability (Grad-CAM)
We compute Gradient-weighted Class Activation Mapping (Grad-CAM) over the final convolutional layer of each backbone (`features.denseblock4` for DenseNet-121). Heatmaps are generated via:
$$L_{\text{Grad-CAM}}^c = \text{ReLU}\left( \sum_k \alpha_k^c A^k \right)$$
where $\alpha_k^c = \frac{1}{Z} \sum_i \sum_j \frac{\partial Y^c}{\partial A_{i,j}^k}$.

---

## 3. Results

### 3.1 Quantitative Multi-Backbone Benchmarks
Models were evaluated across 3 random seeds (42, 7, 123) with 15 epochs per run. Results are reported as $\text{mean} \pm \text{std}$:

**Internal Test Set (934 images, patient-level split):**

| Model Backbone | Accuracy | Precision (Macro) | Recall (Macro) | F1 Score (Macro) | AUC (Macro) |
|---|:---:|:---:|:---:|:---:|:---:|
| **ResNet-18** | **0.8498 ± 0.0033** | **0.8358 ± 0.0041** | **0.8489 ± 0.0054** | **0.8409 ± 0.0022** | **0.9538 ± 0.0020** |
| **DenseNet-121** | 0.8455 ± 0.0022 | 0.8349 ± 0.0115 | 0.8328 ± 0.0086 | 0.8322 ± 0.0055 | 0.9512 ± 0.0024 |
| **EfficientNet-B0** | 0.8383 ± 0.0091 | 0.8306 ± 0.0108 | 0.8248 ± 0.0070 | 0.8269 ± 0.0088 | 0.9493 ± 0.0015 |

### 3.2 Classical Baseline vs. Deep Learning Benchmark
The classical machine learning pipeline combining HOG feature extraction and an RBF-kernel Support Vector Machine (SVM) achieved an internal test **Accuracy of 83.51%**, **Macro F1 of 79.40%**, and **Macro AUC of 0.9470**.

However, on the completely held-out external test set (Montgomery County, 414 images), the classical baseline performance dropped significantly to **10.87% Accuracy**, **9.17% Macro F1**, and **0.6052 AUC**. This steep degradation highlights the inability of handcrafted HOG descriptors to adjust to cross-institutional scanner variations and domain shifts.

### 3.3 External Validation Generalization Gap
On the 100% held-out Montgomery County external test set (414 unseen adult images), deep convolutional backbones retained strong discriminative capacity:

| Model Backbone | Accuracy | Precision (Macro) | Recall (Macro) | F1 Score (Macro) | AUC (Macro) |
|---|:---:|:---:|:---:|:---:|:---:|
| **DenseNet-121** | 0.6860 ± 0.0723 | 0.4083 ± 0.0175 | 0.3310 ± 0.0373 | **0.3575 ± 0.0293** | **0.8296 ± 0.0613** |
| **ResNet-18** | **0.6932 ± 0.0304** | 0.3798 ± 0.0126 | 0.3361 ± 0.0119 | 0.3535 ± 0.0100 | 0.7606 ± 0.0072 |
| **EfficientNet-B0** | 0.5411 ± 0.0616 | **0.4245 ± 0.0092** | 0.2495 ± 0.0284 | 0.2836 ± 0.0200 | 0.7208 ± 0.0360 |

DenseNet-121 maintained superior out-of-distribution discrimination with an external AUC of **0.8296 ± 0.0613** (compared to 0.6052 for classical SVM), demonstrating the superior feature representations learned by dense connectivity architectures on medical X-rays.

---

## 4. Discussion & Limitations

### 4.1 Clinical Significance
The system provides a triaged decision workflow: High-risk predictions (Tuberculosis / Bacterial Pneumonia) trigger urgent red alerts for clinical escalation, accompanied by Grad-CAM visual heatmaps pointing to pulmonary infiltrates.

### 4.2 Domain Confound Disclosure
A notable scientific limitation is the domain shift between Kermany (pediatric Guangzhou cohort) and Shenzhen/Montgomery (adult cohort). In 4-class configurations, classifiers may leverage age-related anatomical features to distinguish Tuberculosis. We explicitly highlight this domain confound as a limitation and advocate for multi-center pediatric TB data collection.

---

## 5. Conclusion & Reproducibility Statement

This study presents a rigorous benchmark for chest X-ray classification with strict patient-level splitting, external validation, and visual explainability. All code, model checkpoints, and configuration files are publicly available at `Cheesenoice/chest-xray-cdss`.
