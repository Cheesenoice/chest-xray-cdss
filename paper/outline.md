# Paper Outline (IMRaD)

## Title
*An Explainable Deep Learning Clinical Decision Support System for Chest X-ray Screening*

## Abstract
- Problem: need for reproducible, explainable, externally-validated benchmark in chest X-ray AI
- Methods: 3-4 backbones, patient-level split, Grad-CAM, external held-out validation
- Results: internal AUC X.XX, external AUC X.XX
- Conclusion: reproducible benchmark with honest generalization gap

## 1. Introduction
- AI for chest X-ray: progress but reproducibility crisis
- Gap: over-reliance on single datasets, lack of external validation, missing explainability
- Our contributions: reproducible benchmark + XAI + external validation + fair comparison

## 2. Related Work
- CNN architectures for chest X-ray (ChetXNet, COVID-Net, etc.)
- Explainability in medical imaging (Grad-CAM variants)
- Dataset leakage issues in published literature

## 3. Methods
### 3.1 Datasets
- Kermany (pediatric, CC BY 4.0)
- Shenzhen / Montgomery (adult, public domain)
- Patient-level split, MD5 dedup, held-out Montgomery

### 3.2 Model Architectures
- ResNet-18, DenseNet-121, EfficientNet-B0
- Transfer learning from ImageNet

### 3.3 Training Protocol
- AdamW, cosine scheduler, early stopping
- Class-balanced sampling
- Fixed seeds for reproducibility

### 3.4 Explainability
- Grad-CAM on last convolutional layer
- Qualitative assessment of heatmap localization

### 3.5 Classical Baseline
- HOG / LBP features + SVM / Logistic Regression

### 3.6 Evaluation Metrics
- Accuracy, Precision, Recall, F1 (macro), AUC (OvR)
- 95% CI via bootstrap, mean ± std over 3 seeds

## 4. Results
- Internal test metrics table
- External validation metrics table
- Confusion matrix, ROC curves
- Grad-CAM examples (correct and incorrect)
- Classical baseline comparison

## 5. Discussion
- Internal performance vs literature
- External validation generalization gap
- Domain confound (pediatric vs adult)
- Clinical applicability and limitations

## 6. Limitations
- Domain confound (4-class)
- Single-source pneumonia data
- No real clinical validation
- Limited to frontal X-rays

## 7. Conclusion
- Reproducible benchmark contributed
- Code, data splits, and model weights publicly available
- Future work: multi-center validation, additional pathologies

## Reproducibility Statement
All code, configurations, and seed values are available at [repo URL]. Experiments run on [hardware] with [library versions].

## References
...
