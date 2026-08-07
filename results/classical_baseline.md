# Classical ML Baseline Benchmark (HOG + SVM)

- **Feature Extractor:** HOG (orientations=9, pixels_per_cell=16x16) + Intensity Histogram (32 bins)
- **Classifier:** Support Vector Machine (RBF Kernel)
- **Random Seed:** 42

## Performance Metrics

| Evaluation Set | Accuracy | Precision (Macro) | Recall (Macro) | F1 (Macro) | AUC (Macro) |
|---|---|---|---|---|---|
| **Internal Test Set** | 0.8351 | 0.8098 | 0.7833 | 0.7940 | 0.9470 |
| **External Test Set (Montgomery)** | 0.1087 | 0.1724 | 0.0625 | 0.0917 | 0.6052 |
