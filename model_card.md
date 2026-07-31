# Model Card — Chest X-ray CDSS

## Model Description
**Purpose:** Multi-class classification of chest X-rays for clinical decision support.
**Architecture:** Transfer learning with ImageNet-pretrained backbones (ResNet-18, DenseNet-121, EfficientNet-B0) via `timm`.
**Task:** 3-class (Normal, Bacterial pneumonia, Viral pneumonia) or 4-class (+ Tuberculosis).

## Intended Use
- Screening triage in primary-care settings lacking radiologists
- Research and educational purposes only
- **NOT** for autonomous clinical diagnosis

## Training Data
| Source | Domain | Classes | License |
|--------|--------|---------|---------|
| Kermany (Chest X-Ray Pneumonia) | Pediatric | Normal, Bacterial, Viral | CC BY 4.0 |
| Shenzhen (Pulmonary Abnormalities) | Adult | Normal, TB | Public domain (NLM/NIH) |
| Montgomery (Pulmonary Abnormalities) | Adult | Normal, TB (held-out external test) | Public domain (NLM/NIH) |

## Evaluation Results
Reported on held-out test set with patient-level split (no leakage). Full results: `results/`.

## Limitations
- Pediatric vs adult domain confound in 4-class setting
- Limited to frontal chest X-rays
- Not validated in real clinical settings
- Single (Kermany) source for pneumonia classes

## Ethical Considerations
- This system supports, not replaces, physician judgment.
- All datasets are de-identified and publicly available.
- Potential demographic biases not fully characterized.
