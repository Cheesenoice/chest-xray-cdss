# Model Card: Chest X-Ray Clinical Decision Support System (CDSS)

## Model Details
- **Model Name:** Explainable Deep Learning Chest X-Ray Classifier
- **Model Architectures:** DenseNet-121, ResNet-18, EfficientNet-B0
- **Version:** 1.0 (2026)
- **Framework:** PyTorch 2.x, timm, albumentations, pytorch-grad-cam
- **License:** Open Source for Academic & Research Use

## Intended Use
- **Primary Intended Use:** Triage and clinical decision support for chest X-ray screening (Normal, Bacterial Pneumonia, Viral Pneumonia, Tuberculosis).
- **Intended Users:** Medical researchers, healthcare practitioners, and primary-care clinicians.
- **Out of Scope Use:** Standalone automated diagnosis without physician oversight; self-diagnosis by patients.

## Training & Evaluation Data
- **Primary Training Data:** Kermany Chest X-Ray Dataset (Pediatric, Guangzhou) + Shenzhen Hospital Dataset (Adult, China).
- **External Evaluation Data:** Montgomery County Dataset (Adult, Maryland, USA - 100% held-out).
- **Data Splitting:** Strict patient-level partitioning (`GroupShuffleSplit` on `patient_id`) with verified zero patient overlap and zero MD5 hash overlap.

## Ethical Considerations & Limitations
1. **Domain Confound:** Pediatric (Pneumonia) vs. Adult (TB) demographic differences present in public data.
2. **Regulatory Status:** This system is an academic research prototype and is NOT FDA/CE-marked for diagnostic use.
