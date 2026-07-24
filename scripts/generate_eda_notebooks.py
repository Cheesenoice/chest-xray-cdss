import nbformat as nbf
from pathlib import Path

def create_kermany_notebook(output_path):
    nb = nbf.v4.new_notebook()
    cells = []
    
    # Title Markdown
    cells.append(nbf.v4.new_markdown_cell("""# 📊 Exploratory Data Analysis: Kermany Chest X-Ray Dataset (Pneumonia)

> **Dataset Reference:** Kermany et al. 2018 (Guangzhou Women and Children's Medical Center)  
> **Source:** Kaggle `paultimothymooney/chest-xray-pneumonia`  
> **License:** Creative Commons Attribution 4.0 International (CC BY 4.0)

---

## 🎯 Purpose & Dataset Usage in Graduation Thesis

1. **Core Training Source:** Primary data source for 3 major classes: `Normal`, `Bacterial Pneumonia`, and `Viral Pneumonia`.
2. **Clinical Narrative:** Represents **pediatric chest X-rays** (patients aged 1 to 5 years). Distinguishing bacterial from viral pneumonia is crucial in pediatric care to prevent unnecessary antibiotic prescriptions.
3. **Sub-label Inference:** Filenames contain `bacteria` or `virus` substrings, allowing 3-class granularity rather than binary classification.
"""))

    # Imports & Setup
    cells.append(nbf.v4.new_code_cell("""import os
import re
import glob
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import cv2

# Plotting settings
sns.set_theme(style="whitegrid")
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['figure.dpi'] = 120

DATA_DIR = Path("../data/raw/kermany")
print(f"Data directory exists: {DATA_DIR.exists()}")
"""))

    # Step 1: Scan & Parse
    cells.append(nbf.v4.new_markdown_cell("""## 1. File Scanning & Metadata Extraction

We scan all `.jpeg` images, exclude macOS metadata (`__MACOSX` / `._*`), and parse:
- Class label (`normal`, `bacterial_pneumonia`, `viral_pneumonia`)
- Patient ID derived from `person{N}` filename pattern
"""))

    cells.append(nbf.v4.new_code_cell("""raw_images = [
    p for p in DATA_DIR.rglob("*.[jJ][pP]*[gG]")
    if "__MACOSX" not in str(p) and not p.name.startswith("._")
]

records = []
for p in raw_images:
    fn = p.name
    parent = p.parent.name.upper()
    
    if "NORMAL" in parent or "IM-" in fn:
        label = "normal"
        patient_id = f"kermany_norm_{p.stem}"
    elif "PNEUMONIA" in parent or "bacteria" in fn or "virus" in fn:
        if "bacteria" in fn.lower():
            label = "bacterial_pneumonia"
        elif "virus" in fn.lower():
            label = "viral_pneumonia"
        else:
            label = "bacterial_pneumonia"
        
        m = re.search(r"person(\\d+)", fn, re.IGNORECASE)
        patient_id = f"person{m.group(1)}" if m else f"kermany_pneu_{p.stem}"
    else:
        continue
        
    records.append({
        "filepath": str(p),
        "filename": fn,
        "label": label,
        "patient_id": patient_id
    })

df = pd.DataFrame(records)
print(f"Total scanned images: {len(df)}")
print(f"Unique patient groups: {df['patient_id'].nunique()}")
df.head()
"""))

    # Step 2: Class Distribution
    cells.append(nbf.v4.new_markdown_cell("""## 2. Class Distribution Analysis

Let's visualize the balance between `Normal`, `Bacterial Pneumonia`, and `Viral Pneumonia`.
"""))

    cells.append(nbf.v4.new_code_cell("""fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Bar Chart
class_counts = df['label'].value_counts()
colors = ['#2b5c8f', '#d95f02', '#7570b3']
sns.barplot(x=class_counts.index, y=class_counts.values, ax=axes[0], hue=class_counts.index, palette=colors, legend=False)
axes[0].set_title("Image Count by Class", fontsize=14, fontweight='bold')
axes[0].set_ylabel("Number of Images")
for i, v in enumerate(class_counts.values):
    axes[0].text(i, v + 30, str(v), ha='center', fontweight='bold')

# Pie Chart
axes[1].pie(class_counts.values, labels=class_counts.index, autopct='%1.1f%%', colors=colors, startangle=140, explode=(0.03, 0.03, 0.03))
axes[1].set_title("Class Proportions", fontsize=14, fontweight='bold')

plt.tight_layout()
plt.show()
"""))

    # Step 3: Image Dimensions
    cells.append(nbf.v4.new_markdown_cell("""## 3. Image Dimensions & Aspect Ratio Analysis

Medical X-rays come in varying resolutions. We inspect width, height, and aspect ratios.
"""))

    cells.append(nbf.v4.new_code_cell("""widths, heights, aspect_ratios = [], [], []

for p in df['filepath'].sample(min(1000, len(df)), random_state=42):
    with Image.open(p) as img:
        w, h = img.size
        widths.append(w)
        heights.append(h)
        aspect_ratios.append(w / h)

df_dim = pd.DataFrame({"width": widths, "height": heights, "aspect_ratio": aspect_ratios})

fig, axes = plt.subplots(1, 3, figsize=(16, 4))
sns.histplot(df_dim['width'], ax=axes[0], color='#2b5c8f', kde=True)
axes[0].set_title("Image Width Distribution")

sns.histplot(df_dim['height'], ax=axes[1], color='#d95f02', kde=True)
axes[1].set_title("Image Height Distribution")

sns.histplot(df_dim['aspect_ratio'], ax=axes[2], color='#7570b3', kde=True)
axes[2].set_title("Aspect Ratio (W/H)")
axes[2].axvline(1.0, color='red', linestyle='--', label='Square (1.0)')
axes[2].legend()

plt.tight_layout()
plt.show()

print(f"Width - Min: {min(widths)}, Max: {max(widths)}, Median: {np.median(widths):.0f}")
print(f"Height - Min: {min(heights)}, Max: {max(heights)}, Median: {np.median(heights):.0f}")
"""))

    # Step 4: Sample Visualizations per Class
    cells.append(nbf.v4.new_markdown_cell("""## 4. Visualizing X-Ray Images per Class

We display representative chest X-rays from each class.
- **Normal:** Clear lung fields, sharp diaphragm boundaries.
- **Bacterial Pneumonia:** Focal/lobar consolidation (dense white opacities in specific lung lobes).
- **Viral Pneumonia:** Diffuse interstitial patterns (bilateral patchiness).
"""))

    cells.append(nbf.v4.new_code_cell("""fig, axes = plt.subplots(3, 4, figsize=(16, 12))

for row_idx, label in enumerate(['normal', 'bacterial_pneumonia', 'viral_pneumonia']):
    samples = df[df['label'] == label].sample(4, random_state=42).reset_index()
    for col_idx in range(4):
        ax = axes[row_idx, col_idx]
        img_path = samples.iloc[col_idx]['filepath']
        img = Image.open(img_path).convert('L')
        ax.imshow(img, cmap='gray')
        pid = samples.iloc[col_idx]['patient_id']
        ax.set_title(f"{label.upper()} \\n Patient: {pid}", fontsize=10)
        ax.axis('off')

plt.suptitle("Representative Pediatric Chest X-Rays by Diagnosis", fontsize=16, fontweight='bold', y=0.98)
plt.tight_layout()
plt.show()
"""))

    # Step 5: Pixel Intensity Distributions
    cells.append(nbf.v4.new_markdown_cell("""## 5. Pixel Intensity Distribution Analysis (Fast Histograms)

Comparing average pixel intensity distributions across classes.
"""))

    cells.append(nbf.v4.new_code_cell("""plt.figure(figsize=(10, 5))

bins = np.arange(257)
bin_centers = (bins[:-1] + bins[1:]) / 2

for label, color in zip(['normal', 'bacterial_pneumonia', 'viral_pneumonia'], ['#2b5c8f', '#d95f02', '#7570b3']):
    sample_paths = df[df['label'] == label]['filepath'].sample(min(150, len(df[df['label'] == label])), random_state=42)
    hists = []
    for p in sample_paths:
        img = np.array(Image.open(p).convert('L').resize((224, 224)))
        counts, _ = np.histogram(img, bins=bins)
        hists.append(counts / counts.sum()) # normalize
    
    avg_hist = np.mean(hists, axis=0)
    plt.plot(bin_centers, avg_hist, label=label, color=color, linewidth=2.5)

plt.title("Normalized Mean Pixel Intensity Distribution by Class", fontsize=14, fontweight='bold')
plt.xlabel("Pixel Value (0 = Dark, 255 = Bright)")
plt.ylabel("Normalized Frequency")
plt.xlim(0, 255)
plt.legend()
plt.tight_layout()
plt.show()
"""))

    # Conclusion & Key Takeaways
    cells.append(nbf.v4.new_markdown_cell("""## 📌 Summary & Key Takeaways for Project Execution

| Aspect | Observation / Technical Insight |
| :--- | :--- |
| **Data Imbalance** | Bacterial Pneumonia is the majority class (~47%), Normal (~27%), Viral (~25%). Requires class weighting / WeightedRandomSampler. |
| **Patient Leakage** | Patient IDs follow `person{N}`. Split logic MUST group by `patient_id` so same child never appears in train and test. |
| **Anatomical Domain** | Pediatric dataset (1-5 yrs old) -> Smaller thoracic cavity, heart occupies larger ratio of chest than adults. |
| **Pre-processing** | All images resized to `224x224` & normalized to ImageNet mean/std for transfer learning backbones (`DenseNet-121`, `ResNet-18`). |
"""))

    nb['cells'] = cells
    with open(output_path, 'w', encoding='utf-8') as f:
        nbf.write(nb, f)
    print(f"Created {output_path}")

def create_pulmonary_notebook(output_path):
    nb = nbf.v4.new_notebook()
    cells = []

    cells.append(nbf.v4.new_markdown_cell("""# 📊 Exploratory Data Analysis: Pulmonary Abnormalities (Shenzhen & Montgomery TB Datasets)

> **Datasets Included:**
> 1. **Shenzhen No.3 Hospital Chest X-ray Set:** 662 images (336 TB, 326 Normal)
> 2. **Montgomery County Chest X-ray Set:** 138 images (58 TB, 80 Normal)
> **Source:** Kaggle `kmader/pulmonary-chest-xray-abnormalities`  
> **License:** Public Domain / NIH / NLM (National Library of Medicine)

---

## 🎯 Purpose & Dataset Usage in Graduation Thesis

1. **Adult Tuberculosis (TB) Class Integration:** Used to construct the 4th class (`Tuberculosis`) for the CDSS demo system.
2. **Honest External Validation Strategy:**
   - **Shenzhen Dataset:** Used for **training & validation** of the Tuberculosis class.
   - **Montgomery Dataset:** Held out **100% as an External Test Set**. Model never sees Montgomery images during training, allowing true evaluation of generalization capability across medical centers.
"""))

    cells.append(nbf.v4.new_code_cell("""import os
import re
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

sns.set_theme(style="whitegrid")
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['figure.dpi'] = 120

DATA_DIR = Path("../data/raw/pulmonary_abnormalities")
print(f"Data directory exists: {DATA_DIR.exists()}")
"""))

    cells.append(nbf.v4.new_markdown_cell("""## 1. File Scanning & Parsing (Shenzhen vs Montgomery)

- **Shenzhen:** Filename `CHNCXR_{id}_{label}.png` (`0`=Normal, `1`=TB) -> Patient ID `CHN_{id}`
- **Montgomery:** Filename `MCUCXR_{id}_{label}.png` (`0`=Normal, `1`=TB) -> Patient ID `MCU_{id}`
"""))

    cells.append(nbf.v4.new_code_cell("""raw_images = [
    p for p in DATA_DIR.rglob("*.[pP][nN][gG]")
    if "__MACOSX" not in str(p) and not p.name.startswith("._")
]

records = []
for p in raw_images:
    fn = p.name
    if fn.startswith("CHNCXR"):
        parts = p.stem.split("_")
        if len(parts) >= 3:
            label = "tuberculosis" if parts[2] == "1" else "normal"
            records.append({
                "filepath": str(p),
                "source": "shenzhen",
                "label": label,
                "patient_id": f"CHN_{parts[1]}"
            })
    elif fn.startswith("MCUCXR"):
        parts = p.stem.split("_")
        if len(parts) >= 3:
            label = "tuberculosis" if parts[2] == "1" else "normal"
            records.append({
                "filepath": str(p),
                "source": "montgomery",
                "label": label,
                "patient_id": f"MCU_{parts[1]}"
            })

df = pd.DataFrame(records)
print(f"Total scanned images: {len(df)}")
print(df['source'].value_counts())
df.head()
"""))

    cells.append(nbf.v4.new_markdown_cell("""## 2. Source & Label Breakdown Comparison"""))

    cells.append(nbf.v4.new_code_cell("""plt.figure(figsize=(10, 5))
ax = sns.countplot(data=df, x="source", hue="label", palette=["#2b5c8f", "#e74c3c"])
plt.title("Label Counts by Dataset Source (Shenzhen vs Montgomery)", fontsize=14, fontweight='bold')
plt.xlabel("Dataset Source")
plt.ylabel("Number of X-Rays")

for p in ax.patches:
    height = p.get_height()
    if height > 0:
        ax.annotate(f'{int(height)}', (p.get_x() + p.get_width() / 2., height + 5), ha='center', fontweight='bold')

plt.tight_layout()
plt.show()
"""))

    cells.append(nbf.v4.new_markdown_cell("""## 3. High-Resolution Scans Analysis

Shenzhen and Montgomery datasets contain high-resolution adult DICOM-converted PNG scans.
"""))

    cells.append(nbf.v4.new_code_cell("""widths, heights = [], []
for p in df['filepath'].sample(min(500, len(df)), random_state=42):
    with Image.open(p) as img:
        w, h = img.size
        widths.append(w)
        heights.append(h)

df_dim = pd.DataFrame({"width": widths, "height": heights})

plt.figure(figsize=(8, 5))
sns.scatterplot(data=df_dim, x="width", y="height", color="#8e44ad", alpha=0.7)
plt.title("Resolution Distribution of Adult X-Rays (Width vs Height)", fontsize=14, fontweight='bold')
plt.xlabel("Width (pixels)")
plt.ylabel("Height (pixels)")
plt.show()

print(f"Mean Resolution: {np.mean(widths):.0f} x {np.mean(heights):.0f} pixels")
"""))

    cells.append(nbf.v4.new_markdown_cell("""## 4. Visual Comparison: Normal vs Tuberculosis

Tuberculosis lesions typically manifest in apical/upper lung zones as cavitating lesions, infiltrates, or pleural effusions.
"""))

    cells.append(nbf.v4.new_code_cell("""fig, axes = plt.subplots(2, 4, figsize=(16, 8))

# Shenzhen TB vs Normal
for i, label in enumerate(['normal', 'tuberculosis']):
    samples = df[(df['source'] == 'shenzhen') & (df['label'] == label)].sample(2, random_state=42).reset_index()
    for j in range(2):
        ax = axes[i, j]
        img = Image.open(samples.iloc[j]['filepath']).convert('L')
        ax.imshow(img, cmap='gray')
        pid = samples.iloc[j]['patient_id']
        ax.set_title(f"SHENZHEN | {label.upper()} \\n Patient: {pid}", fontsize=9)
        ax.axis('off')

# Montgomery TB vs Normal
for i, label in enumerate(['normal', 'tuberculosis']):
    samples = df[(df['source'] == 'montgomery') & (df['label'] == label)].sample(2, random_state=42).reset_index()
    for j in range(2):
        ax = axes[i, j + 2]
        img = Image.open(samples.iloc[j]['filepath']).convert('L')
        ax.imshow(img, cmap='gray')
        pid = samples.iloc[j]['patient_id']
        ax.set_title(f"MONTGOMERY | {label.upper()} \\n Patient: {pid}", fontsize=9)
        ax.axis('off')

plt.suptitle("Adult Chest X-Rays: Normal vs Tuberculosis (Shenzhen & Montgomery)", fontsize=15, fontweight='bold')
plt.tight_layout()
plt.show()
"""))

    cells.append(nbf.v4.new_markdown_cell("""## 📌 Summary & Key Takeaways for Project Execution

| Aspect | Insight / Strategy |
| :--- | :--- |
| **External Validation Isolation** | All Montgomery images (`MCUCXR_*`) are strictly held out for external validation (`external_test.csv`). |
| **Adult Anatomy** | Adult ribcages are larger, vertical, with elongated lung fields compared to pediatric Kermany X-rays. |
| **Resolution Standardization** | High-res scans (~3000x3000) are downsampled to 224x224 for uniform model input. |
"""))

    nb['cells'] = cells
    with open(output_path, 'w', encoding='utf-8') as f:
        nbf.write(nb, f)
    print(f"Created {output_path}")

def create_consolidated_notebook(output_path):
    nb = nbf.v4.new_notebook()
    cells = []

    cells.append(nbf.v4.new_markdown_cell("""# 📊 Exploratory Data Analysis: Consolidated 4-Class Dataset & Domain Confound Analysis

> **Classes:** `Normal`, `Bacterial Pneumonia`, `Viral Pneumonia`, `Tuberculosis`  
> **Manifest File:** `data/processed/manifest.csv`  
> **Split Directory:** `data/processed/splits/`

---

## 🎯 Purpose & Research Context

1. **Integrated 4-Class Pipeline:** Combines Kermany (pediatric pneumonia) + Shenzhen (adult TB) + Montgomery (adult held-out external test).
2. **Domain Confound Analysis (Mandatory Thesis Requirement):**
   - **Pediatric (Kermany):** Normal, Bacterial, Viral classes.
   - **Adult (Shenzhen & Montgomery):** Tuberculosis class.
   - **Confound Risk:** The model could potentially learn "child vs adult anatomy" rather than "pneumonia vs tuberculosis pathology".
   - **Mitigation:** Transparent disclosure in paper/defense + benchmark clean 3-class Kermany model alongside 4-class demo model + external test validation.
"""))

    cells.append(nbf.v4.new_code_cell("""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from pathlib import Path

sns.set_theme(style="whitegrid")
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['figure.dpi'] = 120

MANIFEST_PATH = Path("../data/processed/manifest.csv")
SPLIT_DIR = Path("../data/processed/splits")

df_manifest = pd.read_csv(MANIFEST_PATH)
df_train = pd.read_csv(SPLIT_DIR / "train.csv")
df_val = pd.read_csv(SPLIT_DIR / "val.csv")
df_test = pd.read_csv(SPLIT_DIR / "test.csv")
df_ext = pd.read_csv(SPLIT_DIR / "external_test.csv")

print(f"Manifest total unique images: {len(df_manifest)}")
"""))

    cells.append(nbf.v4.new_markdown_cell("""## 1. Consolidated Manifest Breakdown"""))

    cells.append(nbf.v4.new_code_cell("""fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# By Label
label_counts = df_manifest['label'].value_counts()
sns.barplot(x=label_counts.index, y=label_counts.values, ax=axes[0], hue=label_counts.index, palette="Blues_r", legend=False)
axes[0].set_title("Overall Manifest Image Count by Diagnosis Label", fontsize=13, fontweight='bold')
axes[0].tick_params(axis='x', rotation=15)
for p in axes[0].patches:
    axes[0].annotate(f"{int(p.get_height())}", (p.get_x() + p.get_width() / 2., p.get_height() + 30), ha='center', fontweight='bold')

# By Source
source_counts = df_manifest['source'].value_counts()
sns.barplot(x=source_counts.index, y=source_counts.values, ax=axes[1], hue=source_counts.index, palette="Dark2", legend=False)
axes[1].set_title("Image Count by Dataset Source", fontsize=13, fontweight='bold')
for p in axes[1].patches:
    axes[1].annotate(f"{int(p.get_height())}", (p.get_x() + p.get_width() / 2., p.get_height() + 30), ha='center', fontweight='bold')

plt.tight_layout()
plt.show()
"""))

    cells.append(nbf.v4.new_markdown_cell("""## 2. Train / Val / Test / External Split Proportions"""))

    cells.append(nbf.v4.new_code_cell("""split_data = []
for name, d in [("Train", df_train), ("Val", df_val), ("Test (Internal)", df_test), ("External Test (Montgomery)", df_ext)]:
    for label, count in d['label'].value_counts().items():
        split_data.append({"Split": name, "Label": label, "Count": count})

df_splits = pd.DataFrame(split_data)

plt.figure(figsize=(12, 6))
sns.barplot(data=df_splits, x="Split", y="Count", hue="Label", palette="Set2")
plt.title("Class Distribution Across Patient-Level Data Splits", fontsize=14, fontweight='bold')
plt.ylabel("Number of Images")
plt.legend(title="Diagnosis")
plt.show()
"""))

    cells.append(nbf.v4.new_markdown_cell("""## 3. Pediatric vs Adult Domain Confound Visual Comparison

We plot side-by-side pediatric X-ray (Kermany) vs adult X-ray (Shenzhen/Montgomery) to visualize anatomical differences.
"""))

    cells.append(nbf.v4.new_code_cell("""fig, axes = plt.subplots(1, 2, figsize=(12, 6))

pediatric_sample = df_manifest[df_manifest['domain'] == 'pediatric'].sample(1, random_state=42).iloc[0]
adult_sample = df_manifest[df_manifest['domain'] == 'adult'].sample(1, random_state=42).iloc[0]

img_ped = Image.open(pediatric_sample['filepath']).convert('L')
img_adult = Image.open(adult_sample['filepath']).convert('L')

axes[0].imshow(img_ped, cmap='gray')
ped_source = pediatric_sample['source'].upper()
ped_label = pediatric_sample['label']
axes[0].set_title(f"PEDIATRIC DOMAIN ({ped_source}) \\n Label: {ped_label}", fontsize=12, fontweight='bold')
axes[0].axis('off')

axes[1].imshow(img_adult, cmap='gray')
adult_source = adult_sample['source'].upper()
adult_label = adult_sample['label']
axes[1].set_title(f"ADULT DOMAIN ({adult_source}) \\n Label: {adult_label}", fontsize=12, fontweight='bold')
axes[1].axis('off')

plt.suptitle("Domain Confound Visualization: Pediatric vs Adult Chest Geometry", fontsize=15, fontweight='bold')
plt.tight_layout()
plt.show()
"""))

    cells.append(nbf.v4.new_markdown_cell("""## 📌 Thesis Defense & Research Paper Strategy

| Problem | Scientific Mitigation | Presentation in Thesis / Paper |
| :--- | :--- | :--- |
| **Domain Confound (Pediatric vs Adult)** | Run 3-class clean benchmark (Kermany only) in parallel with 4-class model. | Disclosed explicitly in Limitations section. Highlighted as reason for external validation. |
| **Class Imbalance** | Apply `WeightedRandomSampler` during PyTorch training. | Reported via macro F1-score & AUC (not just overall accuracy). |
| **External Generalization** | Validate on 100% held-out Montgomery source. | Honest drop in performance reported & discussed. |
"""))

    nb['cells'] = cells
    with open(output_path, 'w', encoding='utf-8') as f:
        nbf.write(nb, f)
    print(f"Created {output_path}")

if __name__ == "__main__":
    notebooks_dir = Path("notebooks")
    notebooks_dir.mkdir(parents=True, exist_ok=True)
    
    create_kermany_notebook(notebooks_dir / "01_eda_kermany.ipynb")
    create_pulmonary_notebook(notebooks_dir / "02_eda_pulmonary.ipynb")
    create_consolidated_notebook(notebooks_dir / "03_eda_consolidated.ipynb")
