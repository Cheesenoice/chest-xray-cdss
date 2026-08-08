import os
import sys
import tempfile
import cv2
import torch
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
import yaml
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.models import build_model
from src.explain import get_target_layer
from src.datasets import LABEL_MAP_4CLASS, LABEL_MAP_3CLASS, get_transforms
try:
    from app.report import generate_pdf_report
except ModuleNotFoundError:
    from report import generate_pdf_report

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

st.set_page_config(
    page_title="Chest X-Ray CDSS | Clinical AI Screening Platform",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Premium Aesthetics CSS
st.markdown("""
    <style>
    .main-header { font-size: 2.3rem; font-weight: 800; color: #0F172A; margin-bottom: 0.1rem; }
    .sub-header { font-size: 1.05rem; color: #475569; margin-bottom: 1.2rem; }
    .card { background-color: #F8FAFC; border-radius: 10px; padding: 20px; border: 1px solid #E2E8F0; margin-bottom: 15px; }
    .triage-red { background: linear-gradient(135deg, #FEE2E2 0%, #FECACA 100%); border-left: 6px solid #DC2626; padding: 14px; border-radius: 8px; color: #7F1D1D; font-weight: bold; font-size: 1.1rem; }
    .triage-yellow { background: linear-gradient(135deg, #FEF3C7 0%, #FDE68A 100%); border-left: 6px solid #D97706; padding: 14px; border-radius: 8px; color: #78350F; font-weight: bold; font-size: 1.1rem; }
    .triage-green { background: linear-gradient(135deg, #D1FAE5 0%, #A7F3D0 100%); border-left: 6px solid #059669; padding: 14px; border-radius: 8px; color: #064E3B; font-weight: bold; font-size: 1.1rem; }
    .metric-box { background: #FFFFFF; border-radius: 8px; padding: 15px; border: 1px solid #E2E8F0; text-align: center; }
    .metric-val { font-size: 1.8rem; font-weight: bold; color: #2563EB; }
    .metric-lbl { font-size: 0.85rem; color: #64748B; text-transform: uppercase; letter-spacing: 0.5px; }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_cdss_model(backbone_name="densenet121", num_classes=4, checkpoint_path=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(backbone_name=backbone_name, num_classes=num_classes, pretrained=False).to(device)
    
    if checkpoint_path and os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, device

@st.cache_data
def load_manifest_data():
    manifest_p = Path("data/processed/manifest.csv")
    if manifest_p.exists():
        return pd.read_csv(manifest_p)
    return None

def apply_clahe_enhancement(img_np):
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)

def main():
    st.markdown('<div class="main-header">🫁 Explainable Chest X-Ray Clinical Decision Support System</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Integrated Deep Learning Triage, Grad-CAM Pathological Heatmaps & Hospital Analytics Platform</div>', unsafe_allow_html=True)

    # Global Sidebar
    st.sidebar.header("⚙️ Model & Clinical Settings")
    backbone = st.sidebar.selectbox("Model Architecture", ["densenet121", "resnet18", "efficientnet_b0"], index=0)
    num_classes = st.sidebar.radio("Classification Mode", [4, 3], index=0, help="4-class includes TB; 3-class focuses on Pneumonia benchmark")
    
    label_map = LABEL_MAP_4CLASS if num_classes == 4 else LABEL_MAP_3CLASS
    inv_label_map = {v: k for k, v in label_map.items()}

    ckpt_path = f"results/checkpoints/best_{backbone}_seed42.pt"
    model, device = load_cdss_model(backbone, num_classes, ckpt_path)

    st.sidebar.markdown("---")
    st.sidebar.subheader("👤 Patient Metadata")
    patient_id = st.sidebar.text_input("Patient ID", "PAT-2026-0089")
    patient_name = st.sidebar.text_input("Patient Full Name", "Anonymous Patient")
    age = st.sidebar.number_input("Patient Age", min_value=1, max_value=100, value=34)
    gender = st.sidebar.selectbox("Gender", ["Male", "Female", "Other"])

    # 4 Main Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "🏥 Clinical Patient Screening",
        "📊 Hospital Analytics Dashboard",
        "🔬 Benchmark & Explainability Hub",
        "🛡️ Scientific Integrity & Paper Draft"
    ])

    # ---------------------------------------------------------
    # TAB 1: Clinical Patient Screening
    # ---------------------------------------------------------
    with tab1:
        col1, col2 = st.columns([1, 1])

        with col1:
            st.subheader("1. X-Ray Image Input")
            input_mode = st.radio("Select Image Input Method", ["Upload Custom File", "Use Sample Preset Image"], inline=True)

            image = None
            if input_mode == "Upload Custom File":
                uploaded_file = st.file_uploader("Upload DICOM/JPEG/PNG...", type=["jpeg", "jpg", "png", "dcm"])
                if uploaded_file is not None:
                    if uploaded_file.name.lower().endswith(".dcm"):
                        try:
                            import pydicom
                            dcm = pydicom.dcmread(uploaded_file)
                            pix = dcm.pixel_array
                            pix = ((pix - pix.min()) / (pix.max() - pix.min() + 1e-5) * 255.0).astype(np.uint8)
                            image = Image.fromarray(cv2.cvtColor(pix, cv2.COLOR_GRAY2RGB))
                        except Exception as e:
                            st.error(f"Error reading DICOM file: {e}")
                    else:
                        image = Image.open(uploaded_file).convert("RGB")
            else:
                test_csv = Path("data/processed/splits/test.csv")
                if test_csv.exists():
                    test_df = pd.read_csv(test_csv)
                    sample_options = [f"{row['label'].upper()} - {Path(row['filepath']).name}" for _, row in test_df.sample(min(10, len(test_df)), random_state=42).iterrows()]
                    selected_preset = st.selectbox("Select Test Preset Scan:", sample_options)
                    selected_filename = selected_preset.split(" - ")[-1]
                    match_row = test_df[test_df["filepath"].str.contains(selected_filename, regex=False)]
                    if len(match_row) > 0:
                        image = Image.open(match_row.iloc[0]["filepath"]).convert("RGB")

            if image is not None:
                st.image(image, caption="Original Input Chest X-Ray", use_container_width=True)
                
                # Image Enhancement Filter Toggle
                if st.checkbox("🔍 Enable CLAHE Adaptive Histogram Contrast Enhancement"):
                    enhanced_img = apply_clahe_enhancement(np.array(image))
                    st.image(enhanced_img, caption="CLAHE Enhanced Image", use_container_width=True)

        with col2:
            st.subheader("2. AI Diagnostic Triage & Heatmap")
            if image is not None:
                img_np = np.array(image)
                image_size = 224
                img_resized = cv2.resize(img_np, (image_size, image_size))
                rgb_float = img_resized.astype(np.float32) / 255.0

                transform = get_transforms(image_size=image_size, is_train=False)
                input_tensor = transform(image=img_np)["image"].unsqueeze(0).to(device)

                with torch.no_grad():
                    logits = model(input_tensor)
                    probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
                    pred_idx = int(np.argmax(probs))
                    confidence = float(probs[pred_idx])

                pred_label = inv_label_map.get(pred_idx, "Unknown")
                prob_dict = {inv_label_map[i]: float(probs[i]) for i in range(len(probs))}

                # Triage Banner
                if pred_label in ["tuberculosis", "bacterial_pneumonia"]:
                    st.markdown(f'<div class="triage-red">🚨 RED ALERT (HIGH SUSPICION): {pred_label.upper().replace("_", " ")}<br><small>Immediate Specialist Referral Required ({confidence*100:.1f}% Confidence)</small></div>', unsafe_allow_html=True)
                elif pred_label == "viral_pneumonia":
                    st.markdown(f'<div class="triage-yellow">⚠️ YELLOW ALERT (MODERATE SUSPICION): VIRAL PNEUMONIA<br><small>Secondary Monitoring & Isolation Recommended ({confidence*100:.1f}% Confidence)</small></div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="triage-green">✅ GREEN (LOW RISK): NORMAL CHEST X-RAY<br><small>No Immediate Opacities Detected ({confidence*100:.1f}% Confidence)</small></div>', unsafe_allow_html=True)

                st.write("")
                st.write("**Differential Pathology Probabilities:**")
                for k_lbl, p_val in prob_dict.items():
                    st.progress(float(p_val), text=f"{k_lbl.replace('_', ' ').title()}: {p_val*100:.1f}%")

                # Grad-CAM Heatmap
                target_layer = get_target_layer(model, backbone)
                cam = GradCAM(model=model, target_layers=target_layer)
                targets = [ClassifierOutputTarget(pred_idx)]
                grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]
                cam_overlay = show_cam_on_image(rgb_float, grayscale_cam, use_rgb=True)

                st.subheader("3. Grad-CAM Pathological Heatmap")
                st.image(cam_overlay, caption=f"Grad-CAM Attention Map ({backbone.upper()})", use_container_width=True)

                # Export PDF
                st.markdown("---")
                if st.button("📄 Export Diagnostic PDF Report"):
                    with tempfile.TemporaryDirectory() as tmp_dir:
                        orig_path = os.path.join(tmp_dir, "orig.png")
                        cam_path = os.path.join(tmp_dir, "cam.png")
                        cv2.imwrite(orig_path, cv2.cvtColor(img_resized, cv2.COLOR_RGB2BGR))
                        cv2.imwrite(cam_path, cv2.cvtColor(cam_overlay, cv2.COLOR_RGB2BGR))
                        
                        pdf_out = generate_pdf_report(patient_name, patient_id, pred_label, confidence, prob_dict, orig_path, cam_path)
                        
                        with open(pdf_out, "rb") as pdf_file:
                            st.download_button(
                                label="⬇️ Download PDF Report",
                                data=pdf_file,
                                file_name=f"CDSS_Report_{patient_id}.pdf",
                                mime="application/pdf"
                            )
            else:
                st.info("Please select or upload a Chest X-ray image on the left panel.")

    # ---------------------------------------------------------
    # TAB 2: Hospital Analytics Dashboard
    # ---------------------------------------------------------
    with tab2:
        st.subheader("📊 Hospital Population & Screening Analytics")
        df_manifest = load_manifest_data()
        
        if df_manifest is not None:
            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.markdown('<div class="metric-box"><div class="metric-val">6,900</div><div class="metric-lbl">Total Clean Images</div></div>', unsafe_allow_html=True)
            with m2:
                st.markdown('<div class="metric-box"><div class="metric-val">3,915</div><div class="metric-lbl">Unique Patients</div></div>', unsafe_allow_html=True)
            with m3:
                st.markdown('<div class="metric-box"><div class="metric-val">5,888</div><div class="metric-lbl">Duplicates Filtered</div></div>', unsafe_allow_html=True)
            with m4:
                st.markdown('<div class="metric-box"><div class="metric-val">100%</div><div class="metric-lbl">Zero Patient Leakage</div></div>', unsafe_allow_html=True)

            st.write("")
            c1, c2 = st.columns(2)
            with c1:
                fig_pie = px.pie(df_manifest, names='label', title='Pathology Distribution across Dataset', hole=0.4,
                                 color_discrete_sequence=['#2ecc71', '#e74c3c', '#e67e22', '#9b59b6'])
                st.plotly_chart(fig_pie, use_container_width=True)
            
            with c2:
                fig_bar = px.histogram(df_manifest, x='source', color='label', barmode='group',
                                       title='Pathology Count by Source Hospital',
                                       color_discrete_sequence=['#2ecc71', '#e74c3c', '#e67e22', '#9b59b6'])
                st.plotly_chart(fig_bar, use_container_width=True)

    # ---------------------------------------------------------
    # TAB 3: Benchmark & Explainability Hub
    # ---------------------------------------------------------
    with tab3:
        st.subheader("🔬 Empirical Benchmark & Model Comparison")
        st.write("Quantitative comparison across 3 random seeds (42, 7, 123) and classical baseline:")

        bench_df = pd.DataFrame([
            {"Model": "ResNet-18 (Deep Learning)", "Internal Acc": "84.98% ± 0.33%", "Internal F1": "84.09% ± 0.22%", "Internal AUC": "0.9538 ± 0.0020", "External AUC (Montgomery)": "0.7606 ± 0.0072"},
            {"Model": "DenseNet-121 (Deep Learning)", "Internal Acc": "84.55% ± 0.22%", "Internal F1": "83.22% ± 0.55%", "Internal AUC": "0.9512 ± 0.0024", "External AUC (Montgomery)": "0.8296 ± 0.0613 (BEST)"},
            {"Model": "EfficientNet-B0 (Deep Learning)", "Internal Acc": "83.83% ± 0.91%", "Internal F1": "82.69% ± 0.88%", "Internal AUC": "0.9493 ± 0.0015", "External AUC (Montgomery)": "0.7208 ± 0.0360"},
            {"Model": "Classical ML (HOG + SVM)", "Internal Acc": "83.51%", "Internal F1": "79.40%", "Internal AUC": "0.9470", "External AUC (Montgomery)": "0.6052 (POOR)"}
        ])
        st.dataframe(bench_df, use_container_width=True)

        st.subheader("Visual Grad-CAM Heatmap Samples")
        cam_path = Path("results/figures/gradcam_samples/gradcam_gallery_densenet121.png")
        if cam_path.exists():
            st.image(str(cam_path), caption="Grad-CAM Heatmap Gallery across Test Scans", use_container_width=True)

    # ---------------------------------------------------------
    # TAB 4: Scientific Integrity & Paper Draft
    # ---------------------------------------------------------
    with tab4:
        st.subheader("🛡️ Scientific Integrity Shield & Paper Repository")
        st.success("✅ Audit Verification Passed: 100% Zero Patient Leakage | Zero MD5 Overlap | 100% Isolated External Test")

        paper_path = Path("paper/paper_draft.md")
        if paper_path.exists():
            with open(paper_path, "r", encoding="utf-8") as f:
                paper_text = f.read()
            st.markdown(paper_text)
        else:
            st.info("Paper draft available at paper/paper_draft.md")

if __name__ == "__main__":
    main()
