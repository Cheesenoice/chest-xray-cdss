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
    page_title="Chest X-Ray CDSS | Explainable AI Screening",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for polished aesthetics
st.markdown("""
    <style>
    .main-title { font-size: 2.2rem; font-weight: 800; color: #1E293B; margin-bottom: 0.2rem; }
    .sub-title { font-size: 1.0rem; color: #64748B; margin-bottom: 1.5rem; }
    .triage-red { background-color: #FEE2E2; border-left: 6px solid #EF4444; padding: 12px; border-radius: 6px; color: #991B1B; font-weight: bold; }
    .triage-yellow { background-color: #FEF3C7; border-left: 6px solid #F59E0B; padding: 12px; border-radius: 6px; color: #92400E; font-weight: bold; }
    .triage-green { background-color: #D1FAE5; border-left: 6px solid #10B981; padding: 12px; border-radius: 6px; color: #065F46; font-weight: bold; }
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

def main():
    st.markdown('<div class="main-title">🫁 Explainable Chest X-Ray Clinical Decision Support System</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Multi-class pathology triage and Grad-CAM heatmap localization for primary-care screening.</div>', unsafe_allow_html=True)

    # Sidebar Config
    st.sidebar.header("⚙️ System Configuration")
    backbone = st.sidebar.selectbox("Select Model Architecture", ["densenet121", "resnet18", "efficientnet_b0"], index=0)
    num_classes = st.sidebar.radio("Active Class Mode", [4, 3], index=0, help="4-class includes TB; 3-class focuses on Pneumonia benchmark")
    
    label_map = LABEL_MAP_4CLASS if num_classes == 4 else LABEL_MAP_3CLASS
    inv_label_map = {v: k for k, v in label_map.items()}

    ckpt_path = f"results/checkpoints/best_{backbone}_seed42.pt"
    model, device = load_cdss_model(backbone, num_classes, ckpt_path)

    st.sidebar.markdown("---")
    st.sidebar.subheader("👤 Patient Metadata")
    patient_id = st.sidebar.text_input("Patient ID", "PAT-2026-0089")
    patient_name = st.sidebar.text_input("Patient Full Name", "Anonymous Patient")

    # Main Layout
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("1. Upload Chest X-Ray Image")
        uploaded_file = st.file_uploader("Choose a DICOM/JPEG/PNG image...", type=["jpeg", "jpg", "png"])
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Uploaded Chest X-Ray", use_column_width=True)

    with col2:
        st.subheader("2. Diagnostic Inference & Explainability")
        if uploaded_file is not None:
            # Process image for inference
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

            # Triage Status Indicator
            if pred_label in ["tuberculosis", "bacterial_pneumonia"]:
                st.markdown(f'<div class="triage-red">🚨 URGENT REFERRAL: {pred_label.upper().replace("_", " ")} ({confidence*100:.1f}% Confidence)</div>', unsafe_allow_html=True)
            elif pred_label == "viral_pneumonia":
                st.markdown(f'<div class="triage-yellow">⚠️ MODERATE SUSPICION: VIRAL PNEUMONIA ({confidence*100:.1f}% Confidence)</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="triage-green">✅ LOW RISK: NORMAL CHEST X-RAY ({confidence*100:.1f}% Confidence)</div>', unsafe_allow_html=True)

            st.write("")
            st.write("**Probability Distribution:**")
            for k_lbl, p_val in prob_dict.items():
                st.progress(float(p_val), text=f"{k_lbl.replace('_', ' ').title()}: {p_val*100:.1f}%")

            # Compute Grad-CAM
            target_layer = get_target_layer(model, backbone)
            cam = GradCAM(model=model, target_layers=target_layer)
            targets = [ClassifierOutputTarget(pred_idx)]
            grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]
            cam_overlay = show_cam_on_image(rgb_float, grayscale_cam, use_rgb=True)

            st.subheader("3. Grad-CAM Pathological Heatmap")
            st.image(cam_overlay, caption=f"Grad-CAM Attention Map ({backbone})", use_column_width=True)

            # Export PDF Report
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
            st.info("Please upload a Chest X-ray image on the left panel to trigger AI screening.")

if __name__ == "__main__":
    main()
