"""
Streamlit demo app for Chest X-ray CDSS.
Run with: streamlit run app/app.py
"""
import os
import sys
import yaml
import tempfile
from pathlib import Path

import streamlit as st
import torch
import numpy as np
import cv2
from PIL import Image

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models import build_model
from src.datasets import LABEL_MAP_4CLASS, LABEL_MAP_3CLASS


@st.cache_resource
def load_model(config_path="configs/default.yaml"):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    num_classes = cfg["data"].get("num_classes", 4)
    backbone = cfg["model"]["backbone"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(backbone_name=backbone, num_classes=num_classes,
                        pretrained=False, drop_rate=0).to(device)
    model.eval()
    ckpt_path = Path("results/checkpoints") / f"best_{backbone}_seed{cfg.get('seed', 42)}_cls{num_classes}.pt"
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        st.warning(f"No checkpoint found at {ckpt_path}. Model is untrained.")
    return model, device, num_classes, cfg


def predict_image(model, img_array, device, num_classes):
    from torchvision import transforms as T
    preprocess = T.Compose([
        T.ToPILImage(),
        T.Resize(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img_tensor = preprocess(img_array).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(img_tensor)
        probs = torch.softmax(output, dim=1).squeeze().cpu().numpy()
    return probs


def triage_level(probs, class_names):
    idx = int(np.argmax(probs))
    conf = probs[idx]
    label = class_names[idx]
    if label == "normal":
        return "Low (Normal)", "green"
    elif conf > 0.7:
        return "High (Abnormal)", "red"
    else:
        return "Medium (Watch)", "yellow"


st.set_page_config(page_title="Chest X-ray CDSS", layout="wide")
st.title("Chest X-ray Clinical Decision Support System")
st.markdown("*Powered by Explainable Deep Learning*")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Upload Chest X-ray")
    uploaded = st.file_uploader("Choose an X-ray image...", type=["png", "jpg", "jpeg"])

with col2:
    st.subheader("Diagnosis & Triage")

if uploaded is not None:
    file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    with col1:
        st.image(img_rgb, caption="Uploaded X-ray", use_container_width=True)

    with col2:
        with st.spinner("Analyzing..."):
            model, device, num_classes, cfg = load_model()
            label_map = LABEL_MAP_4CLASS if num_classes == 4 else LABEL_MAP_3CLASS
            class_names = [k for k, v in sorted(label_map.items(), key=lambda x: x[1])]

            probs = predict_image(model, img_rgb, device, num_classes)
            pred_idx = int(np.argmax(probs))
            pred_label = class_names[pred_idx]
            conf = probs[pred_idx]

            st.metric("Prediction", pred_label, f"{conf:.1%}")
            for i, (name, p) in enumerate(zip(class_names, probs)):
                st.progress(float(p), text=f"{name}: {p:.1%}")

            triage_text, color = triage_level(probs, class_names)
            st.markdown(f"### Triage: :{color}[{triage_text}]")

            st.info("This system is for clinical decision support only. "
                    "Not a substitute for physician diagnosis.")
