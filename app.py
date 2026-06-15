import streamlit as st
import torch
import cv2
import numpy as np
import segmentation_models_pytorch as smp
from huggingface_hub import hf_hub_download
# Import your new engine modules
from engine.metrics import calculate_full_metrics, crowded_disc_triage

# 1. SETUP
st.set_page_config(page_title="NAION AI Support v2", layout="wide")
st.title("👁️ NAION-Risk: AI Decision Support")

# 2. MODEL LOADING (Kept here as it's UI/App specific)
@st.cache_resource
def load_ai_model():
    model_path = hf_hub_download(
        repo_id="jani2904/NAION-Risk-Analyzer",
        filename="NAION_Risk_Unet_v1.pth",
        repo_type="dataset"
    )
    model = smp.Unet(encoder_name="resnet34", encoder_weights=None, in_channels=3, classes=2)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    return model

model = load_ai_model()

# 3. INTERFACE & LOGIC
uploaded_file = st.sidebar.file_uploader("Upload Fundus Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Inference logic
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    original_img = cv2.imdecode(file_bytes, 1)
    original_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    h, w = original_rgb.shape[:2]

    input_tensor = torch.from_numpy(cv2.resize(original_rgb, (256, 256))).permute(2, 0, 1).float().unsqueeze(0) / 255.0
    with torch.no_grad():
        output = torch.sigmoid(model(input_tensor)).squeeze().cpu().numpy()

    mask_disc = cv2.resize((output[0] > 0.3).astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
    mask_cup = cv2.resize((output[1] > 0.1).astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)

    # CALLING THE ENGINE (This is where the magic happens)
    metrics = calculate_full_metrics(mask_disc, mask_cup, original_rgb)
    
    if metrics:
        triage_text, triage_level = crowded_disc_triage(metrics)
        # Display results (Visualization code goes here)
        st.write(f"Triage: {triage_text}")
