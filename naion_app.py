import streamlit as st
import torch
import cv2
import numpy as np
import segmentation_models_pytorch as smp
from PIL import Image

# --- 1. MEASUREMENT ENGINE ---

def calculate_isnt_rims(mask_disc, mask_cup):
    """Calculates superior rim thickness and vCDR."""
    disc_coords = np.argwhere(mask_disc)
    if len(disc_coords) == 0:
        return 0, 0, 0
    
    center_x = int(np.mean(disc_coords[:, 1]))
    
    # Get vertical bounds of Disc
    disc_y_coords = disc_coords[disc_coords[:, 1] == center_x][:, 0]
    if len(disc_y_coords) == 0:
        return 0, 0, 0
    disc_top = np.min(disc_y_coords)
    disc_bottom = np.max(disc_y_coords)
    disc_height = disc_bottom - disc_top
    
    # Get vertical bounds of Cup
    cup_coords = np.argwhere(mask_cup)
    if len(cup_coords) == 0:
        center_y = int(np.mean(disc_coords[:, 0]))
        rim_s = center_y - disc_top
        return max(0, rim_s), 0.0, disc_height

    cup_y_coords = cup_coords[cup_coords[:, 1] == center_x][:, 0]
    if len(cup_y_coords) == 0:
        center_y = int(np.mean(disc_coords[:, 0]))
        rim_s = center_y - disc_top
        return max(0, rim_s), 0.0, disc_height
    
    cup_top = np.min(cup_y_coords)
    cup_bottom = np.max(cup_y_coords)
    rim_s = cup_top - disc_top
    v_cdr_val = (cup_bottom - cup_top) / disc_height if disc_height > 0 else 0
    
    return max(0, rim_s), v_cdr_val, disc_height

def calculate_vessel_density(image_rgb, mask_disc):
    """Calculates vessel density in the superior quadrant."""
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    vessels = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                    cv2.THRESH_BINARY_INV, 11, 2)
    
    disc_coords = np.argwhere(mask_disc)
    if len(disc_coords) == 0: return 0
    
    min_y, max_y = np.min(disc_coords[:, 0]), np.max(disc_coords[:, 0])
    mid_y = (min_y + max_y) // 2
    
    sup_mask = np.zeros_like(mask_disc)
    sup_mask[min_y:mid_y, :] = 1
    target_area = cv2.bitwise_and(mask_disc, sup_mask)
    
    vessel_pixels = np.sum(cv2.bitwise_and(vessels, vessels, mask=target_area) > 0)
    total_pixels = np.sum(target_area)
    
    return (vessel_pixels / total_pixels) if total_pixels > 0 else 0

# --- 2. UI SETUP & MODEL LOADING ---

st.set_page_config(page_title="NAION AI Support", layout="wide")
st.title("👁️ NAION-Risk: AI Decision Support")
st.markdown("Automated anatomical and vascular profiling for Optic Nerve Head analysis.")

@st.cache_resource
def load_ai_model():
    model = smp.Unet(encoder_name="resnet34", encoder_weights=None, in_channels=3, classes=2)
    model.load_state_dict(torch.load("NAION_Risk_Unet_v1.pth", map_location='cpu'))
    model.eval()
    return model

model = load_ai_model()

from huggingface_hub import hf_hub_download
import os

@st.cache_resource
def load_ai_model():
    REPO_ID = "jani2904/NAION-Risk-Analyzer" 
    FILENAME = "NAION_Risk_Unet_v1.pth"
    
    # 1. This downloads the file and returns the ACTUAL path on the Streamlit server
    model_path = hf_hub_download(repo_id=REPO_ID, filename=FILENAME)
    
    model = smp.Unet(
        encoder_name="resnet34", 
        encoder_weights=None, 
        in_channels=3, 
        classes=2
    )
    
    # 2. CRITICAL CHANGE: Use model_path variable, NOT the filename string "NAION_..."
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    
    model.eval()
    return model
# --- 3. SIDEBAR & FILE UPLOAD ---

uploaded_file = st.sidebar.file_uploader("Upload Fundus Image", type=['jpg', 'png', 'jpeg'])

if uploaded_file is not None:
    # A. IMAGE LOADING
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    original_img = cv2.imdecode(file_bytes, 1)
    original_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    h, w = original_rgb.shape[:2]

    # B. AI INFERENCE
    input_img = cv2.resize(original_rgb, (256, 256)) 
    input_tensor = torch.from_numpy(input_img).transpose(0, 2).transpose(1, 2).float().unsqueeze(0) / 255.0

    with torch.no_grad():
        output = torch.sigmoid(model(input_tensor)).squeeze().cpu().numpy()
        mask_disc_raw = (output[0] > 0.3).astype(np.uint8) 
        mask_cup_raw = (output[1] > 0.1).astype(np.uint8)

    # C. RESCALE MASKS
    mask_disc = cv2.resize(mask_disc_raw, (w, h), interpolation=cv2.INTER_NEAREST)
    mask_cup = cv2.resize(mask_cup_raw, (w, h), interpolation=cv2.INTER_NEAREST)

    # D. ANALYTICS
    rim_s_val, vcdr_calc, disc_h = calculate_isnt_rims(mask_disc, mask_cup)
    density_val = calculate_vessel_density(original_rgb, mask_disc)

    # E. DASHBOARD DISPLAY
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Clinical Source")
        st.image(original_rgb, use_container_width=True)
    with col2:
        st.subheader("AI Analysis Overlay")
        if np.sum(mask_disc) > 0:
            vis_mask = np.zeros((h, w, 3), dtype=np.uint8)
            vis_mask[mask_disc == 1] = [0, 255, 0] # Green Disc
            vis_mask[mask_cup == 1] = [255, 0, 0]  # Red Cup
            blended = cv2.addWeighted(original_rgb, 0.7, vis_mask, 0.3, 0)
            st.image(blended, caption="AI Detection (Green=Disc, Red=Cup)", use_container_width=True)
        else:
            st.error("⚠️ No Disc Detected.")

    # F. LOGICAL RISK INTERPRETATION
    st.markdown("---")
    st.subheader("Automated Risk Metrics")
    
    # NEW LOGIC: Only High Risk if Rim is thick AND Cup is small (< 0.2)
    # We also use a percentage (Rim/Disc Height) to make it scale-independent
    rim_ratio = (rim_s_val / disc_h) if disc_h > 0 else 0
    
    is_high_risk = (rim_ratio > 0.15) and (vcdr_calc < 0.2)
    risk_label = "High Risk (Crowded)" if is_high_risk else "Normal (Buffered)"
    risk_color = "inverse" if is_high_risk else "normal"

    m1, m2, m3 = st.columns(3)
    m1.metric("Superior Rim Thickness", f"{int(rim_s_val)} px", delta=risk_label, delta_color=risk_color)
    m2.metric("vCDR", f"{vcdr_calc:.2f}")
    m3.metric("Superior Vessel Density", f"{density_val:.1%}")

    st.markdown("### Clinical Interpretation")
    if vcdr_calc == 0:
        st.error("**Finding: Cupless Phenotype.** This indicates 'Mechanical Collapse' where vascular stress is immobilized by axonal crowding.")
    elif vcdr_calc <= 0.2:
        st.warning("**Finding: Transition Zone.** The patient is at the 'Vascular Cliff' where perfusion drops rapidly.")
    else:
        st.success("**Finding: Healthy Architecture.** Sufficient cup space observed to buffer mechanical pressure.")
