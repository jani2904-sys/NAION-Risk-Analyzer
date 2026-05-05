import streamlit as st
import torch
import cv2
import numpy as np
import segmentation_models_pytorch as smp
from huggingface_hub import hf_hub_download
from skimage.morphology import skeletonize
from skimage.measure import label, regionprops

# -----------------------------
# 1. PAGE SETUP
# -----------------------------
st.set_page_config(page_title="NAION AI Support", layout="wide")
st.title("👁️ NAION-Risk: Crowded Disc Screening Support")
st.caption("System Version: 2.1.0 | Clinical Audit Profile: May 2026 | Dataset: jani2904/NAION-Risk-Analyzer")

# -----------------------------
# 2. MODEL LOADING (V2 INTEGRATION)
# -----------------------------
@st.cache_resource
def load_ai_model():
    try:
        # Pulling the 2nd set of weights from Hugging Face
        model_path = hf_hub_download(
            repo_id="jani2904/NAION-Risk-Analyzer",
            filename="NAION_Risk_Unet_v1.pth", # Ensure this filename matches your HF upload
            repo_type="dataset"
        )

        model = smp.Unet(
            encoder_name="resnet34",
            encoder_weights=None,
            in_channels=3,
            classes=2
        )

        state_dict = torch.load(model_path, map_location="cpu")
        model.load_state_dict(state_dict)
        model.eval()
        return model

    except Exception as e:
        st.error(f"Critical Error loading model: {e}")
        return None

model = load_ai_model()
if model is None:
    st.stop()

# -----------------------------
# 3. VESSEL / TORTUOSITY ENGINE
# -----------------------------
def extract_vessels(image_rgb, mask_disc):
    green = image_rgb[:, :, 1]
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    green_eq = clahe.apply(green)
    inv = 255 - green_eq
    inv_blur = cv2.GaussianBlur(inv, (3, 3), 0)
    
    vessels = cv2.adaptiveThreshold(
        inv_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 21, -2
    )
    vessels = cv2.bitwise_and(vessels, vessels, mask=mask_disc.astype(np.uint8))
    kernel = np.ones((2, 2), np.uint8)
    vessels = cv2.morphologyEx(vessels, cv2.MORPH_OPEN, kernel)
    return cv2.morphologyEx(vessels, cv2.MORPH_CLOSE, kernel)

def calculate_skeleton_tortuosity(vessels, min_branch_pixels=30):
    vessel_binary = vessels > 0
    if np.sum(vessel_binary) == 0:
        return 1.0, 0
    
    skeleton = skeletonize(vessel_binary)
    labeled = label(skeleton)
    tort_values = []

    for region in regionprops(labeled):
        coords = region.coords
        if len(coords) < min_branch_pixels:
            continue
        
        arc_length = len(coords)
        y, x = coords[:, 0], coords[:, 1]
        chord_length = np.sqrt((np.max(x) - np.min(x))**2 + (np.max(y) - np.min(y))**2)
        
        if chord_length > 0:
            tort = arc_length / chord_length
            if 1.0 <= tort <= 3.0:
                tort_values.append(tort)

    return (float(np.median(tort_values)), len(tort_values)) if tort_values else (1.0, 0)

# -----------------------------
# 4. MEASUREMENT ENGINE
# -----------------------------
def calculate_full_metrics(mask_disc, mask_cup, image_rgb):
    results = {}
    disc_coords = np.argwhere(mask_disc > 0)
    if len(disc_coords) == 0: return None

    min_y, max_y = np.min(disc_coords[:, 0]), np.max(disc_coords[:, 0])
    min_x, max_x = np.min(disc_coords[:, 1]), np.max(disc_coords[:, 1])
    disc_height = max_y - min_y
    
    cup_coords = np.argwhere(mask_cup > 0)
    if len(cup_coords) > 0:
        c_min_y, c_max_y = np.min(cup_coords[:, 0]), np.max(cup_coords[:, 0])
        c_min_x, c_max_x = np.min(cup_coords[:, 1]), np.max(cup_coords[:, 1])
        results["rim_s"] = max(0, c_min_y - min_y)
        results["rim_i"] = max(0, max_y - c_max_y)
        results["vcdr"] = (c_max_y - c_min_y) / disc_height
    else:
        results["rim_s"], results["rim_i"], results["vcdr"] = disc_height/2, disc_height/2, 0.0

    results["rim_s_ratio"] = results["rim_s"] / disc_height
    results["rim_i_ratio"] = results["rim_i"] / disc_height
    
    vessels = extract_vessels(image_rgb, mask_disc)
    results["density"] = np.sum(vessels > 0) / np.sum(mask_disc > 0)
    results["tortuosity"], results["vessel_branch_count"] = calculate_skeleton_tortuosity(vessels)
    results["vessels_mask"] = vessels
    return results

# -----------------------------
# 5. TRIAGE LOGIC
# -----------------------------
def crowded_disc_triage(metrics):
    vcdr, density = metrics["vcdr"], metrics["density"]
    thick_rim = (metrics["rim_s_ratio"] > 0.20) and (metrics["rim_i_ratio"] > 0.20)

    if vcdr < 0.05: return "High likelihood: Crowded / cupless disc", "high"
    if vcdr < 0.20 and thick_rim: return "Likely crowded disc anatomy", "high"
    if vcdr < 0.20: return "Possible crowded disc anatomy", "moderate"
    if vcdr < 0.30 and density > 0.18: return "Borderline small cup; review recommended", "moderate"
    return "Not crowded by cup-disc anatomy", "low"

# -----------------------------
# 6. APP INTERFACE
# -----------------------------
uploaded_file = st.sidebar.file_uploader("Upload Fundus Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    original_img = cv2.imdecode(file_bytes, 1)
    original_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    h, w = original_rgb.shape[:2]

    # Preprocess & Inference
    input_tensor = torch.from_numpy(cv2.resize(original_rgb, (256, 256))).permute(2, 0, 1).float().unsqueeze(0) / 255.0
    with torch.no_grad():
        output = torch.sigmoid(model(input_tensor)).squeeze().cpu().numpy()

    mask_disc = cv2.resize((output[0] > 0.3).astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
    mask_cup = cv2.resize((output[1] > 0.1).astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)

    metrics = calculate_full_metrics(mask_disc, mask_cup, original_rgb)
    if not metrics:
        st.error("⚠️ No optic disc detected.")
        st.stop()

    triage_text, triage_level = crowded_disc_triage(metrics)

    # Visualization
    c1, c2, c3 = st.columns(3)
    c1.image(original_rgb, caption="Source Fundus", use_container_width=True)
    
    vis_mask = np.zeros_like(original_rgb)
    vis_mask[mask_disc == 1], vis_mask[mask_cup == 1] = [0, 255, 0], [255, 0, 0]
    c2.image(cv2.addWeighted(original_rgb, 0.7, vis_mask, 0.3, 0), caption="Green=Disc, Red=Cup", use_container_width=True)
    c3.image(metrics["vessels_mask"], caption="Vessel Map", use_container_width=True, clamp=True)

    # Analytics Dashboard
    st.markdown("---")
    st.subheader("📊 Automated Metrics & Triage")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("vCDR", f"{metrics['vcdr']:.2f}")
    m2.metric("Vessel Density", f"{metrics['density']:.1%}")
    m3.metric("Tortuosity", f"{metrics['tortuosity']:.2f}")
    m4.metric("Branches", metrics['vessel_branch_count'])

    if triage_level == "high": st.error(f"🚨 {triage_text}")
    elif triage_level == "moderate": st.warning(f"⚠️ {triage_text}")
    else: st.success(f"✅ {triage_text}")

    st.caption("Disclaimer: Image-derived screening estimates only. Not a primary diagnostic device.")
