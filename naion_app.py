import streamlit as st
import torch
import cv2
import numpy as np
import segmentation_models_pytorch as smp
from huggingface_hub import hf_hub_download


# -----------------------------
# 1. PAGE SETUP
# -----------------------------

st.set_page_config(page_title="NAION AI Support", layout="wide")
st.title("👁️ NAION-Risk: AI Decision Support")


# -----------------------------
# 2. MODEL LOADING
# -----------------------------

@st.cache_resource
def load_ai_model():
    try:
        model_path = hf_hub_download(
            repo_id="jani2904/NAION-Risk-Analyzer",
            filename="NAION_Risk_Unet_v1.pth",
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
# 3. MEASUREMENT ENGINE
# -----------------------------

def calculate_full_metrics(mask_disc, mask_cup, image_rgb):
    results = {}

    disc_coords = np.argwhere(mask_disc > 0)
    if len(disc_coords) == 0:
        return None

    cy, cx = np.mean(disc_coords, axis=0).astype(int)

    min_y, max_y = np.min(disc_coords[:, 0]), np.max(disc_coords[:, 0])
    min_x, max_x = np.min(disc_coords[:, 1]), np.max(disc_coords[:, 1])

    disc_height = max_y - min_y
    disc_width = max_x - min_x

    if disc_height <= 0 or disc_width <= 0:
        return None

    cup_coords = np.argwhere(mask_cup > 0)

    if len(cup_coords) > 0:
        c_min_y, c_max_y = np.min(cup_coords[:, 0]), np.max(cup_coords[:, 0])
        c_min_x, c_max_x = np.min(cup_coords[:, 1]), np.max(cup_coords[:, 1])

        results["rim_s"] = max(0, c_min_y - min_y)
        results["rim_i"] = max(0, max_y - c_max_y)
        results["rim_left"] = max(0, c_min_x - min_x)
        results["rim_right"] = max(0, max_x - c_max_x)

        cup_height = c_max_y - c_min_y
        results["vcdr"] = cup_height / disc_height if disc_height > 0 else 0.0

    else:
        results["rim_s"] = cy - min_y
        results["rim_i"] = max_y - cy
        results["rim_left"] = cx - min_x
        results["rim_right"] = max_x - cx
        results["vcdr"] = 0.0

    # Vessel density estimate
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)

    vessels = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        11,
        2
    )

    vessels = cv2.bitwise_and(vessels, vessels, mask=mask_disc.astype(np.uint8))

    disc_area = np.sum(mask_disc > 0)
    vessel_pixels = np.sum(vessels > 0)

    results["density"] = vessel_pixels / disc_area if disc_area > 0 else 0.0

    # Approximate tortuosity
    contours, _ = cv2.findContours(
        vessels,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE
    )

    if contours:
        cnt = max(contours, key=cv2.contourArea)

        arc_length = cv2.arcLength(cnt, False)

        x_vals = cnt[:, :, 0]
        y_vals = cnt[:, :, 1]

        chord_length = np.sqrt(
            (np.max(x_vals) - np.min(x_vals)) ** 2 +
            (np.max(y_vals) - np.min(y_vals)) ** 2
        )

        results["tortuosity"] = arc_length / chord_length if chord_length > 0 else 1.0

    else:
        results["tortuosity"] = 1.0

    return results


# -----------------------------
# 4. IMAGE UPLOAD
# -----------------------------

uploaded_file = st.sidebar.file_uploader(
    "Upload Fundus Image",
    type=["jpg", "jpeg", "png"]
)


if uploaded_file is not None:

    # Load image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    original_img = cv2.imdecode(file_bytes, 1)

    if original_img is None:
        st.error("Could not read uploaded image.")
        st.stop()

    original_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    h, w = original_rgb.shape[:2]

    # Preprocess image
    input_img = cv2.resize(original_rgb, (256, 256))
    input_tensor = (
        torch.from_numpy(input_img)
        .permute(2, 0, 1)
        .float()
        .unsqueeze(0)
        / 255.0
    )

    # Inference
    with torch.no_grad():
        output = torch.sigmoid(model(input_tensor)).squeeze().cpu().numpy()

    mask_disc_raw = (output[0] > 0.3).astype(np.uint8)
    mask_cup_raw = (output[1] > 0.1).astype(np.uint8)

    # Resize masks to original image size
    mask_disc = cv2.resize(mask_disc_raw, (w, h), interpolation=cv2.INTER_NEAREST)
    mask_cup = cv2.resize(mask_cup_raw, (w, h), interpolation=cv2.INTER_NEAREST)

    # Calculate metrics
    metrics = calculate_full_metrics(mask_disc, mask_cup, original_rgb)

    if metrics is None:
        st.error("⚠️ No optic disc detected. Try another image.")
        st.stop()

    # -----------------------------
    # 5. DISPLAY RESULTS
    # -----------------------------

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Clinical Source")
        st.image(original_rgb, use_container_width=True)

    with col2:
        st.subheader("AI Segmentation Overlay")

        vis_mask = np.zeros_like(original_rgb)
        vis_mask[mask_disc == 1] = [0, 255, 0]   # Disc = green
        vis_mask[mask_cup == 1] = [255, 0, 0]    # Cup = red

        blended = cv2.addWeighted(original_rgb, 0.7, vis_mask, 0.3, 0)

        st.image(
            blended,
            caption="Green = Disc, Red = Cup",
            use_container_width=True
        )

    st.markdown("---")
    st.subheader("📊 Automated Ocular Metrics")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Superior Rim Estimate", f"{int(metrics['rim_s'])} px")
    c2.metric("Inferior Rim Estimate", f"{int(metrics['rim_i'])} px")
    c3.metric("Left Rim Estimate", f"{int(metrics['rim_left'])} px")
    c4.metric("Right Rim Estimate", f"{int(metrics['rim_right'])} px")

    c5, c6, c7 = st.columns(3)

    c5.metric("vCDR Estimate", f"{metrics['vcdr']:.2f}")
    c6.metric("Disc Vessel Density Estimate", f"{metrics['density']:.1%}")
    c7.metric("Tortuosity Estimate", f"{metrics['tortuosity']:.2f}")

    st.markdown("---")
    st.subheader("Clinical Interpretation")

    if metrics["vcdr"] == 0:
        st.error(
            "Finding: Cupless / very small cup phenotype. "
            "This may suggest a crowded optic disc pattern."
        )

    elif metrics["vcdr"] < 0.2:
        st.warning(
            "Finding: Small cup / crowded disc pattern. "
            "Consider this as a higher-risk anatomical profile."
        )

    elif metrics["tortuosity"] > 1.9:
        st.warning(
            "Finding: Elevated estimated vessel tortuosity. "
            "Review image quality and vascular pattern manually."
        )

    else:
        st.success(
            "Finding: Baseline optic disc architecture appears relatively stable."
        )

    st.caption(
        "Note: This app provides image-derived estimates only. "
        "It is not a diagnostic medical device and should not replace clinical review."
    )
        
