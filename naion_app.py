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
# 3. VESSEL / TORTUOSITY ENGINE
# -----------------------------

def extract_vessels(image_rgb, mask_disc):
    """
    Estimate vessels inside optic disc using green-channel enhancement.
    This is still approximate, but more stable than raw grayscale thresholding.
    """

    green = image_rgb[:, :, 1]

    # Contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    green_eq = clahe.apply(green)

    # Vessels are darker in green channel, so invert
    inv = 255 - green_eq

    # Blur to reduce small noise
    inv_blur = cv2.GaussianBlur(inv, (3, 3), 0)

    # Adaptive threshold
    vessels = cv2.adaptiveThreshold(
        inv_blur,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        21,
        -2
    )

    # Restrict to disc
    vessels = cv2.bitwise_and(vessels, vessels, mask=mask_disc.astype(np.uint8))

    # Morphological cleanup
    kernel = np.ones((2, 2), np.uint8)
    vessels = cv2.morphologyEx(vessels, cv2.MORPH_OPEN, kernel)
    vessels = cv2.morphologyEx(vessels, cv2.MORPH_CLOSE, kernel)

    return vessels


def calculate_skeleton_tortuosity(vessels, min_branch_pixels=30):
    """
    Improved tortuosity estimate:
    - skeletonizes vessel mask
    - separates connected branches
    - computes arc/chord ratio per branch
    - returns median of valid branches

    This avoids using one huge noisy merged contour.
    """

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

        # Arc length approximated by number of skeleton pixels
        arc_length = len(coords)

        # Estimate chord length using farthest endpoint approximation
        y = coords[:, 0]
        x = coords[:, 1]

        points = np.column_stack([x, y])

        # Use bounding-box diagonal as conservative chord estimate
        chord_length = np.sqrt(
            (np.max(x) - np.min(x)) ** 2 +
            (np.max(y) - np.min(y)) ** 2
        )

        if chord_length <= 0:
            continue

        tort = arc_length / chord_length

        # Keep physiologically plausible values only
        if 1.0 <= tort <= 3.0:
            tort_values.append(tort)

    if len(tort_values) == 0:
        return 1.0, 0

    median_tortuosity = float(np.median(tort_values))

    return median_tortuosity, len(tort_values)


# -----------------------------
# 4. MEASUREMENT ENGINE
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

    results["disc_height"] = disc_height
    results["disc_width"] = disc_width

    # Rim ratios
    results["rim_s_ratio"] = results["rim_s"] / disc_height
    results["rim_i_ratio"] = results["rim_i"] / disc_height

    # Vessel density
    vessels = extract_vessels(image_rgb, mask_disc)

    disc_area = np.sum(mask_disc > 0)
    vessel_pixels = np.sum(vessels > 0)

    results["density"] = vessel_pixels / disc_area if disc_area > 0 else 0.0

    # Improved tortuosity
    tortuosity, branch_count = calculate_skeleton_tortuosity(vessels)

    results["tortuosity"] = tortuosity
    results["vessel_branch_count"] = branch_count
    results["vessels_mask"] = vessels

    return results


# -----------------------------
# 5. TRIAGE LOGIC
# -----------------------------

def crowded_disc_triage(metrics):
    vcdr = metrics["vcdr"]
    density = metrics["density"]
    rim_s_ratio = metrics["rim_s_ratio"]
    rim_i_ratio = metrics["rim_i_ratio"]

    thick_rim_support = (rim_s_ratio > 0.20) and (rim_i_ratio > 0.20)
    density_support = density > 0.18

    if vcdr < 0.05:
        return "High likelihood: Crowded / cupless disc", "high"

    elif vcdr < 0.20 and thick_rim_support:
        return "Likely crowded disc anatomy", "high"

    elif vcdr < 0.20:
        return "Possible crowded disc anatomy", "moderate"

    elif vcdr < 0.30 and density_support:
        return "Borderline small cup; manual review recommended", "moderate"

    else:
        return "Not crowded by cup-disc anatomy", "low"


# -----------------------------
# 6. IMAGE UPLOAD
# -----------------------------

uploaded_file = st.sidebar.file_uploader(
    "Upload Fundus Image",
    type=["jpg", "jpeg", "png"]
)


if uploaded_file is not None:

    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    original_img = cv2.imdecode(file_bytes, 1)

    if original_img is None:
        st.error("Could not read uploaded image.")
        st.stop()

    original_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    h, w = original_rgb.shape[:2]

    # Preprocess
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

    # Resize masks
    mask_disc = cv2.resize(mask_disc_raw, (w, h), interpolation=cv2.INTER_NEAREST)
    mask_cup = cv2.resize(mask_cup_raw, (w, h), interpolation=cv2.INTER_NEAREST)

    metrics = calculate_full_metrics(mask_disc, mask_cup, original_rgb)

    if metrics is None:
        st.error("⚠️ No optic disc detected. Try another image.")
        st.stop()

    triage_text, triage_level = crowded_disc_triage(metrics)

    # -----------------------------
    # 7. VISUAL DISPLAY
    # -----------------------------

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Clinical Source")
        st.image(original_rgb, use_container_width=True)

    with col2:
        st.subheader("Disc/Cup Overlay")

        vis_mask = np.zeros_like(original_rgb)
        vis_mask[mask_disc == 1] = [0, 255, 0]
        vis_mask[mask_cup == 1] = [255, 0, 0]

        blended = cv2.addWeighted(original_rgb, 0.7, vis_mask, 0.3, 0)

        st.image(
            blended,
            caption="Green = Disc, Red = Cup",
            use_container_width=True
        )

    with col3:
        st.subheader("Estimated Vessel Map")
        st.image(
            metrics["vessels_mask"],
            caption="Estimated vessels inside disc",
            use_container_width=True,
            clamp=True
        )

    # -----------------------------
    # 8. METRICS
    # -----------------------------

    st.markdown("---")
    st.subheader("📊 Automated Ocular Metrics")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Superior Rim Estimate", f"{int(metrics['rim_s'])} px")
    c2.metric("Inferior Rim Estimate", f"{int(metrics['rim_i'])} px")
    c3.metric("Left Rim Estimate", f"{int(metrics['rim_left'])} px")
    c4.metric("Right Rim Estimate", f"{int(metrics['rim_right'])} px")

    c5, c6, c7, c8 = st.columns(4)

    c5.metric("vCDR Estimate", f"{metrics['vcdr']:.2f}")
    c6.metric("Disc Vessel Density Estimate", f"{metrics['density']:.1%}")
    c7.metric("Median Tortuosity Estimate", f"{metrics['tortuosity']:.2f}")
    c8.metric("Vessel Branches Used", f"{metrics['vessel_branch_count']}")

    # -----------------------------
    # 9. TRIAGE RESULT
    # -----------------------------

    st.markdown("---")
    st.subheader("Crowded Disc Triage")

    if triage_level == "high":
        st.error(f"🚨 {triage_text}")

    elif triage_level == "moderate":
        st.warning(f"⚠️ {triage_text}")

    else:
        st.success(f"✅ {triage_text}")

    # Tortuosity interpretation
    st.markdown("### Vessel Pattern Interpretation")

    if metrics["vessel_branch_count"] < 3:
        st.warning(
            "Tortuosity estimate is based on too few vessel branches. "
            "Interpret cautiously."
        )

    elif metrics["tortuosity"] < 1.3:
        st.success("Estimated vessel tortuosity appears within expected range.")

    elif metrics["tortuosity"] < 1.7:
        st.warning("Mildly increased estimated tortuosity.")

    else:
        st.warning(
            "Elevated estimated tortuosity. Confirm manually because fundus-based "
            "threshold vessel detection may overestimate tortuosity."
        )

    st.caption(
        "Note: This app provides image-derived screening estimates only. "
        "It is not a diagnostic medical device and should not replace clinical review. "
        "Primary intended signal: crowded / cupless optic disc anatomy."
    )
