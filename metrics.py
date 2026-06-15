import cv2
import numpy as np
from skimage.morphology import skeletonize
from skimage.measure import label, regionprops

def extract_vessels(image_rgb, mask_disc):
    """Processes image to extract vessel map using CLAHE and adaptive thresholding."""
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
    """Calculates median vessel tortuosity based on arc-chord ratio."""
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

def calculate_full_metrics(mask_disc, mask_cup, image_rgb):
    """Orchestrates all diagnostic metric calculations."""
    results = {}
    disc_coords = np.argwhere(mask_disc > 0)
    if len(disc_coords) == 0: return None

    min_y, max_y = np.min(disc_coords[:, 0]), np.max(disc_coords[:, 0])
    disc_height = max_y - min_y
    
    cup_coords = np.argwhere(mask_cup > 0)
    if len(cup_coords) > 0:
        c_min_y, c_max_y = np.min(cup_coords[:, 0]), np.max(cup_coords[:, 0])
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

def crowded_disc_triage(metrics):
    """Business logic for automated clinical risk assessment."""
    vcdr, density = metrics["vcdr"], metrics["density"]
    thick_rim = (metrics["rim_s_ratio"] > 0.20) and (metrics["rim_i_ratio"] > 0.20)

    if vcdr < 0.05: return "High likelihood: Crowded / cupless disc", "high"
    if vcdr < 0.20 and thick_rim: return "Likely crowded disc anatomy", "high"
    if vcdr < 0.20: return "Possible crowded disc anatomy", "moderate"
    if vcdr < 0.30 and density > 0.18: return "Borderline small cup; review recommended", "moderate"
    return "Not crowded by cup-disc anatomy", "low"
