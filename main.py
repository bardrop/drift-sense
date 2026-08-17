"""
Drift-Sense Synthetic SEM Dataset Generator (IRDS 2024 FinFET Architecture)
Hackathon 2026 - Applied Materials Problem Statement 02
Uses correct pre-downscale optical physics to prevent aliasing.
"""

import os
import csv
import math
import random
import numpy as np
import cv2

# ==========================================
# CONFIGURATION
# ==========================================
NUM_PAIRS = 500
CANVAS_SIZE = 10000  # 10,000 x 10,000 nm (10 um x 10 um)
IMAGE_SIZE = 1000  # 1000 x 1000 pixels
DOWNSCALE_FACTOR = 10.0  # 10x magnification ratio

OUTPUT_DIR = "dataset_drift_sense"
REF_DIR = os.path.join(OUTPUT_DIR, "reference")
SEARCH_DIR = os.path.join(OUTPUT_DIR, "search")
MANIFEST_PATH = os.path.join(OUTPUT_DIR, "manifest.csv")

# FinFET Geometric Parameters (IRDS 2024 Table MM-7)
FIN_PITCH = 24
FIN_WIDTH = 8
GATE_PITCH = 48
GATE_WIDTH = 18


def generate_finfet_macro_die(size=CANVAS_SIZE):
    """
    Generates a realistic FinFET floorplan with standard cell rows
    and macro power boundaries, simulating continuous electron-optical lines.
    """
    # Base dark silicon substrate
    canvas = np.full((size, size), 40, dtype=np.uint8)

    num_blocks = 4
    street_width = 250
    block_pitch = size // num_blocks
    block_size = block_pitch - street_width

    for by in range(num_blocks):
        for bx in range(num_blocks):
            x0 = bx * block_pitch + street_width // 2
            y0 = by * block_pitch + street_width // 2
            x1 = x0 + block_size
            y1 = y0 + block_size

            # Draw Continuous Vertical Fins
            for fx in range(x0 + 10, x1 - 10, FIN_PITCH):
                canvas[y0:y1, fx : fx + FIN_WIDTH] = 90

            # Draw Continuous Horizontal Gates crossing the fins
            for gy in range(y0 + 10, y1 - 10, GATE_PITCH):
                canvas[gy : gy + GATE_WIDTH, x0:x1] = 140

            # Draw periodic Contact Vias (bright dots)
            for cy in range(y0 + int(GATE_PITCH * 1.5), y1 - 20, GATE_PITCH * 4):
                for cx in range(x0 + int(FIN_PITCH * 2.5), x1 - 20, FIN_PITCH * 4):
                    cv2.circle(canvas, (cx, cy), 8, 220, -1)

    # Add Macro Power Rails (The thick grid lines seen in 10x search)
    for b in range(1, num_blocks):
        coord = b * block_pitch
        cv2.line(canvas, (coord, 0), (coord, size), 180, 40)
        cv2.line(canvas, (0, coord), (size, coord), 180, 40)

    return canvas


def apply_edge_charging(img, thickness=2, boost=1.5):
    """Simulates secondary electron emission at feature edges."""
    k_size = thickness * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k_size, k_size))
    dilation = cv2.dilate(img, kernel)
    erosion = cv2.erode(img, kernel)
    gradient = cv2.subtract(dilation, erosion)
    return cv2.addWeighted(img, 1.0, gradient, boost, 0)


def apply_detector_noise(img, sigma):
    """Applies zero-mean Gaussian approximation of Poisson shot noise."""
    noise = np.random.normal(loc=0.0, scale=sigma, size=img.shape).astype(np.float32)
    return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def generate_dataset():
    os.makedirs(REF_DIR, exist_ok=True)
    os.makedirs(SEARCH_DIR, exist_ok=True)

    csv_header = [
        "pair_id",
        "reference_path",
        "search_path",
        "ground_truth_x",
        "ground_truth_y",
        "rotation_deg",
        "drift_x_nm",
        "drift_y_nm",
        "total_drift_nm",
        "blur_sigma",
        "ref_noise_sigma",
        "search_noise_sigma",
        "random_seed",
    ]
    manifest_rows = []

    print("Generating 30 realistic IRDS 2024 FinFET SEM image pairs...")

    for i in range(1, NUM_PAIRS + 1):
        seed = 1000 + i
        random.seed(seed)
        np.random.seed(seed)

        # 1. Build Virtual Wafer Canvas
        canvas_raw = generate_finfet_macro_die(CANVAS_SIZE)

        # 2. Apply Edge Charging FIRST (at 1x physical scale)
        edge_thickness = random.choice([2, 3])
        canvas_charged = apply_edge_charging(
            canvas_raw, thickness=edge_thickness, boost=1.5
        )

        # 3. Target Selection with Translational Drift (100 - 500 nm)
        base_corner = 2500
        drift_angle = random.uniform(0, 2 * math.pi)
        drift_dist = random.uniform(100.0, 500.0)
        drift_x = drift_dist * math.cos(drift_angle)
        drift_y = drift_dist * math.sin(drift_angle)

        target_x = base_corner + drift_x
        target_y = base_corner + drift_y

        # 4. Extract 100x Reference Crop
        crop_x0 = int(round(target_x - (IMAGE_SIZE / 2)))
        crop_y0 = int(round(target_y - (IMAGE_SIZE / 2)))
        ref_raw = canvas_charged[
            crop_y0 : crop_y0 + IMAGE_SIZE, crop_x0 : crop_x0 + IMAGE_SIZE
        ].copy()

        # Slight focal blur for the reference image
        ref_blurred = cv2.GaussianBlur(ref_raw, (3, 3), 0.8)

        # 5. Apply 1-2 Degree Stage Rotation
        rot_angle = random.uniform(1.0, 2.0) * random.choice([-1, 1])
        rot_center = (CANVAS_SIZE // 2, CANVAS_SIZE // 2)
        rot_mat = cv2.getRotationMatrix2D(rot_center, rot_angle, 1.0)

        rotated_canvas = cv2.warpAffine(
            canvas_charged,
            rot_mat,
            (CANVAS_SIZE, CANVAS_SIZE),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT,
        )

        # Track true top-left (0,0) center mapping
        target_pt = np.array([target_x, target_y, 1.0])
        transformed_pt = rot_mat.dot(target_pt)
        gt_x = float(transformed_pt[0] / DOWNSCALE_FACTOR)
        gt_y = float(transformed_pt[1] / DOWNSCALE_FACTOR)

        # 6. Optical Pre-Blur (Crucial to prevent 10x checkerboard aliasing)
        optical_blur_canvas = cv2.GaussianBlur(rotated_canvas, (9, 9), 2.5)

        # 7. Downscale to 10x Search Image (1000x1000 px, 10 nm/px)
        search_raw = cv2.resize(
            optical_blur_canvas, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA
        )

        # 8. Apply Final Beam Blur to Search Image ONLY (S.D 1.0 - 2.0)
        blur_sigma = random.uniform(1.0, 2.0)
        search_blurred = cv2.GaussianBlur(
            search_raw, (3, 3), sigmaX=blur_sigma, sigmaY=blur_sigma
        )

        # 9. Independent Detector Shot Noise (S.D 15 - 35) applied LAST
        ref_noise_sigma = random.uniform(15.0, 35.0)
        search_noise_sigma = random.uniform(15.0, 35.0)

        ref_final = apply_detector_noise(ref_blurred, ref_noise_sigma)
        search_final = apply_detector_noise(search_blurred, search_noise_sigma)

        # 10. Save Image Files
        ref_path = os.path.join(REF_DIR, f"ref_{i:03d}.png")
        search_path = os.path.join(SEARCH_DIR, f"search_{i:03d}.png")

        cv2.imwrite(ref_path, ref_final)
        cv2.imwrite(search_path, search_final)

        # 11. Write to Manifest
        manifest_rows.append(
            [
                i,
                ref_path,
                search_path,
                round(gt_x, 4),
                round(gt_y, 4),
                round(rot_angle, 3),
                round(drift_x, 2),
                round(drift_y, 2),
                round(drift_dist, 2),
                round(blur_sigma, 3),
                round(ref_noise_sigma, 2),
                round(search_noise_sigma, 2),
                seed,
            ]
        )

        print(f"Generated Pair {i:02d}/{NUM_PAIRS} | Error logged successfully.")

    with open(MANIFEST_PATH, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(csv_header)
        writer.writerows(manifest_rows)

    print("Generation Complete! Ready for localization testing.")


if __name__ == "__main__":
    generate_dataset()
