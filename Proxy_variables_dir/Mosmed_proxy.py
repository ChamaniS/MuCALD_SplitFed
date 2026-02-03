import os
import cv2
import numpy as np
import pandas as pd
from glob import glob

def compute_mosmed_tags(image_path, mask_path):
    image = cv2.imread(image_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    binary_mask = (mask == 255).astype(np.uint8)
    lesion_size = int(np.sum(binary_mask))

    # Masked pixel locations
    indices = np.where(binary_mask == 1)

    # Mean intensity (grayscale)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mean_intensity = gray[indices].mean()

    # RGB stats
    r_vals = image_rgb[:, :, 0][indices]
    g_vals = image_rgb[:, :, 1][indices]
    b_vals = image_rgb[:, :, 2][indices]
    mean_r, mean_g, mean_b = np.mean(r_vals), np.mean(g_vals), np.mean(b_vals)
    std_r, std_g, std_b = np.std(r_vals), np.std(g_vals), np.std(b_vals)
    var_r, var_g, var_b = np.var(r_vals), np.var(g_vals), np.var(b_vals)

    color_entropy = -(np.histogram(r_vals, bins=256, range=(0, 256), density=True)[0] *
                      np.log2(np.histogram(r_vals, bins=256, range=(0, 256), density=True)[0] + 1e-9)).sum()

    # HSV stats
    h_vals = image_hsv[:, :, 0][indices]
    s_vals = image_hsv[:, :, 1][indices]
    v_vals = image_hsv[:, :, 2][indices]
    mean_h, mean_s, mean_v = np.mean(h_vals), np.mean(s_vals), np.mean(v_vals)

    # Color diversity = std of all RGB channels
    color_diversity = np.mean([std_r, std_g, std_b])

    # Border / contour metrics
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        main_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(main_contour)
        perimeter = cv2.arcLength(main_contour, True)

        compactness = (perimeter ** 2) / (4 * np.pi * area + 1e-5)

        x, y, w, h = cv2.boundingRect(main_contour)
        bbox_size = w * h

        if len(main_contour) >= 5:
            _, _, angle = cv2.fitEllipse(main_contour)
        else:
            angle = 0.0

        hull = cv2.convexHull(main_contour)
        hull_area = cv2.contourArea(hull)
        solidity = float(area) / (hull_area + 1e-5)

        M = cv2.moments(main_contour)
        cx = int(M['m10'] / (M['m00'] + 1e-5))
        cy = int(M['m01'] / (M['m00'] + 1e-5))
    else:
        area = perimeter = compactness = bbox_size = angle = solidity = cx = cy = 0

    # Asymmetry
    vert_flip_diff = np.sum(np.abs(binary_mask - np.flipud(binary_mask)))
    horiz_flip_diff = np.sum(np.abs(binary_mask - np.fliplr(binary_mask)))
    asymmetry = float(vert_flip_diff + horiz_flip_diff) / (lesion_size + 1e-5)

    return {
        "Image": os.path.basename(image_path),
        "Lesion_Size": lesion_size,
        "Perimeter": perimeter,
        "Compactness": compactness,
        "BoundingBox_Area": bbox_size,
        "Orientation": angle,
        "Solidity": solidity,
        "Centroid_X": cx,
        "Centroid_Y": cy,
        "Asymmetry": asymmetry,
        "Mean_Intensity": mean_intensity,
        "Mean_RGB": np.mean([mean_r, mean_g, mean_b]),
        "Mean_HSV": np.mean([mean_h, mean_s, mean_v]),
        "Color_Std": color_diversity,
        "Color_Entropy": color_entropy,
        "Color_Variance": np.mean([var_r, var_g, var_b])
    }

def compute_mosmed_tags_old(image_path, mask_path):
    # Load grayscale CT image and binary mask
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    # Binary mask (1 = lesion, 0 = background)
    binary_mask = (mask > 0).astype(np.uint8)

    # Lesion area = number of white pixels
    lesion_area = int(np.sum(binary_mask))

    # Lesion mean intensity (in the original CT image)
    if lesion_area > 0:
        mean_intensity = float(np.mean(image[binary_mask == 1]))
    else:
        mean_intensity = 0.0

    # Contour-based metrics: shape complexity & centroid
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)
        perimeter = cv2.arcLength(largest_contour, True)

        # Shape complexity: 4π * Area / Perimeter^2
        shape_complexity = (4 * np.pi * area) / (perimeter ** 2 + 1e-5)

        # Centroid: (cx, cy)
        M = cv2.moments(largest_contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            cx, cy = 0, 0
    else:
        shape_complexity = 0.0
        cx, cy = 0, 0

    return {
        "Image": os.path.basename(image_path),
        "Lesion_Area": lesion_area,
        "Mean_Intensity": mean_intensity,
        "Shape_Complexity": shape_complexity,
        "Centroid_X": cx,
        "Centroid_Y": cy
    }


def process_split(split_name, base_path):
    img_dir = os.path.join(base_path, f"{split_name}_imgs")
    mask_dir = os.path.join(base_path, f"{split_name}_masks")

    img_paths = sorted(glob(os.path.join(img_dir, "*.png")))
    mask_paths = sorted(glob(os.path.join(mask_dir, "*.png")))

    results = []
    for img_path, mask_path in zip(img_paths, mask_paths):
        result = compute_mosmed_tags(img_path, mask_path)
        results.append(result)

    df = pd.DataFrame(results)
    df.to_csv(f"mosmed_proxy_tags_{split_name}.csv", index=False)
    print(f"Saved: mosmed_proxy_tags_{split_name}.csv")


# Base directory setup
base_path = "XXXXX/Projects/Data/mosmedplus/mosmedplus_centralized"

# Run for all splits
for split in ["train", "val", "test"]:
    process_split(split, base_path)
