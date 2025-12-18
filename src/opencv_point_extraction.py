from __future__ import annotations

from pathlib import Path
import json

import cv2
import numpy as np
import pandas as pd


def _auto_detect_plot_area(img_bgr: np.ndarray):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    lower_black = np.array([0, 0, 0])
    upper_black = np.array([180, 255, 110])
    mask_black = cv2.inRange(hsv, lower_black, upper_black)

    kernel = np.ones((5, 5), np.uint8)
    mask_black = cv2.morphologyEx(mask_black, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(mask_black, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    H, W = img_bgr.shape[:2]
    best = None
    best_area = 0

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        if area <= best_area:
            continue
        if w < 100 or h < 100:
            continue
        if w > 0.98 * W or h > 0.98 * H:
            continue
        best_area = area
        best = (x, y, x + w, y + h)  # x_min, y_top, x_max, y_bottom

    if best is None:
        raise ValueError("Could not auto-detect plot area; provide plot_area_pixels manually.")
    return best


def digitize_raman_spectrum(
    image_path: str | Path,
    llm_params_json: str | Path,
    output_csv: str | Path = "raman1.csv",
):
    params = json.loads(Path(llm_params_json).read_text(encoding="utf-8"))

    x_min = float(params["x_axis"]["min_value"])
    x_max = float(params["x_axis"]["max_value"])

    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Could not load image at {image_path}")

    pa = params.get("plot_area_pixels") or {}
    if any(pa.get(k) is None for k in ["x_min", "x_max", "y_top", "y_bottom"]):
        px_x_min, px_y_top, px_x_max, px_y_bottom = _auto_detect_plot_area(img)
        plot_area_source = "opencv_auto"
    else:
        px_x_min = int(pa["x_min"])
        px_x_max = int(pa["x_max"])
        px_y_top = int(pa["y_top"])
        px_y_bottom = int(pa["y_bottom"])
        plot_area_source = "llm"

    H, W = img.shape[:2]
    px_x_min = max(0, min(px_x_min, W - 1))
    px_x_max = max(0, min(px_x_max, W))
    px_y_top = max(0, min(px_y_top, H - 1))
    px_y_bottom = max(0, min(px_y_bottom, H))

    if px_x_max <= px_x_min or px_y_bottom <= px_y_top:
        raise ValueError("Invalid plot_area_pixels after clamping")

    width = px_x_max - px_x_min
    height = px_y_bottom - px_y_top

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower_blue = np.array([100, 100, 50])
    upper_blue = np.array([140, 255, 255])

    mask_curve = cv2.inRange(hsv, lower_blue, upper_blue)
    roi = mask_curve[px_y_top:px_y_bottom, px_x_min:px_x_max]

    kernel = np.ones((3, 3), np.uint8)
    roi = cv2.morphologyEx(roi, cv2.MORPH_CLOSE, kernel, iterations=1)

    data_points = []

    x_den = max(width - 1, 1)
    y_den = max(height - 1, 1)

    for col in range(width):
        ys = np.where(roi[:, col] > 0)[0]
        if ys.size == 0:
            continue

        y_local = float(np.median(ys))

        data_x = (col / x_den) * (x_max - x_min) + x_min
        intensity_norm = (y_den - y_local) / y_den

        data_points.append((data_x, intensity_norm))

    df = pd.DataFrame(data_points, columns=["Raman_Shift_cm-1", "Intensity_norm"])
    df.to_csv(output_csv, index=False)

    print(f"[digitize] image: {image_path}")
    print(f"[digitize] params: {llm_params_json}")
    print(f"[digitize] plot_area source: {plot_area_source}")
    print(f"[digitize] plot_area: x[{px_x_min}:{px_x_max}] y[{px_y_top}:{px_y_bottom}]")
    print(f"[digitize] axis: X[{x_min}..{x_max}]")
    print(f"[digitize] points: {len(df)}")
    print(f"[digitize] saved: {output_csv}")
    return df

if __name__ == "__main__":
    output_path = "DIGITIZATION/OpenCV/raman1.csv"
    digitize_raman_spectrum(
        "/Users/prabhleenkaur/Desktop/Picasso/Chart-Comprehension/models/lit-agent/DIGITIZATION/GT/all_pts/raman1.png",
        llm_params_json="DIGITIZATION/llm_params/raman1_limits.json",
        output_csv=output_path,
    )
