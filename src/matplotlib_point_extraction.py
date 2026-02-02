from __future__ import annotations
from pathlib import Path
import json
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pandas as pd
import numpy as np

from src.hough_line_transform import get_pixel_coordinates

"""
Placeholder for a Matplotlib-based point extraction pipeline.

This module is intended to house deterministic digitization logic powered by
Matplotlib (e.g., interactive point selection, image sampling, etc.). The
Figure Reader CLI currently exposes a stub tool (`MatplotlibImageExtractor`)
that references this module conceptually, but the actual implementation is
not yet provided.

TODO:
    - Implement helpers that accept an image path and return digitized (x, y)
      pairs using Matplotlib-based techniques.
    - Provide functions mirroring the OpenCV interface so the agent can call
      them interchangeably.
"""

# TODO: Replace this placeholder with actual Matplotlib-based digitization logic.
# For now manually giving in X-axis and Y-axis limits to 
def digitize_with_matplotlib(
      image_path: str | Path,
      llm_params_json: str | Path,
      output_csv: str | Path = "raman_mpl.csv",
    ):

    params = json.loads(Path(llm_params_json).read_text(encoding="utf-8"))
    x_min = float(params["x_axis"]["min_value"])
    x_max = float(params["x_axis"]["max_value"])
    y_min = float(params.get("y_axis", {}).get("min_value", 0))
    y_max = float(params.get("y_axis", {}).get("max_value", 1))

    pixel_coordinates = get_pixel_coordinates(image_path)

    origin_x = pixel_coordinates["origin_x"]
    origin_y = pixel_coordinates["origin_y"]
    terminal_x = pixel_coordinates["terminal_x"]
    terminal_y = pixel_coordinates["terminal_y"]
    px_x_min = pixel_coordinates["px_x_min"]
    px_x_max = pixel_coordinates["px_x_max"]
    px_y_min = pixel_coordinates["px_y_min"]
    px_y_max = pixel_coordinates["px_y_max"]

    img = plt.imread(str(image_path))
    if img.max() > 1.0: # Handle 0-255 uint8 images
        img = img / 255.0
    
    H, W, _ = img.shape


    # Detect Blue Curve using NumPy - this is equivalent to OpenCV inRange masking
    hsv = mcolors.rgb_to_hsv(img[:, :, :3])
    lower_blue = (0.55, 0.3, 0.2)
    upper_blue = (0.75, 1.0, 1.0)

    mask = (
        (hsv[:, :, 0] >= lower_blue[0]) & (hsv[:, :, 0] <= upper_blue[0]) &
        (hsv[:, :, 1] >= lower_blue[1]) & (hsv[:, :, 1] <= upper_blue[1]) &
        (hsv[:, :, 2] >= lower_blue[2]) & (hsv[:, :, 2] <= upper_blue[2])
    )

    # Extract Points
    data_points = []
    roi_width = terminal_x - origin_x
    roi_height = origin_y - terminal_y

    for col_idx in range(origin_x, terminal_x):
        y_indices = np.where(mask[terminal_y:origin_y, col_idx])[0]
        # rel_x = (col_idx - px_x_min) / max(1, roi_width - 1)
        data_x = x_min + (col_idx - px_x_min)* (x_max - x_min) / (px_x_max - px_x_min)
        
        for y_local in y_indices:
            # y_local = np.median(y_indices)
            # intensity_norm = (roi_height - 1 - y_local) / max(1, roi_height - 1)
            p_y = y_local + terminal_y
            intensity_final = y_min + (p_y - px_y_min)* (y_max - y_min) / (px_y_max - px_y_min)
            
            data_points.append((data_x, intensity_final))

    # 6. Save and Return
    df = pd.DataFrame(data_points, columns=["Raman_Shift_cm-1", "Intensity_norm"])
    df.to_csv(output_csv, index=False)
    
    print(f"Digitized {len(df)} points using Matplotlib/NumPy.")
    return df


    raise NotImplementedError(
        "Matplotlib point extraction not implemented. "
        "Fill in src/matplotlib_point_extraction.py with real logic."
    )
