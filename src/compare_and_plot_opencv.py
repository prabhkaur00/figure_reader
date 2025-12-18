from __future__ import annotations

from pathlib import Path
import re

import numpy as np
import matplotlib.pyplot as plt


def read_xy_txt(path: str | Path):
    path = Path(path)
    xs, ys = [], []
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            parts = re.split(r"[,\s]+", s)
            if len(parts) < 2:
                continue
            try:
                x = float(parts[0])
                y = float(parts[1])
            except ValueError:
                continue
            xs.append(x)
            ys.append(y)

    x = np.asarray(xs, dtype=np.float64)
    y = np.asarray(ys, dtype=np.float64)
    if x.size:
        o = np.argsort(x)
        x, y = x[o], y[o]
    return x, y


def read_opencv_csv(path: str | Path):
    path = Path(path)
    xs, ys = [], []
    with path.open("r", encoding="utf-8", errors="replace") as f:
        header_skipped = False
        for line in f:
            s = line.strip()
            if not s:
                continue
            if not header_skipped and ("Raman_Shift" in s or "Intensity" in s):
                header_skipped = True
                continue

            parts = [p.strip() for p in s.split(",")]
            if len(parts) < 2:
                continue
            try:
                x = float(parts[0])
                y = float(parts[1])
            except ValueError:
                continue
            xs.append(x)
            ys.append(y)

    x = np.asarray(xs, dtype=np.float64)
    y = np.asarray(ys, dtype=np.float64)
    if x.size:
        o = np.argsort(x)
        x, y = x[o], y[o]
    return x, y


def normalize_01(y: np.ndarray):
    if y.size == 0:
        return y
    y0 = np.nanmin(y)
    y1 = np.nanmax(y)
    denom = y1 - y0
    if not np.isfinite(denom) or denom <= 0:
        return np.zeros_like(y)
    return (y - y0) / denom


def _interp_to_gt(x_gt: np.ndarray, x_pred: np.ndarray, y_pred: np.ndarray):
    if x_gt.size == 0 or x_pred.size == 0:
        return np.asarray([], dtype=np.float64), np.asarray([], dtype=np.float64), np.asarray([], dtype=bool)
    x_min = max(np.min(x_gt), np.min(x_pred))
    x_max = min(np.max(x_gt), np.max(x_pred))
    mask = (x_gt >= x_min) & (x_gt <= x_max)
    if not np.any(mask):
        return np.asarray([], dtype=np.float64), np.asarray([], dtype=np.float64), np.asarray([], dtype=bool)
    y_interp = np.interp(x_gt[mask], x_pred, y_pred)
    return x_gt[mask], y_interp, mask


def overlay_compare_opencv_vs_gt(
    opencv_csv_path: str | Path,
    gt_txt_path: str | Path,
    *,
    out_png: str | Path | None = None,
    show_interp: bool = True,
):
    x_gt, y_gt_raw = read_xy_txt(gt_txt_path)
    x_cv, y_cv_raw = read_opencv_csv(opencv_csv_path)

    # Normalize BOTH series to [0, 1] independently (arbitrary Raman intensity)
    y_gt = normalize_01(y_gt_raw)
    y_cv = normalize_01(y_cv_raw)

    # Compare on GT x-grid (over overlap only), using normalized values
    x_eval, y_cv_on_gt, mask = _interp_to_gt(x_gt, x_cv, y_cv)
    y_gt_eval = y_gt[mask] if mask.size else np.asarray([], dtype=np.float64)

    rmse = float(np.sqrt(np.mean((y_cv_on_gt - y_gt_eval) ** 2))) if x_eval.size else float("nan")
    mae = float(np.mean(np.abs(y_cv_on_gt - y_gt_eval))) if x_eval.size else float("nan")

    fig = plt.figure()
    plt.plot(x_gt, y_gt, linewidth=1.2, label=f"GT norm ({Path(gt_txt_path).stem})")
    plt.plot(x_cv, y_cv, linewidth=1.2, label=f"OpenCV norm ({Path(opencv_csv_path).stem})")

    if show_interp and x_eval.size:
        plt.plot(x_eval, y_cv_on_gt, linewidth=1.2, label="OpenCV norm interpolated to GT x")

    plt.xlabel("Raman shift (cm$^{-1}$)")
    plt.ylabel("Normalized intensity (0–1)")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.title(f"Overlay (normalized) + metrics (RMSE={rmse:.4g}, MAE={mae:.4g})")

    plt.xlim(np.min(x_gt), np.max(x_gt))
    plt.ylim(0.0, 1.0)
    plt.tight_layout()

    if out_png is None:
        out_png = Path(gt_txt_path).with_suffix("").name + "_opencv_overlay_norm.png"

    fig.savefig(Path(out_png), dpi=250, bbox_inches="tight")
    plt.show()
    plt.close(fig)

    print(f"Eval points (overlap): {x_eval.size}")
    print(f"RMSE (normalized): {rmse}")
    print(f"MAE  (normalized): {mae}")
    return out_png


overlay_compare_opencv_vs_gt(
    "/Users/prabhleenkaur/Desktop/Picasso/Chart-Comprehension/models/lit-agent/DIGITIZATION/OpenCV/raman1.csv",
    "/Users/prabhleenkaur/Desktop/Picasso/Chart-Comprehension/models/lit-agent/DIGITIZATION/GT/all_pts/raman1.txt",
    out_png="raman2_opencv_overlay.png",
)