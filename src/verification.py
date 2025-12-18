from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np


def read_xy_txt(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            parts = [p for p in s.replace(",", " ").split() if p]
            if len(parts) < 2:
                continue
            try:
                xs.append(float(parts[0]))
                ys.append(float(parts[1]))
            except ValueError:
                continue
    x = np.asarray(xs, dtype=np.float64)
    y = np.asarray(ys, dtype=np.float64)
    if x.size:
        order = np.argsort(x)
        x, y = x[order], y[order]
    return x, y


def read_digitized_csv(path: Path) -> Tuple[np.ndarray, np.ndarray]:
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
                xs.append(float(parts[0]))
                ys.append(float(parts[1]))
            except ValueError:
                continue
    x = np.asarray(xs, dtype=np.float64)
    y = np.asarray(ys, dtype=np.float64)
    if x.size:
        order = np.argsort(x)
        x, y = x[order], y[order]
    return x, y


def normalize_01(y: np.ndarray) -> np.ndarray:
    if y.size == 0:
        return y
    y0 = np.nanmin(y)
    y1 = np.nanmax(y)
    denom = y1 - y0
    if not np.isfinite(denom) or denom <= 0:
        return np.zeros_like(y)
    return (y - y0) / denom


def _interp_overlap(
    x_gt: np.ndarray, x_pred: np.ndarray, y_pred: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if x_gt.size == 0 or x_pred.size == 0:
        return (
            np.asarray([], dtype=np.float64),
            np.asarray([], dtype=np.float64),
            np.asarray([], dtype=bool),
        )
    x_min = max(np.min(x_gt), np.min(x_pred))
    x_max = min(np.max(x_gt), np.max(x_pred))
    mask = (x_gt >= x_min) & (x_gt <= x_max)
    if not np.any(mask):
        return (
            np.asarray([], dtype=np.float64),
            np.asarray([], dtype=np.float64),
            np.asarray([], dtype=bool),
        )
    y_interp = np.interp(x_gt[mask], x_pred, y_pred)
    return x_gt[mask], y_interp, mask


def verify_digitized_vs_gt(
    *,
    digitized_csv_path: Path,
    gt_txt_path: Path,
    out_dir: Path,
    image_stem: str,
    tool_name: str,
    show_interp: bool = True,
) -> Dict[str, float | int | Path]:
    x_gt, y_gt_raw = read_xy_txt(gt_txt_path)
    x_pred, y_pred_raw = read_digitized_csv(digitized_csv_path)

    y_gt = normalize_01(y_gt_raw)
    y_pred = normalize_01(y_pred_raw)

    x_eval, y_pred_on_gt, mask = _interp_overlap(x_gt, x_pred, y_pred)
    y_gt_eval = y_gt[mask] if mask.size else np.asarray([], dtype=np.float64)

    rmse = float(np.sqrt(np.mean((y_pred_on_gt - y_gt_eval) ** 2))) if x_eval.size else float("nan")
    mae = float(np.mean(np.abs(y_pred_on_gt - y_gt_eval))) if x_eval.size else float("nan")

    fig = plt.figure()
    plt.plot(x_gt, y_gt, linewidth=1.2, label=f"GT norm ({gt_txt_path.stem})")
    plt.plot(x_pred, y_pred, linewidth=1.2, label=f"{tool_name} digitized norm")
    if show_interp and x_eval.size:
        plt.plot(x_eval, y_pred_on_gt, linewidth=1.0, label=f"{tool_name} interp to GT x")
    plt.xlabel("Raman shift (cm$^{-1}$)")
    plt.ylabel("Normalized intensity (0–1)")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.title(f"{image_stem} verification ({tool_name})")
    if x_gt.size:
        plt.xlim(np.min(x_gt), np.max(x_gt))
    plt.ylim(0.0, 1.0)
    plt.tight_layout()

    out_dir.mkdir(parents=True, exist_ok=True)
    overlay_path = out_dir / f"{image_stem}_{tool_name}_verify.png"
    fig.savefig(overlay_path, dpi=250, bbox_inches="tight")
    plt.close(fig)

    return {
        "overlay_path": overlay_path,
        "rmse": rmse,
        "mae": mae,
        "overlap_points": int(x_eval.size),
    }
