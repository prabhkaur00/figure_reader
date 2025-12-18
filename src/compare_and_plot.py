from __future__ import annotations

from pathlib import Path
import json
import re

import numpy as np
import matplotlib.pyplot as plt


def read_xy_txt(path: str | Path):
    path = Path(path)
    xs, ys = [], []
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            if s.startswith("#"):
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

def read_vlm_json_points(
    json_path: str | Path,
    *,
    figure_id: str | None = None,
    subplot_id: str | None = None,
    curve_label: str | None = None,
):
    obj = json.loads(Path(json_path).read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ValueError("JSON top-level must be an object with key 'figures'")

    figs = obj.get("figures", [])
    if not figs:
        raise ValueError("No 'figures' found in JSON")

    if figure_id is None:
        fig = figs[0]
    else:
        fig = next((f for f in figs if str(f.get("figure_id")) == str(figure_id)), None)
        if fig is None:
            available = [f.get("figure_id") for f in figs]
            raise ValueError(f"figure_id={figure_id} not found. available={available}")

    subplots = fig.get("subplots", [])
    if not subplots:
        raise ValueError("No 'subplots' found in selected figure")

    if subplot_id is None:
        sp = subplots[0]
    else:
        sp = next((s for s in subplots if str(s.get("subplot_id")) == str(subplot_id)), None)
        if sp is None:
            available = [s.get("subplot_id") for s in subplots]
            raise ValueError(f"subplot_id={subplot_id} not found. available={available}")

    curves = sp.get("curves", [])
    if not curves:
        raise ValueError("No 'curves' found in selected subplot")

    if curve_label is None:
        curve = curves[0]
    else:
        curve = next((c for c in curves if c.get("label") == curve_label), None)
        if curve is None:
            available = [c.get("label") for c in curves]
            raise ValueError(f"curve_label={curve_label!r} not found. available={available}")

    pts = curve.get("points", [])
    if not pts:
        raise ValueError("Selected curve has no 'points'")

    xs = np.asarray([p["x"] for p in pts], dtype=np.float64)
    ys = np.asarray([p["y"] for p in pts], dtype=np.float64)

    if xs.size:
        o = np.argsort(xs)
        xs, ys = xs[o], ys[o]

    axes = sp.get("axes", None)
    fig_id = str(fig.get("figure_id", "figure"))
    return xs, ys, curve.get("label", "curve"), axes, fig_id


def overlay_plot(
    vlm_json_path: str | Path,
    gt_txt_path: str | Path,
    *,
    out_png: str | Path | None = None,
    figure_id: str | None = None,
    subplot_id: str | None = None,
    curve_label: str | None = None,
):
    x_gt, y_gt = read_xy_txt(gt_txt_path)
    x_v, y_v, v_label, axes, fig_id = read_vlm_json_points(
        vlm_json_path,
        figure_id=figure_id,
        subplot_id=subplot_id,
        curve_label=curve_label,
    )

    fig = plt.figure()
    plt.plot(x_gt, y_gt, linewidth=1.2, label=Path(gt_txt_path).stem)
    plt.plot(x_v, y_v, linewidth=1.2, label=f"VLM ({v_label})")
    plt.xlabel("Raman shift (cm$^{-1}$)")
    plt.ylabel("Intensity (a.u.)")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.title(f"{fig_id} overlay")

    if axes and "x_range" in axes:
        plt.xlim(axes["x_range"][0], axes["x_range"][1])
    if axes and "y_range" in axes:
        plt.ylim(axes["y_range"][0], axes["y_range"][1])

    plt.tight_layout()

    if out_png is None:
        out_png = Path(gt_txt_path).with_suffix("").name + "_overlay.png"

    fig.savefig(Path(out_png), dpi=250, bbox_inches="tight")
    plt.show()
    plt.close(fig)
    return out_png


# --- usage ---
overlay_plot(
    "/Users/prabhleenkaur/Desktop/Picasso/Chart-Comprehension/models/lit-agent/DIGITIZATION/Gemini/raman1.json",
    "/Users/prabhleenkaur/Desktop/Picasso/Chart-Comprehension/models/lit-agent/DIGITIZATION/GT/raman1.txt",
    out_png="raman1_overlay.png",
)